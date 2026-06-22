"""한국어 SentencePiece Transformer용 대화형 추론 CLI.

현재 디렉터리의 SentencePiece 모델과 ``histories``의 체크포인트를
파일명 ID로 연결하여 한국어 Transformer 구조를 복원한다.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import signal
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

# sentencepiece 0.1.x의 protobuf 생성 코드가 최신 protobuf에서도 동작하도록 한다.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentencepiece import sentencepiece_model_pb2


BASE_DIR = Path(__file__).resolve().parent
EXIT_COMMANDS = {"종료", "quit", "exit", "q"}
PREPROCESS_PROFILE = "korean_chatbot_v1"
ARCHITECTURE_NAME = "korean_transformer_v1"


class ChatConfigurationError(RuntimeError):
    """추론 파일이나 설정의 호환성이 맞지 않을 때 발생한다."""


class ChatInputError(ValueError):
    """현재 모델에서 처리할 수 없는 사용자 입력일 때 발생한다."""


@dataclass(frozen=True)
class ModelProfile:
    """자동으로 발견한 tokenizer/checkpoint 조합."""

    tokenizer_id: str
    tokenizer_path: Path
    checkpoint_path: Path
    preprocess_profile: str = PREPROCESS_PROFILE
    architecture: str = ARCHITECTURE_NAME


def tokenizer_id_from_path(path: Path) -> str:
    """파일명에서 tokenizer/checkpoint 연결에 사용할 ID를 반환한다."""

    return path.stem.removeprefix("spm_")


def discover_model_profiles(
    base_dir: Path = BASE_DIR,
    history_dir: Path | None = None,
) -> list[ModelProfile]:
    """``spm_<id>.model``과 파일명에 ``<id>``가 포함된 checkpoint를 찾는다."""

    history_dir = history_dir or base_dir / "histories"
    tokenizers = {
        tokenizer_id_from_path(path): path
        for path in sorted(base_dir.glob("spm_*.model"))
        if tokenizer_id_from_path(path)
    }
    profiles: list[ModelProfile] = []

    for checkpoint_path in sorted(history_dir.glob("*.pt")):
        matches = [
            tokenizer_id
            for tokenizer_id in tokenizers
            if tokenizer_id in checkpoint_path.stem
        ]
        if not matches:
            continue

        longest_length = max(map(len, matches))
        longest_matches = sorted(
            tokenizer_id for tokenizer_id in matches if len(tokenizer_id) == longest_length
        )
        if len(longest_matches) > 1:
            ambiguous = ", ".join(longest_matches)
            raise ChatConfigurationError(
                f"checkpoint 파일명이 여러 tokenizer ID와 일치합니다: "
                f"{checkpoint_path.name} ({ambiguous})"
            )

        tokenizer_id = longest_matches[0]
        profiles.append(
            ModelProfile(
                tokenizer_id=tokenizer_id,
                tokenizer_path=tokenizers[tokenizer_id],
                checkpoint_path=checkpoint_path,
            )
        )

    return sorted(
        profiles,
        key=lambda profile: (profile.tokenizer_id, profile.checkpoint_path.name),
    )


def preprocess_sentence(sentence: str) -> str:
    """노트북에서 한국어 챗봇 학습에 사용한 전처리를 재현한다."""

    sentence = unicodedata.normalize("NFC", str(sentence))

    # 현재 두 노트북의 저장된 학습 결과에서는 구두점이 공백으로 치환되었다.
    # 추론에서도 학습 분포와 맞추기 위해 같은 결과를 만든다.
    sentence = re.sub(r"([?.!,])", " ", sentence)
    sentence = re.sub(r"[^가-힣a-zA-Z0-9?.!,]+", " ", sentence)
    return re.sub(r"\s+", " ", sentence).strip()


PREPROCESSORS: dict[str, Callable[[str], str]] = {
    PREPROCESS_PROFILE: preprocess_sentence,
}


def create_padding_mask(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = (x == pad_id).float()
    return mask.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    seq_len = x.size(1)
    look_ahead_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=x.device),
        diagonal=1,
    )
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)
    padding_mask = create_padding_mask(x, pad_id=pad_id)
    return torch.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, position: int, d_model: int):
        super().__init__()
        self.position = position
        self.d_model = d_model
        self.register_buffer("pos_encoding", self._build_pos_encoding(position, d_model))

    @staticmethod
    def _build_pos_encoding(position: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        angle_rads = pos / torch.pow(10000, (2 * (i // 2)) / d_model)

        pos_encoding = torch.zeros(position, d_model)
        pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        return pos_encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.position:
            raise ChatInputError(
                f"sequence length {seq_len} exceeds positional encoding length {self.position}"
            )
        return x + self.pos_encoding[:, :seq_len, :]


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    matmul_qk = torch.matmul(query, key.transpose(-1, -2))
    logits = matmul_qk / math.sqrt(key.size(-1))

    if mask is not None:
        logits = logits + (mask * -1e9)

    attention_weights = F.softmax(logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ChatConfigurationError(
                f"d_model({d_model})은 num_heads({num_heads})로 나누어 떨어져야 합니다."
            )

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)
        self.out_dense = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        query = self.split_heads(self.query_dense(query), batch_size)
        key = self.split_heads(self.key_dense(key), batch_size)
        value = self.split_heads(self.value_dense(value), batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        return self.out_dense(concat_attention)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output = self.dropout1(self.mha(x, x, x, mask))
        out1 = self.norm1(x + attn_output)
        ffn_output = self.dropout2(self.ffn(out1))
        return self.norm2(out1 + ffn_output)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.encdec_mha = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_outputs: torch.Tensor,
        look_ahead_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_attn_out = self.dropout1(self.self_mha(x, x, x, mask=look_ahead_mask))
        out1 = self.norm1(x + self_attn_out)
        encdec_attn_out = self.dropout2(
            self.encdec_mha(out1, enc_outputs, enc_outputs, mask=padding_mask)
        )
        out2 = self.norm2(out1 + encdec_attn_out)
        ffn_out = self.dropout3(self.ffn(out2))
        return self.norm3(out2 + ffn_out)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        ff_dim: int,
        d_model: int,
        num_heads: int,
        max_position: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(max_position, d_model)
        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoding(x))
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        ff_dim: int,
        d_model: int,
        num_heads: int,
        max_position: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(max_position, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        enc_outputs: torch.Tensor,
        look_ahead_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoding(x))
        for layer in self.dec_layers:
            x = layer(x, enc_outputs, look_ahead_mask, padding_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        units: int,
        d_model: int,
        num_heads: int,
        max_encoder_position: int,
        max_decoder_position: int,
        dropout: float,
        pad_id: int,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_encoder_position = max_encoder_position
        self.max_decoder_position = max_decoder_position
        self.encoder = Encoder(
            vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            max_encoder_position,
            dropout,
            pad_id,
        )
        self.decoder = Decoder(
            vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            max_decoder_position,
            dropout,
            pad_id,
        )
        self.final_linear = nn.Linear(d_model, vocab_size)

    def forward(self, inputs: torch.Tensor, dec_inputs: torch.Tensor) -> torch.Tensor:
        enc_padding_mask = create_padding_mask(inputs, self.pad_id)
        look_ahead_mask = create_look_ahead_mask(dec_inputs, self.pad_id)
        dec_padding_mask = create_padding_mask(inputs, self.pad_id)
        enc_outputs = self.encoder(inputs, mask=enc_padding_mask)
        dec_outputs = self.decoder(
            dec_inputs,
            enc_outputs,
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask,
        )
        return self.final_linear(dec_outputs)


MODEL_BUILDERS = {ARCHITECTURE_NAME: Transformer}


@dataclass
class ChatRuntime:
    profile: ModelProfile
    model: Transformer
    tokenizer: spm.SentencePieceProcessor
    preprocessor: Callable[[str], str]
    config: Mapping[str, Any]
    device: torch.device


def tokenizer_model_type(tokenizer_path: Path) -> str:
    try:
        model_proto = sentencepiece_model_pb2.ModelProto()
        model_proto.ParseFromString(tokenizer_path.read_bytes())
        enum_value = model_proto.trainer_spec.model_type
        return sentencepiece_model_pb2.TrainerSpec.ModelType.Name(enum_value).lower()
    except Exception as exc:
        raise ChatConfigurationError(
            f"SentencePiece 종류를 확인할 수 없습니다: {tokenizer_path}\n원인: {exc}"
        ) from exc


def require_files(tokenizer_path: Path, checkpoint_path: Path) -> None:
    missing = [path for path in (tokenizer_path, checkpoint_path) if not path.is_file()]
    if not missing:
        return

    missing_text = "\n".join(f"  - {path}" for path in missing)
    raise ChatConfigurationError(
        f"추론에 필요한 파일이 없습니다:\n{missing_text}\n"
        "tokenizer와 checkpoint는 같은 학습 실행에서 만들어진 조합을 사용하세요."
    )


def load_checkpoint(checkpoint_path: Path) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any], Mapping[str, Any]]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="TypedStorage is deprecated.*",
                category=UserWarning,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        # weights_only 인자가 없는 구버전 PyTorch 호환 경로이다.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise ChatConfigurationError(
            f"체크포인트를 읽을 수 없습니다: {checkpoint_path}\n원인: {exc}"
        ) from exc

    if not isinstance(checkpoint, Mapping):
        raise ChatConfigurationError("체크포인트 최상위 객체는 dictionary여야 합니다.")
    if "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ChatConfigurationError(
            "체크포인트에 'model_state_dict'와 'config'가 모두 필요합니다. "
            "가중치만 저장한 state_dict로는 모델 구조를 복원할 수 없습니다."
        )

    state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    metadata = checkpoint.get("metadata", {})
    if not isinstance(state_dict, Mapping) or not isinstance(config, Mapping):
        raise ChatConfigurationError("model_state_dict와 config는 dictionary여야 합니다.")
    if not isinstance(metadata, Mapping):
        raise ChatConfigurationError("선택적인 metadata는 dictionary여야 합니다.")
    return state_dict, config, metadata


def validate_config(config: Mapping[str, Any]) -> None:
    required_keys = {
        "num_layers",
        "d_model",
        "num_heads",
        "units",
        "dropout",
        "vocab_size",
        "max_encoder_position",
        "max_decoder_position",
        "max_generation_length",
        "pad_id",
        "bos_id",
        "eos_id",
        "unk_id",
    }
    missing = sorted(required_keys - set(config))
    if missing:
        raise ChatConfigurationError(
            "체크포인트 config에 필요한 값이 없습니다: " + ", ".join(missing)
        )

    positive_keys = (
        "num_layers",
        "d_model",
        "num_heads",
        "units",
        "vocab_size",
        "max_encoder_position",
        "max_decoder_position",
        "max_generation_length",
    )
    for key in positive_keys:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ChatConfigurationError(f"config['{key}']는 양의 정수여야 합니다.")
    if config["d_model"] % config["num_heads"] != 0:
        raise ChatConfigurationError("d_model은 num_heads로 나누어 떨어져야 합니다.")


def metadata_value(
    key: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Any:
    return metadata.get(key, config.get(key))


def validate_tokenizer(
    profile: ModelProfile,
    tokenizer: spm.SentencePieceProcessor,
    actual_type: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    if tokenizer.get_piece_size() != config["vocab_size"]:
        raise ChatConfigurationError(
            "vocab 크기가 맞지 않습니다: "
            f"tokenizer={tokenizer.get_piece_size()}, checkpoint={config['vocab_size']}"
        )

    special_ids = {
        "pad_id": tokenizer.pad_id(),
        "bos_id": tokenizer.bos_id(),
        "eos_id": tokenizer.eos_id(),
        "unk_id": tokenizer.unk_id(),
    }
    for key, actual_id in special_ids.items():
        if actual_id != config[key]:
            raise ChatConfigurationError(
                f"{key}가 맞지 않습니다: tokenizer={actual_id}, checkpoint={config[key]}"
            )

    tokenizer_type_meta = metadata_value("tokenizer_type", config, metadata)
    if tokenizer_type_meta is not None and str(tokenizer_type_meta).lower() != actual_type:
        raise ChatConfigurationError(
            "체크포인트 tokenizer_type 메타데이터가 tokenizer 파일과 맞지 않습니다."
        )

    preprocess_meta = metadata_value("preprocess_profile", config, metadata)
    if preprocess_meta is not None and preprocess_meta != profile.preprocess_profile:
        raise ChatConfigurationError(
            "체크포인트의 preprocess_profile이 선택한 프로필과 맞지 않습니다."
        )


def validate_state_shapes(
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> None:
    expected_shapes = {
        "encoder.embedding.weight": (config["vocab_size"], config["d_model"]),
        "decoder.embedding.weight": (config["vocab_size"], config["d_model"]),
        "final_linear.weight": (config["vocab_size"], config["d_model"]),
    }
    for key, expected_shape in expected_shapes.items():
        tensor = state_dict.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ChatConfigurationError(f"state_dict에 '{key}' tensor가 없습니다.")
        if tuple(tensor.shape) != expected_shape:
            raise ChatConfigurationError(
                f"{key} shape이 맞지 않습니다: "
                f"state_dict={tuple(tensor.shape)}, config={expected_shape}"
            )


def build_runtime(
    profile: ModelProfile,
    tokenizer_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> ChatRuntime:
    require_files(tokenizer_path, checkpoint_path)

    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.load(str(tokenizer_path)):
        raise ChatConfigurationError(f"SentencePiece 모델을 불러오지 못했습니다: {tokenizer_path}")

    actual_type = tokenizer_model_type(tokenizer_path)

    state_dict, config, metadata = load_checkpoint(checkpoint_path)
    validate_config(config)
    validate_tokenizer(
        profile,
        tokenizer,
        actual_type,
        config,
        metadata,
    )
    validate_state_shapes(state_dict, config)

    architecture = metadata_value("architecture", config, metadata) or profile.architecture
    if architecture != profile.architecture:
        raise ChatConfigurationError(
            f"지원하지 않는 architecture입니다: {architecture} "
            f"(필요: {profile.architecture})"
        )
    model_builder = MODEL_BUILDERS.get(architecture)
    if model_builder is None:
        raise ChatConfigurationError(f"등록되지 않은 model builder입니다: {architecture}")
    preprocessor = PREPROCESSORS.get(profile.preprocess_profile)
    if preprocessor is None:
        raise ChatConfigurationError(
            f"등록되지 않은 preprocess_profile입니다: {profile.preprocess_profile}"
        )

    model = model_builder(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        units=config["units"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        max_encoder_position=config["max_encoder_position"],
        max_decoder_position=config["max_decoder_position"],
        dropout=config["dropout"],
        pad_id=config["pad_id"],
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ChatConfigurationError(
            f"체크포인트 구조가 {architecture} 모델과 맞지 않습니다.\n원인: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return ChatRuntime(profile, model, tokenizer, preprocessor, config, device)


def predict(runtime: ChatRuntime, sentence: str) -> str:
    sentence = runtime.preprocessor(sentence)
    if not sentence:
        raise ChatInputError("전처리 후 남은 한글·영문·숫자가 없습니다.")

    encoder_ids = runtime.tokenizer.encode_as_ids(sentence)
    if not encoder_ids:
        raise ChatInputError("tokenizer가 입력을 token으로 변환하지 못했습니다.")
    if len(encoder_ids) > runtime.model.max_encoder_position:
        raise ChatInputError(
            f"입력 token 길이({len(encoder_ids)})가 encoder position 길이"
            f"({runtime.model.max_encoder_position})보다 깁니다."
        )

    encoder_input = torch.tensor([encoder_ids], dtype=torch.long, device=runtime.device)
    decoder_input = torch.tensor(
        [[runtime.config["bos_id"]]], dtype=torch.long, device=runtime.device
    )
    generated_ids: list[int] = []

    with torch.no_grad():
        for _ in range(runtime.config["max_generation_length"]):
            if decoder_input.size(1) > runtime.model.max_decoder_position:
                break

            logits = runtime.model(encoder_input, decoder_input)
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())

            if next_id == runtime.config["eos_id"]:
                break
            if next_id in {runtime.config["pad_id"], runtime.config["bos_id"]}:
                break

            generated_ids.append(next_id)
            next_token = torch.tensor([[next_id]], dtype=torch.long, device=runtime.device)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

    return runtime.tokenizer.decode_ids(generated_ids)


def select_profile(profiles: list[ModelProfile]) -> ModelProfile:
    """번호 입력으로 사용할 tokenizer/checkpoint 조합을 선택한다."""

    if not profiles:
        raise ChatConfigurationError(
            f"실행 가능한 모델 조합이 없습니다. {BASE_DIR.name}/spm_<id>.model과 "
            "histories/의 <id>가 포함된 .pt 파일을 확인하세요."
        )

    print("사용할 모델/tokenizer 조합을 선택하세요.")
    for number, profile in enumerate(profiles, start=1):
        print(
            f"{number}. [{profile.tokenizer_id}] "
            f"{profile.tokenizer_path.name} + {profile.checkpoint_path.name}"
        )

    while True:
        try:
            selection = input(f"선택 (1-{len(profiles)}): ").strip()
        except EOFError as exc:
            raise ChatConfigurationError("모델 선택 입력이 종료되었습니다.") from exc

        if selection.isdigit():
            index = int(selection) - 1
            if 0 <= index < len(profiles):
                return profiles[index]

        print(f"1부터 {len(profiles)} 사이의 번호를 입력하세요.")


def run_chat(runtime: ChatRuntime) -> None:
    print("\n독립 질문-답변 모드입니다. 이전 대화 내용은 다음 입력에 전달되지 않습니다.")
    print("종료하려면 '종료', 'quit', 'exit', 'q' 중 하나를 입력하세요.\n")

    while True:
        try:
            sentence = input("나: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n대화를 종료합니다.")
            break

        if not sentence:
            continue
        if sentence.lower() in EXIT_COMMANDS:
            print("대화를 종료합니다.")
            break

        try:
            answer = predict(runtime, sentence)
        except ChatInputError as exc:
            print(f"입력 오류: {exc}")
            continue

        print("챗봇:", answer if answer else "(빈 응답)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="한국어 SentencePiece Transformer 대화형 추론"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="번호 선택을 건너뛰고 사용할 SentencePiece .model 경로입니다.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="번호 선택을 건너뛰고 사용할 PyTorch .pt 경로입니다.",
    )
    args = parser.parse_args()
    if (args.tokenizer_path is None) != (args.checkpoint_path is None):
        parser.error("--tokenizer-path와 --checkpoint-path는 함께 지정해야 합니다.")
    return args


def handle_sigint(_signum: int, _frame: Any) -> None:
    print("\n대화를 종료합니다.")
    raise SystemExit(0)


def main() -> int:
    signal.signal(signal.SIGINT, handle_sigint)
    args = parse_args()

    try:
        if args.tokenizer_path is not None:
            tokenizer_path = args.tokenizer_path.expanduser().resolve()
            checkpoint_path = args.checkpoint_path.expanduser().resolve()
            profile = ModelProfile(
                tokenizer_id=tokenizer_id_from_path(tokenizer_path),
                tokenizer_path=tokenizer_path,
                checkpoint_path=checkpoint_path,
            )
        else:
            profile = select_profile(discover_model_profiles())
            tokenizer_path = profile.tokenizer_path.resolve()
            checkpoint_path = profile.checkpoint_path.resolve()
    except ChatConfigurationError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        runtime = build_runtime(
            profile,
            tokenizer_path,
            checkpoint_path,
            device,
        )
    except ChatConfigurationError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 1

    print(f"tokenizer ID: {profile.tokenizer_id}")
    print(f"device: {device}")
    print(f"tokenizer: {tokenizer_path}")
    print(f"checkpoint: {checkpoint_path}")
    print("호환성 검사: 구조·vocab·특수 token이 일치합니다.")

    run_chat(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
