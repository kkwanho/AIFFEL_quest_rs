from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
HISTORY_DIR = BASE_DIR / "outputs" / "tokenizer_comparison" / "histories"
FIGURE_DIR = BASE_DIR / "outputs" / "tokenizer_comparison" / "figures"
FIGURE_PATH = FIGURE_DIR / "training_overfitting_comparison.png"

EXPERIMENTS = [
    ("mecab_morph_v8000", "MeCab Morph 8k"),
    ("okt_morph_v8000", "Okt Morph 8k"),
    ("sp_unigram_v8000", "SP Unigram 8k"),
    ("sp_bpe_v8000", "SP BPE 8k"),
    ("sp_unigram_v16000", "SP Unigram 16k"),
]

REQUIRED_COLUMNS = {
    "epoch",
    "train_loss",
    "val_loss",
    "val_acc",
}


def load_histories():
    histories = {}

    for experiment_name, display_name in EXPERIMENTS:
        history_path = HISTORY_DIR / f"history_{experiment_name}.csv"
        if not history_path.exists():
            raise FileNotFoundError(f"History CSV not found: {history_path}")

        history = pd.read_csv(history_path)
        missing_columns = REQUIRED_COLUMNS.difference(history.columns)
        if missing_columns:
            raise ValueError(
                f"{history_path.name} is missing columns: "
                f"{sorted(missing_columns)}"
            )
        if history[list(REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError(f"{history_path.name} contains missing values.")

        histories[display_name] = history

    return histories


def plot_training_curves(histories):
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("tab10", len(histories))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for (display_name, history), color in zip(histories.items(), colors):
        axes[0].plot(
            history["epoch"],
            history["train_loss"],
            marker="o",
            linewidth=2,
            label=display_name,
            color=color,
        )
        axes[1].plot(
            history["epoch"],
            history["val_loss"],
            marker="o",
            linewidth=2,
            label=display_name,
            color=color,
        )
        axes[2].plot(
            history["epoch"],
            history["val_acc"] * 100,
            marker="o",
            linewidth=2,
            label=display_name,
            color=color,
        )

        best_val_loss_index = history["val_loss"].idxmin()
        best_epoch = history.loc[best_val_loss_index, "epoch"]
        best_val_loss = history.loc[best_val_loss_index, "val_loss"]
        best_val_acc = history.loc[best_val_loss_index, "val_acc"] * 100

        axes[1].scatter(
            best_epoch,
            best_val_loss,
            color=color,
            edgecolor="black",
            s=90,
            zorder=5,
        )
        axes[2].scatter(
            best_epoch,
            best_val_acc,
            color=color,
            edgecolor="black",
            s=90,
            zorder=5,
        )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")

    for axis in axes:
        axis.set_xticks(range(1, 12))
        axis.grid(alpha=0.3)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        title="Experiment",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=5,
    )

    fig.suptitle(
        "Training Curves for Overfitting Analysis",
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        (
            "Black-edged points indicate the epoch with the lowest "
            "validation loss for each experiment."
        ),
        ha="center",
        fontsize=10,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0.16, 1, 0.94))
    plt.savefig(
        FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def print_overfitting_summary(histories):
    rows = []

    for display_name, history in histories.items():
        best_index = history["val_loss"].idxmin()
        best_row = history.loc[best_index]
        final_row = history.iloc[-1]

        rows.append(
            {
                "experiment": display_name,
                "best_epoch": int(best_row["epoch"]),
                "best_val_loss": best_row["val_loss"],
                "best_val_acc": best_row["val_acc"],
                "final_train_loss": final_row["train_loss"],
                "final_val_loss": final_row["val_loss"],
                "final_loss_gap": (
                    final_row["val_loss"] - final_row["train_loss"]
                ),
            }
        )

    summary = pd.DataFrame(rows)
    print("\nOverfitting summary")
    print(
        summary.to_string(
            index=False,
            formatters={
                "best_val_loss": "{:.4f}".format,
                "best_val_acc": "{:.2%}".format,
                "final_train_loss": "{:.4f}".format,
                "final_val_loss": "{:.4f}".format,
                "final_loss_gap": "{:.4f}".format,
            },
        )
    )


def main():
    histories = load_histories()
    plot_training_curves(histories)
    print_overfitting_summary(histories)
    print(f"\nSaved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
