import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pulses(df_bad: pd.DataFrame, title: str) -> None:
    """Plot original vs reconstruction for each row in df_bad."""
    n = len(df_bad)
    if n == 0:
        print("No bad pulses to display.")
        return

    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)

    for plot_idx, (df_idx, row) in enumerate(df_bad.iterrows()):
        r, c = divmod(plot_idx, cols)
        ax = axes[r][c]

        raw = np.array(row["values"], dtype=np.float64)
        recon = np.array(row["reconstruction"], dtype=np.float64)

        ax.plot(raw, label="original", linewidth=0.7)
        ax.plot(recon, label="reconstructed", linewidth=0.7, alpha=0.8)

        mse = row["reconstruction_error"]
        ts = row.get("timestamps", "")
        subtitle = f"idx {df_idx}  MSE={mse:.4f}"
        if ts != "":
            subtitle += f"\n{ts}"

        ax.set_title(subtitle, fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for plot_idx in range(n, rows * cols):
        r, c = divmod(plot_idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.savefig("reconstructed_bad_pulses.png")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect bad pulses from labeled DataFrame"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--n",
        type=int,
        default=40,
        help="Number of bad pulses to display (default: 8)",
    )
    group.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Specific DataFrame indices to plot (overrides --n)",
    )

    parser.add_argument(
        "--sort-by-error",
        action="store_true",
        help="Sort bad pulses by reconstruction error (worst first)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample bad pulses randomly instead of taking the first N",
    )
    args = parser.parse_args()

    df = pd.read_pickle("/data/df_classified_mixed_pulses.pkl")
    print(
        f"Loaded {len(df)} pulses  |  good: {(df['label'] == 'good').sum()}  bad: {(df['label'] == 'bad').sum()}"
    )

    if args.indices is not None:
        selection = df.loc[args.indices]
        title = f"Selected Pulses (n={len(selection)})"
    else:
        df_bad = df[df["label"] == "bad"].copy()

        if args.sort_by_error:
            df_bad = df_bad.sort_values("reconstruction_error", ascending=False)

        if args.random:
            selection = df_bad.sample(n=min(args.n, len(df_bad)), random_state=None)
            title = f"Random Bad Pulses (n={len(selection)})"
        else:
            selection = df_bad.head(args.n)
            title = f"Bad Pulses — {'worst first' if args.sort_by_error else 'first N'} (n={len(selection)})"

    plot_pulses(selection, title)


if __name__ == "__main__":
    main()
