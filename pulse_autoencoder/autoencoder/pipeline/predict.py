import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from pulse_autoencoder.autoencoder.model.autoencoder_model import PulseAutoEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_signal(signal: np.ndarray, target_len: int) -> torch.Tensor:
    """Resample and normalize a single signal, returning a (1, 1, target_len) tensor."""
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    resampled = interp1d(x_old, signal, kind="linear")(x_new)
    norm = resampled / np.max(np.abs(resampled))
    return torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,L)


def compute_reconstruction_errors(
    model: torch.nn.Module,
    signals: list[np.ndarray],
    target_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return per-pulse MSE reconstruction error."""
    model.eval()
    errors = []
    reconstructions = []

    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            batch_signals = signals[i : i + batch_size]
            tensors = [preprocess_signal(s, target_len) for s in batch_signals]
            batch = torch.cat(tensors, dim=0).to(device)  # (B, 1, L)

            reconstructed = model(batch)
            # per-sample MSE: mean over channels and length dims
            mse = ((batch - reconstructed) ** 2).mean(dim=(1, 2))  # (B,)
            errors.append(mse.cpu().numpy())

            # rescale each reconstruction back to original signal scale & length
            recon_np = reconstructed.cpu().squeeze(1).numpy()  # (B, L)
            for j, sig in enumerate(batch_signals):
                raw = np.array(sig, dtype=np.float64)
                reconstructions.append(rescale_reconstruction(recon_np[j], raw))

    return np.concatenate(errors), reconstructions


def rescale_reconstruction(
    recon_norm: np.ndarray, raw_signal: np.ndarray
) -> np.ndarray:
    """Un-normalize and resample a reconstruction back to match the original signal."""
    scale = np.max(np.abs(raw_signal))
    recon_scaled = recon_norm * scale
    x_recon = np.linspace(0, 1, len(recon_scaled))
    x_orig = np.linspace(0, 1, len(raw_signal))
    return interp1d(x_recon, recon_scaled, kind="linear")(x_orig)


def main() -> None:
    data_path = "/data/mixed/mixed_df.pkl"
    model_path = "/pulse_autoencoder/autoencoder/model/mixed_data_autoencoder.pth"
    output = "df_classified_mixed_pulses.pkl"

    percentile = None
    threshold_mse = 0.00005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    df = pd.read_pickle(data_path)
    signals = df["values"].to_list()
    target_len = max(len(s) for s in signals)
    logger.info(f"Loaded {len(signals)} pulses  (target_len={target_len})")

    model = PulseAutoEncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info("Model loaded")

    errors, reconstructions = compute_reconstruction_errors(
        model, signals, target_len, device
    )

    if threshold_mse is not None:
        threshold = threshold_mse
        threshold_label = f"MSE = {threshold:.6f}"
    elif percentile is not None:
        threshold = np.percentile(errors, percentile)
        threshold_label = f"p{percentile} = {threshold:.6f}"
    else:
        raise ValueError("Set either percentile or threshold_mse")

    labels = np.where(errors > threshold, "bad", "good")
    n_bad = (labels == "bad").sum()
    logger.info(
        f"Threshold ({threshold_label})  |  "
        f"bad: {n_bad}  good: {len(labels) - n_bad}"
    )

    df["reconstruction_error"] = errors
    df["label"] = labels
    df["reconstruction"] = reconstructions
    df.to_pickle(output)
    logger.info(f"Labeled dataframe saved to {output}")

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.logspace(np.log10(errors[errors > 0].min()), np.log10(errors.max()), 100)
    ax.hist(errors, bins=bins, edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.set_xscale("log")
    ax.axvline(threshold, color="red", linestyle="--", label=threshold_label)
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Count")
    ax.set_title("Reconstruction Error Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig("error_distribution.png", dpi=150)
    logger.info("Saved error_distribution.png")

    bad_indices = np.where(labels == "bad")[0]

    if len(bad_indices) == 0:
        logger.info("No bad pulses found – nothing to plot.")
        return

    if len(bad_indices) > 10:  # noqa
        n_plots = 10
        plot_bad_indices = bad_indices[:n_plots]
    else:
        n_plots = len(bad_indices)
        plot_bad_indices = bad_indices

    # Determine grid size
    cols = min(4, n_plots)
    rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)

    model.eval()
    with torch.no_grad():
        for plot_idx, sig_idx in enumerate(plot_bad_indices):
            r, c = divmod(plot_idx, cols)
            ax = axes[r][c]

            raw = np.array(signals[sig_idx], dtype=np.float64)
            scale = np.max(np.abs(raw))

            inp = preprocess_signal(raw, target_len).to(device)
            recon = model(inp).cpu().squeeze().numpy()

            # un-normalize reconstruction back to original scale
            recon_scaled = recon * scale
            # resample reconstruction to original signal length
            x_recon = np.linspace(0, 1, len(recon_scaled))
            x_orig = np.linspace(0, 1, len(raw))
            recon_at_orig = interp1d(x_recon, recon_scaled, kind="linear")(x_orig)

            ax.plot(x_orig, raw, label="original", linewidth=0.7)
            ax.plot(
                x_orig, recon_at_orig, label="reconstructed", linewidth=0.7, alpha=0.8
            )
            ax.set_title(f"idx {sig_idx}  MSE={errors[sig_idx]:.4f}", fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

    # hide unused subplots
    for plot_idx in range(n_plots, rows * cols):
        r, c = divmod(plot_idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"Bad Pulses (n={n_plots}, {threshold_label})", fontsize=13)
    fig.tight_layout()
    fig.savefig("bad_pulses.png", dpi=150)
    logger.info(f"Saved bad_pulses.png  ({n_plots} pulses)")

    plt.show()


if __name__ == "__main__":
    main()
