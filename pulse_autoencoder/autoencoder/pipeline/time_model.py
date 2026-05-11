import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from pulse_autoencoder.autoencoder.model.autoencoder_model import PulseAutoEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "mixed" / "mixed_df.pkl"
MODEL_PATH = (
    PROJECT_ROOT
    / "pulse_autoencoder"
    / "autoencoder"
    / "model"
    / "mixed_data_autoencoder.pth"
)
THRESHOLD_MSE = 0.00005
N_ITERATIONS = 1000


def preprocess_signal(signal: np.ndarray, target_len: int) -> torch.Tensor:
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    resampled = interp1d(x_old, signal, kind="linear")(x_new)
    norm = resampled / np.max(np.abs(resampled))
    return torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_pickle(DATA_PATH)
    signals = df["values"].to_list()
    target_len = max(len(s) for s in signals)

    model = PulseAutoEncoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # pick a random pulse
    rng = np.random.default_rng(42)
    sample = signals[rng.integers(len(signals))]

    # warmup (important for CUDA and also for CPU caches)
    with torch.no_grad():
        inp = preprocess_signal(sample, target_len).to(device)
        for _ in range(10):
            _ = model(inp)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # timed runs: preprocess + forward + threshold
    times = []
    with torch.no_grad():
        for _ in range(N_ITERATIONS):
            t0 = time.perf_counter()

            inp = preprocess_signal(sample, target_len).to(device)
            recon = model(inp)
            mse = ((inp - recon) ** 2).mean().item()
            label = "bad" if mse > THRESHOLD_MSE else "good"

            if device.type == "cuda":
                torch.cuda.synchronize()

            times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000

    print(f"\n{'─' * 45}")
    print(f"  Single-pulse labeling  ({N_ITERATIONS} iterations)")
    print(f"{'─' * 45}")
    print(f"  Mean:   {times_ms.mean():.3f} ms")
    print(f"  Std:    {times_ms.std():.3f} ms")
    print(f"  Min:    {times_ms.min():.3f} ms")
    print(f"  Max:    {times_ms.max():.3f} ms")
    print(f"  Median: {np.median(times_ms):.3f} ms")
    print(f"{'─' * 45}")
    print(f"  Last label: {label}  (MSE={mse:.6f})")


if __name__ == "__main__":
    main()
