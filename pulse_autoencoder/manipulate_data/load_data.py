import os

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def create_mixed_dataset(path: str) -> pd.DataFrame:
    dfs = []

    for f in os.listdir(path):
        new_df = load_data(os.path.join(path, f))
        dfs.append(new_df)

    return pd.concat(dfs, ignore_index=True)


def load_data(data_pth: str) -> pd.DataFrame:
    data_dict = pd.read_pickle(data_pth)
    print(f"Data loaded: {data_dict.keys()}")

    key = next(iter(data_dict.keys()))
    timestamps_raw, values = data_dict[key]

    # Convert timestamps from UNIX seconds to UTC
    timestamps_utc = pd.to_datetime(timestamps_raw, unit="s", utc=True)
    return pd.DataFrame({"timestamp_utc": timestamps_utc, "values": values})


def extract_specific_timestamp(data_path: str, timestamp: str) -> torch.Tensor:
    df = load_data(data_path)
    target = pd.Timestamp(timestamp, tz="UTC")

    df["time_diff"] = (df["timestamp_utc"] - target).abs()
    closest_row = df.loc[df["time_diff"].idxmin()]
    s = closest_row["values"]
    print("Closest timestamp:", closest_row["timestamp_utc"])
    # return s

    s = np.pad(s, (0, (4 - len(s) % 4) % 4))
    s = (s - s.mean()) / s.std()
    return torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def prepare_data(data_path: str) -> tuple:
    signals_df = pd.read_pickle(data_path)
    signals = signals_df["values"].to_list()

    max_len = max(len(s) for s in signals)

    train_signals, test_signals = train_test_split(
        signals, test_size=0.2, random_state=42
    )

    # create datasets
    train_dataset = SignalDataset(train_signals, max_len)
    test_dataset = SignalDataset(test_signals, max_len)
    all_dataset = SignalDataset(signals, max_len)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    all_data_loader = DataLoader(all_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, all_data_loader


class SignalDataset(Dataset):
    def __init__(self, signals: list, max_len: int) -> None:
        self.signals = signals
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> torch.Tensor:
        s = self.signals[idx]

        x_old = np.linspace(0, 1, len(s))
        x_new = np.linspace(0, 1, self.max_len)
        f = interp1d(x_old, s, kind="linear")
        resampled = f(x_new)
        norm_s = resampled / np.max(np.abs(resampled))

        # convert to tensor
        return torch.tensor(norm_s, dtype=torch.float32).unsqueeze(0)


if __name__ == "__main__":
    path = "/data/"
    df = create_mixed_dataset(path)
    df.to_pickle("/data/mixed_df.pkl")
    print(df)
