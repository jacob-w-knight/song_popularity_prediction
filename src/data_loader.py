"""
Data loading and downloading utilities
"""

import pandas as pd
from pathlib import Path


def download_data(force_download=False):

    # Download Spotify dataset from HuggingFace

    # Parameters
    # ----------
    # force_download : bool, optional
    #     Force re-download even if file exists

    # Returns
    # -------
    # pd.DataFrame
    #     Raw Spotify dataset

    data_path = Path("data/raw/spotify_data.csv")

    if data_path.exists() and not force_download:
        print(f"Loading cached data from {data_path}")
        return pd.read_csv(data_path)

    print("Downloading data from HuggingFace...")
    df = pd.read_csv(
        "hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv"
    ).iloc[:, 1:]

    # Create directory if needed
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    return df


def load_processed_data(scaled=True):

    # Load preprocessed data

    # Parameters
    # ----------
    # scaled : bool, optional
    #     Whether to load scaled or unscaled features

    # Returns
    # -------
    # pd.DataFrame
    #     Processed dataset

    filename = "scaled_data.csv" if scaled else "unscaled_data.csv"
    return pd.read_csv(f"data/processed/{filename}")
