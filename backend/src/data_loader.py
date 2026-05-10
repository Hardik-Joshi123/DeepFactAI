"""
Data Loader for Fake News Detection
------------------------------------
Supports loading:
  1. Kaggle Fake and Real News Dataset (Fake.csv / True.csv)
  2. LIAR dataset (train.tsv)
  3. FakeNewsNet JSON files
  4. Custom CSV with 'text' and 'label' columns

Usage:
  from src.data_loader import load_kaggle_dataset, load_liar_dataset
  df = load_kaggle_dataset("data/Fake.csv", "data/True.csv")
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_kaggle_dataset(
    fake_path: str,
    real_path: str,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the Kaggle 'Fake and Real News' dataset.
    Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

    Args:
        fake_path: Path to Fake.csv
        real_path: Path to True.csv
        max_samples: Optionally cap dataset size per class

    Returns:
        DataFrame with columns: text, label (0=REAL, 1=FAKE), title, subject, date
    """
    logger.info("Loading Kaggle Fake News dataset…")

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    fake_df["label"] = 1
    real_df["label"] = 0

    fake_df["label_str"] = "FAKE"
    real_df["label_str"] = "REAL"

    df = pd.concat([fake_df, real_df], ignore_index=True)

    # Combine title and text
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["text"] = df["text"].str.strip()

    if max_samples:
        fake_sample = df[df["label"] == 1].sample(
            min(max_samples, len(df[df["label"] == 1])), random_state=42
        )
        real_sample = df[df["label"] == 0].sample(
            min(max_samples, len(df[df["label"] == 0])), random_state=42
        )
        df = pd.concat([fake_sample, real_sample], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(
        "Loaded %d samples (FAKE: %d, REAL: %d)",
        len(df),
        df["label"].sum(),
        len(df) - df["label"].sum(),
    )
    return df[["text", "label", "label_str"]]


def load_liar_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load the LIAR dataset (Wang, 2017).
    Download from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

    Label mapping: pants-fire, false, barely-true → FAKE
                   half-true, mostly-true, true     → REAL

    Args:
        data_dir: Directory containing train.tsv, valid.tsv, test.tsv

    Returns:
        DataFrame with columns: text, label (0=REAL, 1=FAKE), raw_label, speaker
    """
    logger.info("Loading LIAR dataset from %s", data_dir)
    cols = [
        "id", "raw_label", "statement", "subjects", "speaker",
        "speaker_job", "state", "party", "barely_true_count",
        "false_count", "half_true_count", "mostly_true_count",
        "pants_on_fire_count", "context",
    ]

    fake_labels = {"pants-fire", "false", "barely-true"}
    real_labels = {"half-true", "mostly-true", "true"}

    frames = []
    for split in ("train.tsv", "valid.tsv", "test.tsv"):
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            logger.warning("File not found: %s", path)
            continue
        df = pd.read_csv(path, sep="\t", header=None, names=cols)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No LIAR .tsv files found in {data_dir}")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["raw_label"].isin(fake_labels | real_labels)].copy()
    df["label"] = df["raw_label"].apply(lambda x: 1 if x in fake_labels else 0)
    df["label_str"] = df["label"].map({1: "FAKE", 0: "REAL"})
    df["text"] = df["statement"].fillna("")

    logger.info("Loaded LIAR: %d samples", len(df))
    return df[["text", "label", "label_str", "raw_label", "speaker"]]


def load_custom_csv(csv_path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Load a custom CSV with user-defined text and label columns.

    Args:
        csv_path: Path to CSV file
        text_col: Column name containing article text
        label_col: Column name containing labels ('FAKE'/'REAL' or 0/1)

    Returns:
        DataFrame with columns: text, label, label_str
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={text_col: "text", label_col: "label_raw"})
    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["text"].str.len() > 20].copy()

    # Normalise labels
    df["label_str"] = df["label_raw"].astype(str).str.upper().str.strip()
    df["label_str"] = df["label_str"].replace({"1": "FAKE", "0": "REAL", "TRUE": "REAL", "FALSE": "FAKE"})
    df["label"] = df["label_str"].map({"FAKE": 1, "REAL": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    logger.info("Loaded custom CSV: %d samples", len(df))
    return df[["text", "label", "label_str"]]


def download_kaggle_dataset(output_dir: str = "data") -> None:
    """
    Helper to download the Kaggle Fake News dataset using the Kaggle API.
    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    """
    try:
        import kaggle  # type: ignore
    except ImportError:
        raise ImportError("Install kaggle: pip install kaggle")

    os.makedirs(output_dir, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "clmentbisaillon/fake-and-real-news-dataset",
        path=output_dir,
        unzip=True,
    )
    logger.info("Downloaded Kaggle dataset to %s", output_dir)


def class_distribution(df: pd.DataFrame) -> dict:
    """Return class distribution statistics."""
    counts = df["label"].value_counts()
    total = len(df)
    return {
        "total": total,
        "fake": int(counts.get(1, 0)),
        "real": int(counts.get(0, 0)),
        "fake_pct": round(counts.get(1, 0) / total * 100, 1),
        "real_pct": round(counts.get(0, 0) / total * 100, 1),
        "balance_ratio": round(
            min(counts.get(1, 1), counts.get(0, 1)) / max(counts.get(1, 1), counts.get(0, 1)), 3
        ),
    }
