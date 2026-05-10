#!/usr/bin/env python3
"""
Compare All Trained Models
============================
Loads saved TF-IDF LR, LSTM, BiLSTM, and BERT models, evaluates them
on the test split, and prints a side-by-side comparison table.

Usage:
  python scripts/evaluate_models.py --data-dir data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def evaluate_tfidf(X_test_text, y_test):
    import joblib
    vect = joblib.load("saved_models/tfidf_vectorizer.joblib")
    clf = joblib.load("saved_models/lr_classifier.joblib")
    from app.predictor import clean_text
    cleaned = [clean_text(t) for t in X_test_text]
    X_vec = vect.transform(cleaned)
    y_pred = clf.predict(X_vec)
    y_proba = clf.predict_proba(X_vec)
    # Map string labels to int
    classes = list(clf.classes_)
    fake_idx = classes.index("FAKE")
    y_pred_int = [1 if p == "FAKE" else 0 for p in y_pred]
    return np.array(y_pred_int), y_proba


def evaluate_lstm(X_test_seq, y_test, mode="lstm", device="cpu"):
    import torch
    from models.lstm_model import build_lstm_model
    from src.preprocessor import TextPreprocessor
    from torch.utils.data import DataLoader, TensorDataset

    prep = TextPreprocessor.load(f"saved_models/preprocessor_{mode}.pkl")
    model = build_lstm_model(
        vocab_size=prep.vocab_size, mode=mode
    ).to(device)
    model.load_state_dict(torch.load(f"saved_models/best_{mode}.pt", map_location=device))
    model.eval()

    ds = TensorDataset(
        torch.tensor(X_test_seq, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=64)
    all_proba, all_preds = [], []

    with torch.no_grad():
        for X_batch, _ in loader:
            proba = model.predict_proba(X_batch.to(device)).cpu().numpy()
            all_proba.extend(proba)
            all_preds.extend((proba >= 0.5).astype(int))

    all_proba = np.array(all_proba)
    return np.array(all_preds), np.column_stack([1 - all_proba, all_proba])


def main(args):
    from sklearn.model_selection import train_test_split
    from src.data_loader import load_kaggle_dataset
    from src.preprocessor import TextPreprocessor
    from utils.metrics import evaluate_model, compare_models

    # Load data
    df = load_kaggle_dataset(f"{args.data_dir}/Fake.csv", f"{args.data_dir}/True.csv")
    texts = df["text"].tolist()
    labels = np.array(df["label"].tolist())

    _, X_temp, _, y_temp = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    _, X_test_text, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    results = {}

    # TF-IDF + LR
    if Path("saved_models/tfidf_vectorizer.joblib").exists():
        preds, proba = evaluate_tfidf(X_test_text, y_test)
        results["TF-IDF + LR"] = evaluate_model(y_test, preds, proba)
    else:
        logger.warning("TF-IDF model not found — skipping.")

    # LSTM
    if Path("saved_models/best_lstm.pt").exists():
        prep = TextPreprocessor.load("saved_models/preprocessor_lstm.pkl")
        X_test_seq = prep.transform(X_test_text)
        preds, proba = evaluate_lstm(X_test_seq, y_test, mode="lstm")
        results["LSTM"] = evaluate_model(y_test, preds, proba)
    else:
        logger.warning("LSTM model not found — run: python scripts/train_lstm.py --mode lstm")

    # BiLSTM
    if Path("saved_models/best_bilstm.pt").exists():
        prep = TextPreprocessor.load("saved_models/preprocessor_bilstm.pkl")
        X_test_seq = prep.transform(X_test_text)
        preds, proba = evaluate_lstm(X_test_seq, y_test, mode="bilstm")
        results["BiLSTM"] = evaluate_model(y_test, preds, proba)
    else:
        logger.warning("BiLSTM model not found — run: python scripts/train_lstm.py --mode bilstm")

    if not results:
        logger.error("No trained models found in saved_models/. Train them first.")
        sys.exit(1)

    compare_models(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    main(args)
