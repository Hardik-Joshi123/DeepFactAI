#!/usr/bin/env python3
"""
Train LSTM / BiLSTM Fake News Classifier
==========================================
Trains both LSTM and BiLSTM models on the Kaggle Fake News dataset using
GloVe-300d pretrained embeddings.

Requirements (install before running):
  uv add torch scikit-learn pandas numpy joblib nltk requests tqdm

Usage:
  # Train LSTM
  python scripts/train_lstm.py --mode lstm --data-dir data --epochs 15

  # Train BiLSTM
  python scripts/train_lstm.py --mode bilstm --data-dir data --epochs 15

  # Specify pre-downloaded GloVe path
  python scripts/train_lstm.py --mode bilstm --glove-path data/glove.6B.300d.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# ── make sure project root is on path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def download_glove(target_dir: str, dim: int = 300) -> str:
    """Download GloVe embeddings if not already present."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    glove_file = target_dir / f"glove.6B.{dim}d.txt"

    if glove_file.exists():
        logger.info("GloVe file already present: %s", glove_file)
        return str(glove_file)

    zip_path = target_dir / "glove.6B.zip"
    if not zip_path.exists():
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        logger.info("Downloading GloVe embeddings from %s …", url)
        urllib.request.urlretrieve(url, zip_path)

    logger.info("Extracting GloVe embeddings …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(f"glove.6B.{dim}d.txt", target_dir)
    os.remove(zip_path)

    logger.info("GloVe saved → %s", glove_file)
    return str(glove_file)


def load_glove_embeddings(glove_path: str, word2idx: dict, embed_dim: int = 300) -> np.ndarray:
    """Load GloVe vectors for the words in word2idx."""
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    found = 0

    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in word2idx:
                vec = np.array(parts[1:], dtype=np.float32)
                embedding_matrix[word2idx[word]] = vec
                found += 1

    logger.info("GloVe: matched %d / %d vocab words (%.1f%%)", found, vocab_size, found / vocab_size * 100)
    return embedding_matrix


def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    from src.data_loader import load_kaggle_dataset, load_custom_csv
    from src.preprocessor import TextPreprocessor
    from src.trainer import Trainer
    from models.lstm_model import build_lstm_model
    from utils.metrics import evaluate_model, print_report, plot_confusion_matrix, plot_roc_curve

    device = get_device()
    logger.info("Using device: %s", device)
    os.makedirs("saved_models", exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if os.path.exists(f"{args.data_dir}/Fake.csv") and os.path.exists(f"{args.data_dir}/True.csv"):
        df = load_kaggle_dataset(f"{args.data_dir}/Fake.csv", f"{args.data_dir}/True.csv")
    else:
        logger.warning(
            "Kaggle dataset not found in %s.\n"
            "Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            "Then place Fake.csv and True.csv in %s/",
            args.data_dir, args.data_dir,
        )
        sys.exit(1)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    preprocessor = TextPreprocessor(max_vocab=args.vocab_size, max_len=args.max_len)
    X = preprocessor.fit_transform(texts)
    y = np.array(labels, dtype=np.float32)
    preprocessor.save(f"saved_models/preprocessor_{args.mode}.pkl")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    logger.info("Split — train: %d | val: %d | test: %d", len(y_train), len(y_val), len(y_test))

    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    # ── 3. Build model ────────────────────────────────────────────────────────
    model = build_lstm_model(
        vocab_size=preprocessor.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mode=args.mode,
    )

    # ── 4. Load GloVe embeddings (optional) ───────────────────────────────────
    if args.use_glove:
        if not args.glove_path:
            args.glove_path = download_glove(args.data_dir, args.embed_dim)
        embedding_matrix = load_glove_embeddings(
            args.glove_path, preprocessor.word2idx, args.embed_dim
        )
        model.load_pretrained_embeddings(embedding_matrix, freeze=False)
        logger.info("GloVe embeddings loaded into model.")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir="saved_models",
        clip_grad=1.0,
    )

    checkpoint_name = f"best_{args.mode}.pt"
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
        scheduler=scheduler,
        checkpoint_name=checkpoint_name,
    )
    Trainer.plot_history(history, save_path=f"saved_models/training_curves_{args.mode}.png")

    # ── 6. Evaluate on test set ───────────────────────────────────────────────
    model.eval()
    all_preds, all_proba, all_true = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            proba = model.predict_proba(X_batch).cpu().numpy()
            preds = (proba >= 0.5).astype(int)
            all_proba.extend(proba)
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy().astype(int))

    y_proba_2d = np.column_stack([1 - np.array(all_proba), all_proba])
    report = evaluate_model(all_true, all_preds, y_proba_2d)
    print_report(report)

    plot_confusion_matrix(
        np.array(all_true), np.array(all_preds),
        save_path=f"saved_models/confusion_{args.mode}.png",
        title=f"Confusion Matrix ({args.mode.upper()})",
    )
    plot_roc_curve(
        np.array(all_true), np.array(all_proba),
        model_name=args.mode.upper(),
        save_path=f"saved_models/roc_{args.mode}.png",
    )

    logger.info("Training complete. Artifacts saved in saved_models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM / BiLSTM fake news classifier")
    parser.add_argument("--mode", choices=["lstm", "bilstm"], default="bilstm")
    parser.add_argument("--data-dir", default="data", help="Directory with Fake.csv / True.csv")
    parser.add_argument("--vocab-size", type=int, default=30_000)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-glove", action="store_true", default=True)
    parser.add_argument("--glove-path", default=None, help="Path to glove.6B.300d.txt")
    args = parser.parse_args()
    main(args)
