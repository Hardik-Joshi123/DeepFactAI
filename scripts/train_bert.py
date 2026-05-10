#!/usr/bin/env python3
"""
Fine-Tune BERT for Fake News Classification
=============================================
Fine-tunes bert-base-uncased on the Kaggle Fake News dataset.

Requirements (install before running):
  uv add torch transformers scikit-learn pandas numpy joblib tqdm accelerate

Usage:
  python scripts/train_bert.py --data-dir data --epochs 3 --batch-size 16
  python scripts/train_bert.py --data-dir data --epochs 5 --freeze-bert --batch-size 32
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
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer, get_linear_schedule_with_warmup
    from sklearn.model_selection import train_test_split

    from src.data_loader import load_kaggle_dataset
    from models.bert_model import BertFakeNewsClassifier, BertDataset
    from utils.metrics import evaluate_model, print_report, plot_confusion_matrix, plot_roc_curve

    device = get_device()
    logger.info("Using device: %s", device)
    os.makedirs("saved_models", exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if not (os.path.exists(f"{args.data_dir}/Fake.csv") and os.path.exists(f"{args.data_dir}/True.csv")):
        logger.error(
            "Dataset not found in %s.\n"
            "Download: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
            args.data_dir,
        )
        sys.exit(1)

    df = load_kaggle_dataset(
        f"{args.data_dir}/Fake.csv",
        f"{args.data_dir}/True.csv",
        max_samples=args.max_samples,
    )
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # ── 2. Tokenise ───────────────────────────────────────────────────────────
    logger.info("Loading BERT tokenizer …")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    tokenizer.save_pretrained("saved_models/bert_tokenizer")

    X_train_text, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info("Split — train: %d | val: %d | test: %d", len(y_train), len(y_val), len(y_test))

    def make_loader(texts_list, labels_list, shuffle=False):
        dataset = BertDataset(texts_list, labels_list, tokenizer, max_len=args.max_len)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train_text, y_train, shuffle=True)
    val_loader = make_loader(X_val_text, y_val)
    test_loader = make_loader(X_test_text, y_test)

    # ── 3. Build model ────────────────────────────────────────────────────────
    logger.info("Loading BERT model: %s …", args.bert_model)
    model = BertFakeNewsClassifier(
        bert_model_name=args.bert_model,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert,
        num_labels=2,
    ).to(device)

    # ── 4. Optimiser + scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    # ── 5. Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            t_loss += loss.item() * len(labels_t)
            preds = logits.argmax(dim=-1)
            t_correct += (preds == labels_t).sum().item()
            t_total += len(labels_t)

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_t = batch["labels"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels_t)
                v_loss += loss.item() * len(labels_t)
                preds = logits.argmax(dim=-1)
                v_correct += (preds == labels_t).sum().item()
                v_total += len(labels_t)

        t_loss /= t_total
        t_acc = t_correct / t_total
        v_loss /= v_total
        v_acc = v_correct / v_total
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f",
            epoch, args.epochs, t_loss, t_acc, v_loss, v_acc,
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), "saved_models/best_bert.pt")
            logger.info("  ✓ New best model saved (val_acc=%.4f)", best_val_acc)

    # ── 6. Load best and evaluate ─────────────────────────────────────────────
    model.load_state_dict(torch.load("saved_models/best_bert.pt", map_location=device))
    model.eval()

    all_preds, all_proba, all_true = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            proba = model.predict_proba(input_ids, attention_mask, token_type_ids).cpu().numpy()
            preds = proba.argmax(axis=-1)
            all_proba.append(proba)
            all_preds.extend(preds)
            all_true.extend(batch["labels"].numpy())

    all_proba = np.vstack(all_proba)
    report = evaluate_model(all_true, all_preds, all_proba)
    print_report(report)

    plot_confusion_matrix(
        np.array(all_true), np.array(all_preds),
        save_path="saved_models/confusion_bert.png",
        title="Confusion Matrix (BERT)",
    )
    plot_roc_curve(
        np.array(all_true), all_proba[:, 1],
        model_name="BERT",
        save_path="saved_models/roc_bert.png",
    )
    logger.info("BERT training complete. Best val_acc: %.4f", best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for fake news detection")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze-bert", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    main(args)
