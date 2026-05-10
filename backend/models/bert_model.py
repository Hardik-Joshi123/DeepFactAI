"""
BERT-based Fake News Classifier
---------------------------------
Wraps HuggingFace bert-base-uncased with a classification head.

Architecture:
  BertModel (pre-trained) → Dropout → Linear(768 → 2) → CrossEntropyLoss

The model fine-tunes all layers by default. To freeze the BERT backbone
and train only the head, call model.freeze_bert().

Usage:
  from models.bert_model import BertFakeNewsClassifier, BertDataset
  model = BertFakeNewsClassifier()
  dataset = BertDataset(texts, labels)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Model checkpoint — uses bert-base-uncased by default.
DEFAULT_BERT_MODEL = "bert-base-uncased"


class BertFakeNewsClassifier(nn.Module):
    """
    BERT-based binary classifier for fake news detection.

    Args:
        bert_model_name: HuggingFace model name or local path
        dropout:         Dropout on the pooled [CLS] representation
        freeze_bert:     Freeze BERT backbone weights (train head only)
        num_labels:      Number of output classes (2 for binary)
    """

    def __init__(
        self,
        bert_model_name: str = DEFAULT_BERT_MODEL,
        dropout: float = 0.3,
        freeze_bert: bool = False,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        try:
            from transformers import BertModel
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        if freeze_bert:
            self.freeze_bert_layers()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:       (batch, seq_len)
            attention_mask:  (batch, seq_len)
            token_type_ids:  (batch, seq_len) — optional

        Returns:
            logits: (batch, num_labels)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output            # (batch, 768)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)          # (batch, 2)
        return logits

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return softmax probabilities [p_real, p_fake]."""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
            return torch.softmax(logits, dim=-1)

    def freeze_bert_layers(self) -> None:
        """Freeze all BERT parameters — only train the classification head."""
        for param in self.bert.parameters():
            param.requires_grad = False
        logger.info("BERT backbone frozen. Training classification head only.")

    def unfreeze_bert_layers(self) -> None:
        """Unfreeze all BERT parameters for full fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True
        logger.info("BERT backbone unfrozen.")


class BertDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenised BERT inputs.

    Args:
        texts:      List of raw article strings
        labels:     List of integer labels (0=REAL, 1=FAKE) or None for inference
        tokenizer:  HuggingFace tokenizer instance
        max_len:    Maximum sequence length (default 512)
    """

    def __init__(
        self,
        texts: list[str],
        labels: Optional[list[int]],
        tokenizer,
        max_len: int = 512,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_bert_model(
    bert_model_name: str = DEFAULT_BERT_MODEL,
    dropout: float = 0.3,
    freeze_bert: bool = False,
) -> BertFakeNewsClassifier:
    """
    Build a BERT classifier with the given configuration.

    Args:
        bert_model_name: HuggingFace checkpoint name
        dropout:         Dropout rate on pooled output
        freeze_bert:     Start with frozen BERT backbone

    Returns:
        BertFakeNewsClassifier
    """
    model = BertFakeNewsClassifier(
        bert_model_name=bert_model_name,
        dropout=dropout,
        freeze_bert=freeze_bert,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "BERT classifier built: %s | trainable %s / %s params",
        bert_model_name,
        f"{trainable:,}",
        f"{total:,}",
    )
    return model
