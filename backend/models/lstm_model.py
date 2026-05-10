"""
LSTM-based Fake News Classifier
---------------------------------
PyTorch implementation of a stacked LSTM with an attention mechanism.

Architecture:
  Embedding (pretrained GloVe) → Dropout → BiLSTM or LSTM →
  Attention → Dropout → FC → Sigmoid

Usage:
  from models.lstm_model import LSTMClassifier, build_lstm_model
  model = build_lstm_model(vocab_size=30002, embed_dim=300, mode="lstm")
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Scaled dot-product self-attention pooling.
    Collapses the sequence dimension to a fixed-size context vector.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
            weights: (batch, seq_len) — attention distribution
        """
        scores = self.attn(lstm_out).squeeze(-1)        # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)             # (batch, seq_len)
        context = torch.bmm(
            weights.unsqueeze(1), lstm_out              # (batch, 1, seq_len) × (batch, seq_len, hidden)
        ).squeeze(1)                                    # (batch, hidden_dim)
        return context, weights


class LSTMClassifier(nn.Module):
    """
    LSTM / Bidirectional LSTM fake-news classifier.

    Args:
        vocab_size:    Vocabulary size (including PAD and UNK)
        embed_dim:     Word embedding dimension (default 300 for GloVe)
        hidden_dim:    LSTM hidden state dimension
        num_layers:    Number of stacked LSTM layers
        dropout:       Dropout probability
        bidirectional: Use BiLSTM if True
        pad_idx:       Index of the padding token
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.num_directions
        self.attention = AttentionLayer(lstm_out_dim)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len) integer token indices
        Returns:
            logits:         (batch,) — raw logit for FAKE class
            attn_weights:   (batch, seq_len) — attention distribution
        """
        embedded = self.embed_dropout(self.embedding(x))      # (B, L, E)
        lstm_out, _ = self.lstm(embedded)                     # (B, L, H*D)
        context, attn_weights = self.attention(lstm_out)      # (B, H*D)
        out = self.fc(self.fc_dropout(context)).squeeze(-1)   # (B,)
        return out, attn_weights

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities in range [0, 1] for the FAKE class."""
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.sigmoid(logits)

    def load_pretrained_embeddings(
        self, embedding_matrix: np.ndarray, freeze: bool = False
    ) -> None:
        """
        Initialise the embedding layer with a pretrained matrix (e.g. GloVe).

        Args:
            embedding_matrix: numpy array of shape (vocab_size, embed_dim)
            freeze: If True the embedding weights will not be updated during training
        """
        assert embedding_matrix.shape[0] == self.embedding.num_embeddings, (
            f"Embedding matrix vocab size {embedding_matrix.shape[0]} "
            f"does not match model vocab {self.embedding.num_embeddings}"
        )
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix).float())
        if freeze:
            self.embedding.weight.requires_grad = False
        nn.init.zeros_(self.embedding.weight.data[0])  # keep PAD row zeroed


def build_lstm_model(
    vocab_size: int,
    embed_dim: int = 300,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    mode: str = "lstm",
    pad_idx: int = 0,
) -> LSTMClassifier:
    """
    Factory function for building LSTM or BiLSTM classifiers.

    Args:
        vocab_size:  Vocabulary size
        embed_dim:   Embedding dimension
        hidden_dim:  LSTM hidden size
        num_layers:  Number of LSTM layers
        dropout:     Dropout rate
        mode:        'lstm' or 'bilstm'
        pad_idx:     Padding token index

    Returns:
        LSTMClassifier instance
    """
    if mode not in ("lstm", "bilstm"):
        raise ValueError(f"mode must be 'lstm' or 'bilstm', got '{mode}'")
    bidirectional = mode == "bilstm"
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        pad_idx=pad_idx,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{mode.upper()}] Trainable parameters: {total_params:,}")
    return model
