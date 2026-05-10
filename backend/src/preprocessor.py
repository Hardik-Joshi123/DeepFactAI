"""
NLP Text Preprocessor
----------------------
Provides a full preprocessing pipeline including:
  - HTML/URL removal
  - Tokenisation
  - Lemmatisation (NLTK WordNetLemmatizer)
  - Stopword removal
  - Sequence encoding for Keras/PyTorch models
  - Padding to fixed length

Usage:
  preprocessor = TextPreprocessor(max_vocab=30000, max_len=512)
  preprocessor.fit(train_texts)
  X_train = preprocessor.transform(train_texts)
"""

from __future__ import annotations

import logging
import pickle
import re
import string
from collections import Counter
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── try to import NLTK components (optional) ─────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords as nltk_sw
    from nltk.stem import WordNetLemmatizer

    for pkg in ("punkt", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_STOPWORDS = set(nltk_sw.words("english"))
    _NLTK_AVAILABLE = True
    logger.info("NLTK loaded successfully.")
except ImportError:
    _LEMMATIZER = None
    _NLTK_STOPWORDS = set()
    _NLTK_AVAILABLE = False
    logger.warning("NLTK not available — using simple fallback preprocessing.")

# Baseline stopwords (always available)
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "that",
    "this", "these", "those", "it", "its",
}
_STOPWORDS = _NLTK_STOPWORDS | _FALLBACK_STOPWORDS

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def _clean_raw(text: str) -> str:
    """Remove URLs, HTML tags, normalise whitespace."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _tokenise(text: str) -> list[str]:
    """Split into tokens, optionally using NLTK."""
    if _NLTK_AVAILABLE:
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except Exception:
            pass
    return text.split()


def _lemmatise(token: str) -> str:
    if _NLTK_AVAILABLE and _LEMMATIZER:
        return _LEMMATIZER.lemmatize(token)
    return token


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Full pipeline: clean → tokenise → lemmatise → remove stopwords → rejoin.

    Args:
        text: Raw article text
        remove_stopwords: Whether to remove common stopwords

    Returns:
        Cleaned, space-joined token string
    """
    cleaned = _clean_raw(text)
    tokens = _tokenise(cleaned)
    tokens = [_lemmatise(t) for t in tokens if t.isalpha() and len(t) > 1]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return " ".join(tokens)


class TextPreprocessor:
    """
    Stateful preprocessor that builds a vocabulary and converts text to
    integer sequences suitable for embedding layers.

    Example:
        prep = TextPreprocessor(max_vocab=30_000, max_len=512)
        prep.fit(train_texts)
        X_train = prep.transform(train_texts)   # shape (N, 512)
        prep.save("models/preprocessor.pkl")
    """

    def __init__(
        self,
        max_vocab: int = 30_000,
        max_len: int = 512,
        remove_stopwords: bool = False,  # keep stopwords for DL models
    ) -> None:
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.remove_stopwords = remove_stopwords
        self.word2idx: dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.vocab_size: int = 2
        self._fitted = False

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "TextPreprocessor":
        """Build vocabulary from training texts."""
        counter: Counter = Counter()
        for text in texts:
            cleaned = preprocess_text(text, self.remove_stopwords)
            counter.update(cleaned.split())

        for word, _ in counter.most_common(self.max_vocab - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)
        self._fitted = True
        logger.info("Vocabulary built: %d tokens", self.vocab_size)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """
        Convert texts to padded integer sequences.

        Returns:
            numpy array of shape (len(texts), max_len)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        sequences = []
        for text in texts:
            cleaned = preprocess_text(text, self.remove_stopwords)
            seq = [self.word2idx.get(w, UNK_IDX) for w in cleaned.split()]
            seq = seq[: self.max_len]  # truncate
            seq += [PAD_IDX] * (self.max_len - len(seq))  # pad
            sequences.append(seq)
        return np.array(sequences, dtype=np.int64)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "TextPreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not a TextPreprocessor")
        return obj

    def __repr__(self) -> str:
        return (
            f"TextPreprocessor(vocab_size={self.vocab_size}, "
            f"max_len={self.max_len}, fitted={self._fitted})"
        )


import os  # noqa: E402 (needed for save path)
