"""
Explainability Utilities
-------------------------
Two approaches for explaining model predictions:

1. LIME (Local Interpretable Model-Agnostic Explanations)
   - Model-agnostic: works with any sklearn/PyTorch/HuggingFace model
   - Perturbs the input and fits a local linear surrogate
   - Requires: pip install lime

2. TF-IDF Coefficient Attribution (built-in, no extra deps)
   - Uses LR coefficients × TF-IDF weights
   - Fast and interpretable for the TF-IDF + LR pipeline

Usage:
  explainer = LIMEExplainer(predict_fn=your_model_predict_fn)
  explanation = explainer.explain(text)
  explainer.plot(explanation)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    Wrapper around the LIME text explainer.

    Args:
        predict_fn:     Function accepting list[str] → np.ndarray of shape (N, 2)
                        where column 0 = P(REAL) and column 1 = P(FAKE).
        class_names:    Labels for each class index (default ['REAL', 'FAKE'])
        num_features:   Number of top features to highlight
        num_samples:    Number of perturbation samples for LIME
    """

    def __init__(
        self,
        predict_fn: Callable[[list[str]], np.ndarray],
        class_names: list[str] | None = None,
        num_features: int = 15,
        num_samples: int = 300,
    ) -> None:
        try:
            from lime.lime_text import LimeTextExplainer as _LTE
        except ImportError:
            raise ImportError(
                "LIME not installed. Run: pip install lime\n"
                "or: uv add lime"
            )
        self._explainer = _LTE(class_names=class_names or ["REAL", "FAKE"])
        self.predict_fn = predict_fn
        self.num_features = num_features
        self.num_samples = num_samples

    def explain(self, text: str, label_idx: int = 1) -> dict:
        """
        Explain prediction for the given text.

        Args:
            text:      Input article text
            label_idx: Class index to explain (1 = FAKE by default)

        Returns:
            dict with keys: words, scores, positive, negative, raw
        """
        exp = self._explainer.explain_instance(
            text,
            self.predict_fn,
            num_features=self.num_features,
            num_samples=self.num_samples,
            labels=(label_idx,),
        )
        features = exp.as_list(label=label_idx)
        words = [f[0] for f in features]
        scores = [f[1] for f in features]
        return {
            "words": words,
            "scores": scores,
            "positive": [(w, s) for w, s in zip(words, scores) if s > 0],
            "negative": [(w, s) for w, s in zip(words, scores) if s < 0],
            "raw": exp,
        }

    def plot(self, explanation: dict, save_path: str | None = None) -> None:
        """Display the LIME explanation as a matplotlib bar chart."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available.")
            return

        words = explanation["words"]
        scores = explanation["scores"]
        colors = ["#ef4444" if s > 0 else "#22c55e" for s in scores]

        fig, ax = plt.subplots(figsize=(10, max(4, len(words) * 0.4)))
        y_pos = range(len(words))
        ax.barh(list(y_pos), scores, color=colors, edgecolor="none")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(words, fontsize=10)
        ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
        ax.set_title("LIME Word Importance (Red=FAKE, Green=REAL)", fontsize=12)
        ax.set_xlabel("LIME Coefficient")
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("LIME plot saved → %s", save_path)
        else:
            plt.show()
        plt.close(fig)


class TFIDFExplainer:
    """
    Fast word-importance explainer for TF-IDF + Logistic Regression pipelines.

    Computes per-word contribution as: tfidf_weight × lr_coefficient

    Args:
        vectorizer:  Fitted sklearn TfidfVectorizer
        classifier:  Fitted sklearn LogisticRegression
    """

    def __init__(self, vectorizer, classifier) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self._coefs = classifier.coef_[0]
        self._feature_names = vectorizer.get_feature_names_out()

    def explain(self, text: str, top_k: int = 15) -> dict:
        """
        Return top-k most important words for the given text.

        Returns:
            dict with:
              words:    list of token strings
              scores:   list of contribution scores
              positive: tokens pushing toward FAKE
              negative: tokens pushing toward REAL
        """
        vec = self.vectorizer.transform([text])
        dense = np.asarray(vec.todense())[0]
        contribution = dense * self._coefs

        present = np.where(dense > 0)[0]
        if len(present) == 0:
            return {"words": [], "scores": [], "positive": [], "negative": []}

        present_scores = contribution[present]
        present_names = [self._feature_names[i] for i in present]

        # Sort by absolute magnitude
        order = np.argsort(np.abs(present_scores))[::-1][:top_k]

        words = [present_names[i] for i in order]
        scores = [float(present_scores[i]) for i in order]

        return {
            "words": words,
            "scores": scores,
            "positive": [(w, s) for w, s in zip(words, scores) if s > 0],
            "negative": [(w, s) for w, s in zip(words, scores) if s < 0],
        }

    def highlight_html(self, text: str, top_k: int = 15) -> str:
        """
        Return an HTML string with the most important words highlighted.
        Red = pushes toward FAKE, Green = pushes toward REAL.
        """
        explanation = self.explain(text, top_k=top_k)
        word_score: dict[str, float] = dict(zip(explanation["words"], explanation["scores"]))

        tokens = text.split()
        html_tokens = []
        for token in tokens:
            clean = token.lower().strip(".,!?\"'()[]{}")
            score = word_score.get(clean, 0.0)
            if abs(score) > 0.01:
                intensity = min(int(abs(score) * 500), 200)
                color = f"rgba(239,68,68,{min(abs(score)*5, 0.9):.2f})" if score > 0 else \
                        f"rgba(34,197,94,{min(abs(score)*5, 0.9):.2f})"
                html_tokens.append(
                    f'<mark style="background:{color};padding:1px 2px;border-radius:2px">{token}</mark>'
                )
            else:
                html_tokens.append(token)

        return " ".join(html_tokens)
