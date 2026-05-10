"""
Evaluation Metrics & Visualisations
--------------------------------------
Provides a full evaluation suite:
  - accuracy, precision, recall, F1 (per-class and macro/weighted)
  - Confusion matrix (text + matplotlib)
  - ROC curve + AUC
  - Precision-Recall curve

Usage:
  from utils.metrics import evaluate_model, plot_confusion_matrix, plot_roc_curve
  report = evaluate_model(y_true, y_pred, y_proba)
  plot_confusion_matrix(y_true, y_pred)
  plot_roc_curve(y_true, y_proba[:, 1])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── core metrics ──────────────────────────────────────────────────────────────

def evaluate_model(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute a full evaluation report.

    Args:
        y_true:       Ground-truth labels (0=REAL, 1=FAKE)
        y_pred:       Predicted labels
        y_proba:      Predicted probabilities shape (N, 2) — optional, needed for ROC/AUC
        class_names:  Class label strings

    Returns:
        dict with accuracy, precision, recall, f1, confusion_matrix, roc_auc (if available)
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = class_names or ["REAL", "FAKE"]

    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=labels, zero_division=0
        ),
    }

    if y_proba is not None:
        fake_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        try:
            report["roc_auc"] = float(roc_auc_score(y_true, fake_proba))
        except ValueError:
            report["roc_auc"] = None

    return report


def print_report(report: dict) -> None:
    """Pretty-print the evaluation report."""
    sep = "-" * 60
    print(sep)
    print(f"{'ACCURACY':30s}: {report['accuracy']:.4f}")
    print(f"{'PRECISION (macro)':30s}: {report['precision_macro']:.4f}")
    print(f"{'RECALL (macro)':30s}: {report['recall_macro']:.4f}")
    print(f"{'F1 (macro)':30s}: {report['f1_macro']:.4f}")
    print(f"{'F1 (weighted)':30s}: {report['f1_weighted']:.4f}")
    if report.get("roc_auc"):
        print(f"{'ROC AUC':30s}: {report['roc_auc']:.4f}")
    print(sep)
    print(report["classification_report"])


# ── visualisations ────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    """Render and save a styled confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix as cm
    except ImportError as e:
        logger.warning("Cannot plot confusion matrix: %s", e)
        return

    labels = class_names or ["REAL", "FAKE"]
    mat = cm(y_true, y_pred)
    mat_normalised = mat.astype(float) / mat.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, ttl in zip(
        axes,
        [mat, mat_normalised],
        ["d", ".2%"],
        [f"{title} (Counts)", f"{title} (Normalised)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(ttl, fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str = "Model",
    save_path: str = "roc_curve.png",
) -> float:
    """
    Plot and save the ROC curve.

    Returns:
        AUC score
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError as e:
        logger.warning("Cannot plot ROC curve: %s", e)
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved → %s", save_path)
    return auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str = "Model",
    save_path: str = "pr_curve.png",
) -> None:
    """Plot and save the Precision-Recall curve."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import average_precision_score, precision_recall_curve
    except ImportError as e:
        logger.warning("Cannot plot PR curve: %s", e)
        return

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#6366f1", lw=2,
            label=f"{model_name} (AP = {avg_precision:.4f})")
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="gray", lw=1, linestyle="--", label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved → %s", save_path)


def compare_models(results: dict[str, dict]) -> None:
    """
    Print a side-by-side comparison table for multiple models.

    Args:
        results: {model_name: evaluate_model_output}
    """
    header = f"{'Model':<25} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'ROC AUC':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, r in results.items():
        auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "  N/A  "
        print(
            f"{name:<25} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
            f"{r['f1_weighted']:>12.4f} {auc:>10}"
        )
    print("=" * len(header) + "\n")
