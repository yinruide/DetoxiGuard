"""
eval_bert.py
------------
Standalone evaluation script for the fine-tuned BERT toxic comment
classifier. Loads the best checkpoint, runs inference on the validation
set, and produces:

    1. Summary metrics table  (console + CSV)
    2. Per-label ROC curves
    3. Per-label Precision-Recall curves
    4. Confusion matrices (2×2 per label, optimal thresholds)
    5. Probability distribution histograms (positive vs negative)
    6. Threshold sensitivity curves (F1 vs threshold per label)
    7. Error analysis  (top FP / FN examples per label)

All figures are saved to  outputs/eval_bert/

Usage:
    python evaluation/eval_bert.py \
        --checkpoint  outputs/bert_final/best_checkpoint \
        --base_model  bert-base-uncased \
        --val_csv     data/val_split.csv \
        --output_dir  outputs/eval_bert

Author: Yanfu Wang
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # headless backend for HPC / remote servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Import from our training script ───────────────
# Adjust sys.path so `classifier.train_bert` is importable when running
# from the repo root (python evaluation/eval_bert.py).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from classifier.train_bert import (
    LABELS,
    load_trained_model,
    predict,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

FIGSIZE_MULTI = (18, 10)   # for 2×3 subplot grids
FIGSIZE_SINGLE = (10, 7)   # for single-panel plots
DPI = 150


# ──────────────────────────────────────────────
# 1. Summary metrics table
# ──────────────────────────────────────────────

def build_metrics_table(
    labels_arr: np.ndarray,
    probs_arr: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """
    Build a DataFrame with per-label and macro/micro rows, comparing
    default (t=0.5) vs optimized thresholds.
    """
    rows = []

    for i, label in enumerate(LABELS):
        y_true = labels_arr[:, i]
        y_prob = probs_arr[:, i]

        # Default threshold = 0.5
        y_pred_def = (y_prob >= 0.5).astype(int)
        f1_def = f1_score(y_true, y_pred_def, zero_division=0)
        p_def = precision_score(y_true, y_pred_def, zero_division=0)
        r_def = recall_score(y_true, y_pred_def, zero_division=0)

        # Optimized threshold
        t_opt = thresholds[label]
        y_pred_opt = (y_prob >= t_opt).astype(int)
        f1_opt = f1_score(y_true, y_pred_opt, zero_division=0)
        p_opt = precision_score(y_true, y_pred_opt, zero_division=0)
        r_opt = recall_score(y_true, y_pred_opt, zero_division=0)

        roc = roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else float("nan")
        pr = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else float("nan")
        support = int(y_true.sum())

        rows.append({
            "label": label,
            "support": support,
            "AUC-ROC": roc,
            "PR-AUC": pr,
            "t=0.5 F1": f1_def,
            "t=0.5 P": p_def,
            "t=0.5 R": r_def,
            "opt_t": t_opt,
            "opt F1": f1_opt,
            "opt P": p_opt,
            "opt R": r_opt,
        })

    df = pd.DataFrame(rows)

    # Macro / Micro summary rows
    all_preds_def = (probs_arr >= 0.5).astype(int)
    thresh_arr = np.array([thresholds[label] for label in LABELS], dtype=np.float32)
    all_preds_opt = (probs_arr >= thresh_arr[None, :]).astype(int)

    macro_row = {
        "label": "MACRO",
        "support": int(labels_arr.sum()),
        "AUC-ROC": np.nanmean(df["AUC-ROC"].values),
        "PR-AUC": np.nanmean(df["PR-AUC"].values),
        "t=0.5 F1": f1_score(labels_arr, all_preds_def, average="macro", zero_division=0),
        "t=0.5 P": precision_score(labels_arr, all_preds_def, average="macro", zero_division=0),
        "t=0.5 R": recall_score(labels_arr, all_preds_def, average="macro", zero_division=0),
        "opt_t": np.nan,
        "opt F1": f1_score(labels_arr, all_preds_opt, average="macro", zero_division=0),
        "opt P": precision_score(labels_arr, all_preds_opt, average="macro", zero_division=0),
        "opt R": recall_score(labels_arr, all_preds_opt, average="macro", zero_division=0),
    }

    micro_row = {
        "label": "MICRO",
        "support": int(labels_arr.sum()),
        "AUC-ROC": roc_auc_score(labels_arr.ravel(), probs_arr.ravel()),
        "PR-AUC": average_precision_score(labels_arr.ravel(), probs_arr.ravel()),
        "t=0.5 F1": f1_score(labels_arr, all_preds_def, average="micro", zero_division=0),
        "t=0.5 P": precision_score(labels_arr, all_preds_def, average="micro", zero_division=0),
        "t=0.5 R": recall_score(labels_arr, all_preds_def, average="micro", zero_division=0),
        "opt_t": np.nan,
        "opt F1": f1_score(labels_arr, all_preds_opt, average="micro", zero_division=0),
        "opt P": precision_score(labels_arr, all_preds_opt, average="micro", zero_division=0),
        "opt R": recall_score(labels_arr, all_preds_opt, average="micro", zero_division=0),
    }

    df = pd.concat([df, pd.DataFrame([macro_row, micro_row])], ignore_index=True)
    return df


# ──────────────────────────────────────────────
# 2. ROC curves
# ──────────────────────────────────────────────

def plot_roc_curves(labels_arr: np.ndarray, probs_arr: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(labels_arr[:, i], probs_arr[:, i])
        score = roc_auc_score(labels_arr[:, i], probs_arr[:, i])
        ax.plot(fpr, tpr, label=f"{label} (AUC={score:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-label ROC Curves — BERT")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 3. Precision-Recall curves
# ──────────────────────────────────────────────

def plot_pr_curves(labels_arr: np.ndarray, probs_arr: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for i, label in enumerate(LABELS):
        prec, rec, _ = precision_recall_curve(labels_arr[:, i], probs_arr[:, i])
        score = average_precision_score(labels_arr[:, i], probs_arr[:, i])
        ax.plot(rec, prec, label=f"{label} (AP={score:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-label Precision-Recall Curves — BERT")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 4. Confusion matrices
# ──────────────────────────────────────────────

def plot_confusion_matrices(
    labels_arr: np.ndarray,
    probs_arr: np.ndarray,
    thresholds: dict[str, float],
    save_path: str,
):
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_MULTI)
    axes = axes.flatten()

    for i, label in enumerate(LABELS):
        t = thresholds[label]
        y_pred = (probs_arr[:, i] >= t).astype(int)
        cm = confusion_matrix(labels_arr[:, i], y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
        )
        axes[i].set_title(f"{label}  (t={t:.2f})", fontsize=11)
        axes[i].set_ylabel("True")
        axes[i].set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices (optimal thresholds) — BERT", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 5. Probability distribution histograms
# ──────────────────────────────────────────────

def plot_prob_distributions(
    labels_arr: np.ndarray,
    probs_arr: np.ndarray,
    thresholds: dict[str, float],
    save_path: str,
):
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_MULTI)
    axes = axes.flatten()

    for i, label in enumerate(LABELS):
        ax = axes[i]
        pos_probs = probs_arr[labels_arr[:, i] == 1, i]
        neg_probs = probs_arr[labels_arr[:, i] == 0, i]

        ax.hist(
            neg_probs,
            bins=80,
            alpha=0.6,
            label=f"Negative (n={len(neg_probs):,})",
            color="#85B7EB",
            density=True,
        )
        ax.hist(
            pos_probs,
            bins=80,
            alpha=0.6,
            label=f"Positive (n={len(pos_probs):,})",
            color="#E24B4A",
            density=True,
        )

        t = thresholds[label]
        ax.axvline(t, color="black", linestyle="--", linewidth=1.2, label=f"opt t={t:.2f}")
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="t=0.50")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper center")

    fig.suptitle("Probability Distributions (positive vs negative) — BERT", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 6. Threshold sensitivity curves
# ──────────────────────────────────────────────

def plot_threshold_sensitivity(
    labels_arr: np.ndarray,
    probs_arr: np.ndarray,
    thresholds: dict[str, float],
    save_path: str,
):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    t_range = np.arange(0.05, 0.96, 0.01)

    for i, label in enumerate(LABELS):
        f1_scores = []
        for t in t_range:
            preds = (probs_arr[:, i] >= t).astype(int)
            f1_scores.append(f1_score(labels_arr[:, i], preds, zero_division=0))

        ax.plot(t_range, f1_scores, label=label)

        # Mark optimal point
        t_opt = thresholds[label]
        f1_opt = f1_score(
            labels_arr[:, i],
            (probs_arr[:, i] >= t_opt).astype(int),
            zero_division=0,
        )
        ax.plot(t_opt, f1_opt, "o", markersize=6, color="black", zorder=5)
        ax.annotate(
            f"{t_opt:.2f}",
            (t_opt, f1_opt),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs Threshold per Label — BERT")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 7. Error analysis
# ──────────────────────────────────────────────

def error_analysis(
    texts: pd.Series,
    labels_arr: np.ndarray,
    probs_arr: np.ndarray,
    thresholds: dict[str, float],
    save_path: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    For each label, find the top-k false positives and false negatives
    ranked by distance from the threshold (i.e. most confident mistakes).
    """
    records = []

    for i, label in enumerate(LABELS):
        t = thresholds[label]
        y_true = labels_arr[:, i]
        y_prob = probs_arr[:, i]
        y_pred = (y_prob >= t).astype(int)

        # False Positives: predicted 1, true 0 — sort by highest prob
        fp_mask = (y_pred == 1) & (y_true == 0)
        fp_indices = np.where(fp_mask)[0]
        fp_sorted = fp_indices[np.argsort(-y_prob[fp_indices])][:top_k]

        for idx in fp_sorted:
            records.append({
                "label": label,
                "error_type": "FP",
                "prob": round(float(y_prob[idx]), 4),
                "threshold": t,
                "text": str(texts.iloc[idx])[:300],
            })

        # False Negatives: predicted 0, true 1 — sort by lowest prob
        fn_mask = (y_pred == 0) & (y_true == 1)
        fn_indices = np.where(fn_mask)[0]
        fn_sorted = fn_indices[np.argsort(y_prob[fn_indices])][:top_k]

        for idx in fn_sorted:
            records.append({
                "label": label,
                "error_type": "FN",
                "prob": round(float(y_prob[idx]), 4),
                "threshold": t,
                "text": str(texts.iloc[idx])[:300],
            })

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")

    # Brief console summary
    for label in LABELS:
        sub = df[df["label"] == label]
        n_fp = len(sub[sub["error_type"] == "FP"])
        n_fn = len(sub[sub["error_type"] == "FN"])
        print(f"    {label:15s}  FP examples: {n_fp}  |  FN examples: {n_fn}")

    return df


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BERT toxic comment classifier"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/bert_final/best_checkpoint",
        help="Path to the best BERT checkpoint directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace base model name",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="data/val_split.csv",
        help="Validation CSV (from split.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval_bert",
        help="Directory to save all evaluation outputs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────
    print("Loading model ...")
    model, tokenizer = load_trained_model(
        checkpoint_dir=args.checkpoint,
        base_model_name=args.base_model,
        device=args.device,
    )

    # ── Load optimal thresholds ───────────────
    thresh_path = os.path.join(args.checkpoint, "optimal_thresholds.json")
    with open(thresh_path) as f:
        thresholds = json.load(f)
    rounded_thresholds = {k: round(float(v), 2) for k, v in thresholds.items()}
    print(f"Loaded thresholds: {rounded_thresholds}")

    # ── Load validation data ──────────────────
    print("Loading validation data ...")
    val_df = pd.read_csv(args.val_csv)
    val_df["comment_text"] = val_df["comment_text"].fillna("").astype(str)
    texts = val_df["comment_text"]
    labels_arr = val_df[LABELS].values.astype(float)
    print(f"Validation samples: {len(val_df):,}")

    # ── Run inference ─────────────────────────
    print("Running inference ...")
    probs_arr = predict(
        model,
        tokenizer,
        texts.tolist(),
        batch_size=args.batch_size,
    )
    print(f"Predictions shape: {probs_arr.shape}")

    # Save raw probabilities for downstream use (e.g., ensemble comparison)
    np.save(os.path.join(args.output_dir, "val_probs_bert.npy"), probs_arr)
    np.save(os.path.join(args.output_dir, "val_labels.npy"), labels_arr)
    print(f"  Saved: val_probs_bert.npy, val_labels.npy")

    # ── 1. Metrics table ──────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    metrics_df = build_metrics_table(labels_arr, probs_arr, thresholds)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
    metrics_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Saved: {csv_path}")

    # ── 2. ROC curves ─────────────────────────
    print("\nGenerating plots ...")
    plot_roc_curves(
        labels_arr,
        probs_arr,
        os.path.join(args.output_dir, "roc_curves.png"),
    )

    # ── 3. PR curves ──────────────────────────
    plot_pr_curves(
        labels_arr,
        probs_arr,
        os.path.join(args.output_dir, "pr_curves.png"),
    )

    # ── 4. Confusion matrices ─────────────────
    plot_confusion_matrices(
        labels_arr,
        probs_arr,
        thresholds,
        os.path.join(args.output_dir, "confusion_matrices.png"),
    )

    # ── 5. Probability distributions ──────────
    plot_prob_distributions(
        labels_arr,
        probs_arr,
        thresholds,
        os.path.join(args.output_dir, "prob_distributions.png"),
    )

    # ── 6. Threshold sensitivity ──────────────
    plot_threshold_sensitivity(
        labels_arr,
        probs_arr,
        thresholds,
        os.path.join(args.output_dir, "threshold_sensitivity.png"),
    )

    # ── 7. Error analysis ─────────────────────
    print("\nError analysis ...")
    error_analysis(
        texts,
        labels_arr,
        probs_arr,
        thresholds,
        os.path.join(args.output_dir, "error_analysis.csv"),
        top_k=10,
    )

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
