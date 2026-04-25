"""
eval_ensemble.py
----------------
Standalone evaluation script for the DetoxiGuard BERT+LLaMA ensemble
classifier. Loads the saved ensemble weights, runs inference on BOTH
the validation set and the held-out test set, and produces per-split:

    1. Summary metrics table  (console + CSV)
    2. Per-label ROC curves
    3. Per-label Precision-Recall curves
    4. Confusion matrices (2×2 per label, learned thresholds)
    5. Probability distribution histograms (positive vs negative)
    6. Threshold sensitivity curves (F1 vs threshold per label)
    7. Error analysis  (top FP / FN examples per label)
    8. Ensemble weight summary table (CSV)

Val results  → {output_dir}/val/
Test results → {output_dir}/test/

Usage (inside Singularity on HPC with GPU):
    python evaluation/eval_ensemble.py \
        --bert_ckpt    outputs/bert_final/best_checkpoint \
        --bert_base    bert-base-uncased \
        --llama_ckpt   outputs/llama_lora/best_checkpoint \
        --llama_base   meta-llama/Llama-3.2-1B \
        --ensemble_dir outputs/ensemble \
        --val_csv      data/val_split.csv \
        --test_csv     data/test_split.csv \
        --output_dir   evaluation/eval_ensemble_results

Co-authored by Ruide Yin and Yanfu Wang
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Import from project modules ────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLASSIFIER_DIR = os.path.join(REPO_ROOT, "classifier")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.isdir(CLASSIFIER_DIR) and CLASSIFIER_DIR not in sys.path:
    # ensemble.py uses lazy imports like `import train_bert`, so we also add the
    # classifier directory itself to sys.path to make those imports resolve.
    sys.path.insert(0, CLASSIFIER_DIR)

from classifier.ensemble import LABELS, load_ensemble, predict_probs

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

FIGSIZE_MULTI = (18, 10)
FIGSIZE_SINGLE = (10, 7)
DPI = 150


# ──────────────────────────────────────────────
# 1. Summary metrics table
# ──────────────────────────────────────────────

def build_metrics_table(labels_arr: np.ndarray, probs_arr: np.ndarray,
                        thresholds: dict[str, float]) -> pd.DataFrame:
    """
    Build a DataFrame with per-label and macro/micro rows, comparing
    default (t=0.5) vs learned thresholds.
    """
    rows = []
    for i, label in enumerate(LABELS):
        y_true = labels_arr[:, i]
        y_prob = probs_arr[:, i]

        y_pred_def = (y_prob >= 0.5).astype(int)
        f1_def = f1_score(y_true, y_pred_def, zero_division=0)
        p_def = precision_score(y_true, y_pred_def, zero_division=0)
        r_def = recall_score(y_true, y_pred_def, zero_division=0)

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

    all_preds_def = (probs_arr >= 0.5).astype(int)
    thresh_arr = np.array([thresholds[l] for l in LABELS])
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

def plot_roc_curves(labels_arr, probs_arr, save_path, split_name=""):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(labels_arr[:, i], probs_arr[:, i])
        score = roc_auc_score(labels_arr[:, i], probs_arr[:, i])
        ax.plot(fpr, tpr, label=f"{label} (AUC={score:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Per-label ROC Curves — Ensemble ({split_name})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 3. Precision-Recall curves
# ──────────────────────────────────────────────

def plot_pr_curves(labels_arr, probs_arr, save_path, split_name=""):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for i, label in enumerate(LABELS):
        prec, rec, _ = precision_recall_curve(labels_arr[:, i], probs_arr[:, i])
        score = average_precision_score(labels_arr[:, i], probs_arr[:, i])
        ax.plot(rec, prec, label=f"{label} (AP={score:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Per-label Precision-Recall Curves — Ensemble ({split_name})")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 4. Confusion matrices
# ──────────────────────────────────────────────

def plot_confusion_matrices(labels_arr, probs_arr, thresholds, save_path, split_name=""):
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
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            ax=axes[i],
        )
        axes[i].set_title(f"{label} (t={t:.2f})")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    fig.suptitle(f"Confusion Matrices (learned thresholds) — Ensemble ({split_name})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 5. Probability distribution histograms
# ──────────────────────────────────────────────

def plot_prob_distributions(labels_arr, probs_arr, thresholds, save_path, split_name=""):
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_MULTI)
    axes = axes.flatten()
    for i, label in enumerate(LABELS):
        y_true = labels_arr[:, i]
        y_prob = probs_arr[:, i]
        t = thresholds[label]

        axes[i].hist(
            y_prob[y_true == 0], bins=50, alpha=0.5, label="Negative",
            color="steelblue", density=True,
        )
        axes[i].hist(
            y_prob[y_true == 1], bins=50, alpha=0.5, label="Positive",
            color="coral", density=True,
        )
        axes[i].axvline(t, color="red", linestyle="--", label=f"t={t:.2f}")
        axes[i].set_title(label)
        axes[i].set_xlabel("Predicted probability")
        axes[i].legend(fontsize=8)

    fig.suptitle(f"Probability Distributions — Ensemble ({split_name})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 6. Threshold sensitivity curves
# ──────────────────────────────────────────────

def plot_threshold_sensitivity(labels_arr, probs_arr, thresholds, save_path, split_name=""):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    t_range = np.arange(0.05, 0.96, 0.01)

    for i, label in enumerate(LABELS):
        y_true = labels_arr[:, i]
        y_prob = probs_arr[:, i]
        f1s = []
        for t in t_range:
            y_pred = (y_prob >= t).astype(int)
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
        ax.plot(t_range, f1s, label=label)

        t_opt = thresholds[label]
        f1_at_opt = f1_score(y_true, (y_prob >= t_opt).astype(int), zero_division=0)
        ax.scatter([t_opt], [f1_at_opt], marker="*", s=100, zorder=5)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"F1 vs Threshold per Label — Ensemble ({split_name})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 7. Error analysis
# ──────────────────────────────────────────────

def error_analysis(texts: pd.Series, labels_arr, probs_arr,
                   thresholds, save_path, top_k: int = 10):
    """
    For each label, find the top-k false positives and false negatives
    ranked by confidence.
    """
    records = []
    for i, label in enumerate(LABELS):
        t = thresholds[label]
        y_true = labels_arr[:, i]
        y_prob = probs_arr[:, i]
        y_pred = (y_prob >= t).astype(int)

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

    for label in LABELS:
        sub = df[df["label"] == label]
        n_fp = len(sub[sub["error_type"] == "FP"])
        n_fn = len(sub[sub["error_type"] == "FN"])
        print(f"    {label:15s}  FP examples: {n_fp}  |  FN examples: {n_fn}")

    return df


# ──────────────────────────────────────────────
# 8. Ensemble weight summary
# ──────────────────────────────────────────────

def save_weight_summary(weights: dict, save_path: str) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        row = {"label": label}
        row.update(weights[label])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, float_format="%.4f")
    print(f"  Saved: {save_path}")
    return df


# ──────────────────────────────────────────────
# Per-split evaluation driver
# ──────────────────────────────────────────────

def evaluate_split(
    csv_path: str,
    split_name: str,
    output_dir: str,
    thresholds: dict[str, float],
    ensemble_weights: dict,
    batch_size: int,
):
    """Run the full evaluation suite on one data split."""

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  Evaluating on: {split_name}  ({csv_path})")
    print(f"{'#' * 60}")

    df = pd.read_csv(csv_path)
    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    texts = df["comment_text"]
    labels_arr = df[LABELS].values.astype(float)
    print(f"  Samples: {len(df):,}")

    print("  Running ensemble inference ...")
    probs_arr = predict_probs(texts.tolist(), batch_size=batch_size)
    print(f"  Predictions shape: {probs_arr.shape}")

    np.save(os.path.join(split_dir, f"{split_name}_probs_ensemble.npy"), probs_arr)
    np.save(os.path.join(split_dir, f"{split_name}_labels.npy"), labels_arr)
    print(f"  Saved: {split_name}_probs_ensemble.npy, {split_name}_labels.npy")

    # ── Metrics table ──
    print(f"\n{'=' * 60}")
    print(f"SUMMARY METRICS ({split_name})")
    print("=" * 60)
    metrics_df = build_metrics_table(labels_arr, probs_arr, thresholds)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    csv_out = os.path.join(split_dir, "metrics_summary.csv")
    metrics_df.to_csv(csv_out, index=False, float_format="%.4f")
    print(f"\n  Saved: {csv_out}")

    # ── Weight summary ──
    print("\n  Saving ensemble weight summary ...")
    save_weight_summary(
        ensemble_weights,
        os.path.join(split_dir, "ensemble_weight_summary.csv"),
    )

    # ── Plots ──
    print("\n  Generating plots ...")
    plot_roc_curves(
        labels_arr, probs_arr,
        os.path.join(split_dir, "roc_curves.png"),
        split_name=split_name,
    )
    plot_pr_curves(
        labels_arr, probs_arr,
        os.path.join(split_dir, "pr_curves.png"),
        split_name=split_name,
    )
    plot_confusion_matrices(
        labels_arr, probs_arr, thresholds,
        os.path.join(split_dir, "confusion_matrices.png"),
        split_name=split_name,
    )
    plot_prob_distributions(
        labels_arr, probs_arr, thresholds,
        os.path.join(split_dir, "prob_distributions.png"),
        split_name=split_name,
    )
    plot_threshold_sensitivity(
        labels_arr, probs_arr, thresholds,
        os.path.join(split_dir, "threshold_sensitivity.png"),
        split_name=split_name,
    )

    # ── Error analysis ──
    print(f"\n  Error analysis ({split_name}) ...")
    error_analysis(
        texts, labels_arr, probs_arr, thresholds,
        os.path.join(split_dir, "error_analysis.csv"),
        top_k=10,
    )

    print(f"\n  {split_name} outputs saved to: {split_dir}/")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BERT+LLaMA ensemble toxic comment classifier"
    )
    parser.add_argument("--bert_ckpt", type=str,
                        default="outputs/bert_final/best_checkpoint",
                        help="Path to the fine-tuned BERT checkpoint directory")
    parser.add_argument("--bert_base", type=str,
                        default="bert-base-uncased",
                        help="HuggingFace model ID for the BERT base model")
    parser.add_argument("--llama_ckpt", type=str,
                        default="outputs/llama/best_checkpoint",
                        help="Path to the fine-tuned LLaMA+LoRA checkpoint directory")
    parser.add_argument("--llama_base", type=str,
                        default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model ID for the LLaMA base model")
    parser.add_argument("--ensemble_dir", type=str,
                        default="outputs/ensemble",
                        help="Directory containing ensemble_weights.json")
    parser.add_argument("--val_csv", type=str,
                        default="data/val_split.csv",
                        help="Validation CSV (from split.py)")
    parser.add_argument("--test_csv", type=str,
                        default="data/test_split.csv",
                        help="Held-out test CSV (from split.py)")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/eval_ensemble_results",
                        help="Directory to save all evaluation outputs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ensemble ...")
    load_ensemble(
        bert_ckpt=args.bert_ckpt,
        bert_base=args.bert_base,
        llama_ckpt=args.llama_ckpt,
        llama_base=args.llama_base,
        ensemble_dir=args.ensemble_dir,
        device=args.device,
    )

    weights_path = os.path.join(args.ensemble_dir, "ensemble_weights.json")
    with open(weights_path) as f:
        ensemble_weights = json.load(f)
    thresholds = {label: float(ensemble_weights[label]["threshold"]) for label in LABELS}
    print(
        "Loaded thresholds:",
        {k: round(v, 2) for k, v in thresholds.items()},
    )

    # ── Evaluate on val ──
    evaluate_split(
        csv_path=args.val_csv,
        split_name="val",
        output_dir=args.output_dir,
        thresholds=thresholds,
        ensemble_weights=ensemble_weights,
        batch_size=args.batch_size,
    )

    # ── Evaluate on test ──
    evaluate_split(
        csv_path=args.test_csv,
        split_name="test",
        output_dir=args.output_dir,
        thresholds=thresholds,
        ensemble_weights=ensemble_weights,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()