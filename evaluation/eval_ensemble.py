"""
eval_ensemble.py
----------------
Standalone evaluation script for the DetoxiGuard BERT+LLaMA ensemble
classifier. Loads the saved ensemble weights, runs inference on the shared
validation set, and produces:

    1. Summary metrics table  (console + CSV)
    2. Per-label ROC curves
    3. Per-label Precision-Recall curves
    4. Confusion matrices (2×2 per label, learned thresholds)
    5. Probability distribution histograms (positive vs negative)
    6. Threshold sensitivity curves (F1 vs threshold per label)
    7. Error analysis  (top FP / FN examples per label)
    8. Ensemble weight summary table (CSV)

All figures are saved to  evaluation/eval_ensemble_results/

Usage (inside Singularity on HPC with GPU):
    python evaluation/eval_ensemble.py \
        --bert_ckpt    outputs/bert_final/best_checkpoint \
        --bert_base    bert-base-uncased \
        --llama_ckpt   outputs/llama_lora/best_checkpoint \
        --llama_base   meta-llama/Llama-3.2-1B \
        --ensemble_dir outputs/ensemble \
        --val_csv      data/val_split.csv \
        --output_dir   evaluation/eval_ensemble_results
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

def plot_roc_curves(labels_arr, probs_arr, save_path):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(labels_arr[:, i], probs_arr[:, i])
        score = roc_auc_score(labels_arr[:, i], probs_arr[:, i])
        ax.plot(fpr, tpr, label=f"{label} (AUC={score:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-label ROC Curves — Ensemble")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 3. Precision-Recall curves
# ──────────────────────────────────────────────

def plot_pr_curves(labels_arr, probs_arr, save_path):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for i, label in enumerate(LABELS):
        prec, rec, _ = precision_recall_curve(labels_arr[:, i], probs_arr[:, i])
        score = average_precision_score(labels_arr[:, i], probs_arr[:, i])
        ax.plot(rec, prec, label=f"{label} (AP={score:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-label Precision-Recall Curves — Ensemble")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 4. Confusion matrices
# ──────────────────────────────────────────────

def plot_confusion_matrices(labels_arr, probs_arr, thresholds, save_path):
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
    fig.suptitle("Confusion Matrices (learned thresholds) — Ensemble", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 5. Probability distribution histograms
# ──────────────────────────────────────────────

def plot_prob_distributions(labels_arr, probs_arr, thresholds, save_path):
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

    fig.suptitle("Probability Distributions (positive vs negative) — Ensemble", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 6. Threshold sensitivity curves
# ──────────────────────────────────────────────

def plot_threshold_sensitivity(labels_arr, probs_arr, thresholds, save_path):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    # Ensemble threshold search in ensemble.py is capped to [0.05, 0.50].
    t_range = np.arange(0.05, 0.51, 0.01)

    for i, label in enumerate(LABELS):
        f1_scores = []
        for t in t_range:
            preds = (probs_arr[:, i] >= t).astype(int)
            f1_scores.append(f1_score(labels_arr[:, i], preds, zero_division=0))
        ax.plot(t_range, f1_scores, label=label)

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
    ax.set_title("F1 vs Threshold per Label — Ensemble")
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
                        default="outputs/llama_lora/best_checkpoint",
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
    parser.add_argument("--output_dir", type=str,
                        default="evaluation/eval_ensemble_results",
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

    print("Loading validation data ...")
    val_df = pd.read_csv(args.val_csv)
    val_df["comment_text"] = val_df["comment_text"].fillna("").astype(str)
    texts = val_df["comment_text"]
    labels_arr = val_df[LABELS].values.astype(float)
    print(f"Validation samples: {len(val_df):,}")

    print("Running ensemble inference ...")
    probs_arr = predict_probs(texts.tolist(), batch_size=args.batch_size)
    print(f"Predictions shape: {probs_arr.shape}")

    np.save(os.path.join(args.output_dir, "val_probs_ensemble.npy"), probs_arr)
    np.save(os.path.join(args.output_dir, "val_labels.npy"), labels_arr)
    print("  Saved: val_probs_ensemble.npy, val_labels.npy")

    print("\n" + "=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    metrics_df = build_metrics_table(labels_arr, probs_arr, thresholds)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
    metrics_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Saved: {csv_path}")

    print("\nSaving ensemble weight summary ...")
    save_weight_summary(
        ensemble_weights,
        os.path.join(args.output_dir, "ensemble_weight_summary.csv"),
    )

    print("\nGenerating plots ...")
    plot_roc_curves(
        labels_arr,
        probs_arr,
        os.path.join(args.output_dir, "roc_curves.png"),
    )
    plot_pr_curves(
        labels_arr,
        probs_arr,
        os.path.join(args.output_dir, "pr_curves.png"),
    )
    plot_confusion_matrices(
        labels_arr,
        probs_arr,
        thresholds,
        os.path.join(args.output_dir, "confusion_matrices.png"),
    )
    plot_prob_distributions(
        labels_arr,
        probs_arr,
        thresholds,
        os.path.join(args.output_dir, "prob_distributions.png"),
    )
    plot_threshold_sensitivity(
        labels_arr,
        probs_arr,
        thresholds,
        os.path.join(args.output_dir, "threshold_sensitivity.png"),
    )

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
