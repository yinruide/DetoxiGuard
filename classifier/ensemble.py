"""
ensemble.py
-----------
Ensemble inference for the DetoxiGuard toxicity classifier.

This script loads the fine-tuned BERT and LLaMA+LoRA models, collects their
probability outputs on the shared validation set, and performs a per-label
joint grid search over (w_bert, threshold) to maximise F2 for each label
independently.  F2 (beta=2) weights recall twice as heavily as precision,
matching the guardrail use-case where missing toxic content is costlier
than over-flagging.

It produces a four-group comparison report:
    1. BERT tuned          (BERT probs + F2-searched thresholds, same range as ensembles)
    2. LLaMA tuned         (LLaMA probs + F2-searched thresholds, same range as ensembles)
    3. Ensemble uniform    (w=0.5 average + searched thresholds)
    4. Ensemble learned    (per-label learned w + threshold)

It also exposes a predict() interface for use by the LangGraph agent pipeline.

Threshold search range is capped at [0.05, 0.50] to favour recall — in this
guardrail context we prefer to flag borderline comments rather than miss them.

Usage:
    python ensemble.py \
        --bert_ckpt   outputs/bert/best_checkpoint \
        --bert_base   bert-base-uncased \
        --llama_ckpt  outputs/llama/best_checkpoint \
        --llama_base  meta-llama/Llama-3.2-1B \
        --val_csv     data/val_split.csv \
        --output_dir  outputs/ensemble

Dependencies:
    Same as train_bert.py + train_llama.py

Co-authored by Ruide Yin and Yanfu Wang
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ──────────────────────────────────────────────
# Config & Logging
# ──────────────────────────────────────────────

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
NUM_LABELS = len(LABELS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    thresholds: dict[str, float],
) -> dict:
    """
    Compute per-label and macro/micro metrics given probabilities and
    per-label thresholds.

    Returns a dict with the same schema used by train_bert / train_llama
    so that comparison is straightforward.
    """
    all_preds = np.zeros_like(all_probs, dtype=int)
    for i, label in enumerate(LABELS):
        all_preds[:, i] = (all_probs[:, i] >= thresholds[label]).astype(int)

    per_label_f1 = {}
    per_label_auc = {}
    per_label_pr_auc = {}
    per_label_precision = {}
    per_label_recall = {}

    for i, label in enumerate(LABELS):
        per_label_f1[label] = float(f1_score(
            all_labels[:, i], all_preds[:, i], zero_division=0
        ))
        per_label_precision[label] = float(precision_score(
            all_labels[:, i], all_preds[:, i], zero_division=0
        ))
        per_label_recall[label] = float(recall_score(
            all_labels[:, i], all_preds[:, i], zero_division=0
        ))
        if all_labels[:, i].sum() > 0:
            per_label_auc[label] = float(roc_auc_score(
                all_labels[:, i], all_probs[:, i]
            ))
            per_label_pr_auc[label] = float(average_precision_score(
                all_labels[:, i], all_probs[:, i]
            ))
        else:
            per_label_auc[label] = float("nan")
            per_label_pr_auc[label] = float("nan")

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    micro_f1 = float(f1_score(all_labels, all_preds, average="micro", zero_division=0))
    macro_auc = float(np.nanmean(list(per_label_auc.values())))
    macro_pr_auc = float(np.nanmean(list(per_label_pr_auc.values())))

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_auc": macro_auc,
        "macro_pr_auc": macro_pr_auc,
        "per_label_f1": per_label_f1,
        "per_label_auc": per_label_auc,
        "per_label_pr_auc": per_label_pr_auc,
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
    }


# ──────────────────────────────────────────────
# Per-label joint grid search
# ──────────────────────────────────────────────

def search_per_label_weight_and_threshold(
    p_bert: np.ndarray,
    p_llama: np.ndarray,
    labels: np.ndarray,
    w_range: np.ndarray | None = None,
    t_range: np.ndarray | None = None,
    beta: float = 2.0,
) -> dict[str, dict]:
    """
    For each label independently, find the (w_bert, threshold) pair that
    maximises F-beta (default beta=2, i.e. F2).

    F2 weights recall twice as heavily as precision, which matches the
    guardrail use-case: missing toxic content is costlier than over-flagging.
    This soft preference complements the hard constraint of capping the
    threshold search range at 0.50.

    The ensemble probability for label i is:
        p_ens_i = w * p_bert[:, i] + (1 - w) * p_llama[:, i]

    Returns:
        {label: {"w_bert": float, "w_llama": float, "threshold": float,
                  "f2": float, "f1": float}}
    """
    if w_range is None:
        w_range = np.arange(0.0, 1.01, 0.05)       # 21 points
    if t_range is None:
        t_range = np.arange(0.05, 0.51, 0.01)       # 46 points

    results: dict[str, dict] = {}

    for i, label in enumerate(LABELS):
        y_true = labels[:, i]
        best_fb, best_w, best_t = -1.0, 0.5, 0.5

        for w in w_range:
            p_ens = w * p_bert[:, i] + (1.0 - w) * p_llama[:, i]
            for t in t_range:
                preds = (p_ens >= t).astype(int)
                fb = fbeta_score(y_true, preds, beta=beta, zero_division=0)
                if fb > best_fb:
                    best_fb = fb
                    best_w = float(w)
                    best_t = float(t)

        # Also record F1 at the chosen (w, t) for reporting convenience.
        p_ens_best = best_w * p_bert[:, i] + (1.0 - best_w) * p_llama[:, i]
        preds_best = (p_ens_best >= best_t).astype(int)
        f1_at_best = float(f1_score(y_true, preds_best, zero_division=0))

        results[label] = {
            "w_bert": round(best_w, 4),
            "w_llama": round(1.0 - best_w, 4),
            "threshold": round(best_t, 4),
            "f2": round(best_fb, 4),
            "f1": round(f1_at_best, 4),
        }
        logger.info(
            f"  {label:15s}  w_bert={best_w:.2f}  threshold={best_t:.2f}  "
            f"F2={best_fb:.4f}  F1={f1_at_best:.4f}"
        )

    return results


def search_uniform_thresholds(
    p_bert: np.ndarray,
    p_llama: np.ndarray,
    labels: np.ndarray,
    t_range: np.ndarray | None = None,
    beta: float = 2.0,
) -> dict[str, float]:
    """
    With fixed w=0.5 (uniform average), search for the best per-label
    threshold by maximising F-beta. This serves as the naive ensemble baseline.
    """
    if t_range is None:
        t_range = np.arange(0.05, 0.51, 0.01)

    p_avg = 0.5 * p_bert + 0.5 * p_llama
    thresholds: dict[str, float] = {}

    for i, label in enumerate(LABELS):
        y_true = labels[:, i]
        best_fb, best_t = -1.0, 0.5

        for t in t_range:
            preds = (p_avg[:, i] >= t).astype(int)
            fb = fbeta_score(y_true, preds, beta=beta, zero_division=0)
            if fb > best_fb:
                best_fb = fb
                best_t = float(t)

        thresholds[label] = round(best_t, 4)
        logger.info(
            f"  {label:15s}  threshold={best_t:.2f}  F2={best_fb:.4f}  (uniform w=0.5)"
        )

    return thresholds


# ──────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────

def log_comparison(name: str, metrics: dict):
    """Pretty-print one group's metrics."""
    logger.info(f"── {name} ──")
    logger.info(
        f"  Macro-F1={metrics['macro_f1']:.4f}  Micro-F1={metrics['micro_f1']:.4f}  "
        f"Macro-AUC={metrics['macro_auc']:.4f}  Macro-PR-AUC={metrics['macro_pr_auc']:.4f}"
    )
    logger.info(f"  {'Label':15s} {'F1':>7s} {'P':>7s} {'R':>7s} {'AUC':>7s} {'PR-AUC':>7s}")
    for label in LABELS:
        logger.info(
            f"  {label:15s} "
            f"{metrics['per_label_f1'][label]:7.4f} "
            f"{metrics['per_label_precision'][label]:7.4f} "
            f"{metrics['per_label_recall'][label]:7.4f} "
            f"{metrics['per_label_auc'][label]:7.4f} "
            f"{metrics['per_label_pr_auc'][label]:7.4f}"
        )


# ──────────────────────────────────────────────
# Inference interface (used by pipeline.py)
# ──────────────────────────────────────────────

# Module-level cache so the models are loaded only once across repeated calls.
_ensemble_cache: dict = {}


def load_ensemble(
    bert_ckpt: str,
    bert_base: str,
    llama_ckpt: str,
    llama_base: str,
    ensemble_dir: str,
    device: str = "cuda",
):
    """
    Load both models and the saved ensemble weights / thresholds into the
    module-level cache.  Subsequent calls to predict() will reuse them.
    """
    # Lazy imports so that pipeline.py does not need torch at import time
    # if it only needs to inspect the module.
    import train_bert
    import train_llama

    logger.info("Loading BERT model for ensemble ...")
    model_bert, tok_bert = train_bert.load_trained_model(bert_ckpt, bert_base, device)

    logger.info("Loading LLaMA model for ensemble ...")
    model_llama, tok_llama = train_llama.load_trained_model(llama_ckpt, llama_base, device)

    weights_path = os.path.join(ensemble_dir, "ensemble_weights.json")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Ensemble weights not found: {weights_path}. "
            "Run ensemble.py first to generate them."
        )
    with open(weights_path) as f:
        ensemble_weights = json.load(f)

    _ensemble_cache.update({
        "model_bert": model_bert,
        "tok_bert": tok_bert,
        "model_llama": model_llama,
        "tok_llama": tok_llama,
        "ensemble_weights": ensemble_weights,
    })
    logger.info("Ensemble loaded and cached.")


def predict_probs(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Return fused ensemble probabilities of shape (N, 6).

    Uses the per-label w_bert weights from ensemble_weights.json to combine
    BERT and LLaMA predictions.  Does NOT apply thresholds — use this when
    the caller needs raw confidence scores (e.g. the agent pipeline).
    """
    import train_bert
    import train_llama

    if not _ensemble_cache:
        raise RuntimeError("Call load_ensemble() before predict_probs().")

    p_bert = train_bert.predict(
        _ensemble_cache["model_bert"],
        _ensemble_cache["tok_bert"],
        texts,
        batch_size=batch_size,
    )
    p_llama = train_llama.predict(
        _ensemble_cache["model_llama"],
        _ensemble_cache["tok_llama"],
        texts,
        batch_size=batch_size,
    )

    weights = _ensemble_cache["ensemble_weights"]
    p_ens = np.zeros_like(p_bert)
    for i, label in enumerate(LABELS):
        w = weights[label]["w_bert"]
        p_ens[:, i] = w * p_bert[:, i] + (1.0 - w) * p_llama[:, i]

    return p_ens.astype(np.float32)


def predict(texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (probs, binary_preds), both of shape (N, 6).

    This is the main entry point for pipeline.py:
        from classifier.ensemble import load_ensemble, predict
        load_ensemble(...)
        probs, preds = predict(["some comment"])
    """
    probs = predict_probs(texts, batch_size=batch_size)
    weights = _ensemble_cache["ensemble_weights"]

    preds = np.zeros_like(probs, dtype=int)
    for i, label in enumerate(LABELS):
        t = weights[label]["threshold"]
        preds[:, i] = (probs[:, i] >= t).astype(int)

    return probs, preds


# ──────────────────────────────────────────────
# Main — validation-set evaluation & weight search
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensemble BERT + LLaMA for toxic comment classification"
    )
    parser.add_argument("--bert_ckpt", type=str, default="outputs/bert_final/best_checkpoint",
                        help="Path to the fine-tuned BERT checkpoint directory")
    parser.add_argument("--bert_base", type=str, default="bert-base-uncased",
                        help="HuggingFace model ID for the BERT base model")
    parser.add_argument("--llama_ckpt", type=str, default="outputs/llama/best_checkpoint",
                        help="Path to the fine-tuned LLaMA+LoRA checkpoint directory")
    parser.add_argument("--llama_base", type=str, default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model ID for the LLaMA base model")
    parser.add_argument("--val_csv", type=str, default="data/val_split.csv",
                        help="Validation split generated by split.py")
    parser.add_argument("--output_dir", type=str, default="outputs/ensemble")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Lazy imports — keep module importable without torch for lightweight use.
    import train_bert
    import train_llama

    # ── 1. Load validation data ──────────────────
    logger.info(f"Loading validation data from {args.val_csv} ...")
    val_df = pd.read_csv(args.val_csv)
    val_df["comment_text"] = val_df["comment_text"].fillna("").astype(str)
    texts = val_df["comment_text"].tolist()
    gt_labels = val_df[LABELS].values.astype(int)   # (N, 6)
    logger.info(f"Validation samples: {len(texts):,}")

    # ── 2. Load models ───────────────────────────
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    logger.info(f"Device: {device}")

    logger.info("Loading BERT model ...")
    model_bert, tok_bert = train_bert.load_trained_model(
        args.bert_ckpt, args.bert_base, device
    )
    logger.info("Loading LLaMA model ...")
    model_llama, tok_llama = train_llama.load_trained_model(
        args.llama_ckpt, args.llama_base, device
    )

    # ── 3. Collect probability matrices ──────────
    logger.info("Running BERT inference on val set ...")
    p_bert = train_bert.predict(model_bert, tok_bert, texts, batch_size=args.batch_size)

    logger.info("Running LLaMA inference on val set ...")
    p_llama = train_llama.predict(model_llama, tok_llama, texts, batch_size=args.batch_size)

    logger.info(f"p_bert  shape={p_bert.shape}  range=[{p_bert.min():.4f}, {p_bert.max():.4f}]")
    logger.info(f"p_llama shape={p_llama.shape}  range=[{p_llama.min():.4f}, {p_llama.max():.4f}]")

    # ── 4. Search single-model baselines (F2, same range as ensembles) ──
    #    Re-search per-label thresholds for BERT-only and LLaMA-only using
    #    the same F2 objective and [0.05, 0.51) range used by the ensemble
    #    groups, so that all four groups are compared fairly.

    logger.info("═══ Searching F2-optimal thresholds for BERT-only baseline ═══")
    bert_search = search_per_label_weight_and_threshold(
        p_bert, p_llama, gt_labels, w_range=np.array([1.0]),
    )
    bert_thresholds = {label: bert_search[label]["threshold"] for label in LABELS}

    logger.info("═══ Searching F2-optimal thresholds for LLaMA-only baseline ═══")
    llama_search = search_per_label_weight_and_threshold(
        p_bert, p_llama, gt_labels, w_range=np.array([0.0]),
    )
    llama_thresholds = {label: llama_search[label]["threshold"] for label in LABELS}

    # ── 5. Compute single-model tuned metrics ────
    metrics_bert_tuned = compute_metrics(gt_labels, p_bert, bert_thresholds)
    metrics_llama_tuned = compute_metrics(gt_labels, p_llama, llama_thresholds)

    # ── 6. Ensemble uniform (w=0.5) + threshold search ──
    logger.info("═══ Searching thresholds for uniform ensemble (w=0.5) ═══")
    uniform_thresholds = search_uniform_thresholds(p_bert, p_llama, gt_labels)
    p_uniform = 0.5 * p_bert + 0.5 * p_llama
    metrics_ens_uniform = compute_metrics(gt_labels, p_uniform, uniform_thresholds)

    # ── 7. Ensemble learned — per-label joint (w, threshold) search ──
    logger.info("═══ Per-label joint (w, threshold) grid search ═══")
    learned_weights = search_per_label_weight_and_threshold(
        p_bert, p_llama, gt_labels
    )

    # Build the fused probability matrix and thresholds for metrics computation.
    p_learned = np.zeros_like(p_bert)
    learned_thresholds: dict[str, float] = {}
    for i, label in enumerate(LABELS):
        w = learned_weights[label]["w_bert"]
        p_learned[:, i] = w * p_bert[:, i] + (1.0 - w) * p_llama[:, i]
        learned_thresholds[label] = learned_weights[label]["threshold"]

    metrics_ens_learned = compute_metrics(gt_labels, p_learned, learned_thresholds)

    # ── 8. Comparison report ─────────────────────
    logger.info("")
    logger.info("══════════════════════════════════════════════════")
    logger.info("          FOUR-GROUP COMPARISON REPORT            ")
    logger.info("══════════════════════════════════════════════════")
    log_comparison("1. BERT tuned", metrics_bert_tuned)
    log_comparison("2. LLaMA tuned", metrics_llama_tuned)
    log_comparison("3. Ensemble uniform (w=0.5)", metrics_ens_uniform)
    log_comparison("4. Ensemble learned (per-label w)", metrics_ens_learned)

    # ── 9. Save artefacts ────────────────────────
    weights_path = os.path.join(args.output_dir, "ensemble_weights.json")
    with open(weights_path, "w") as f:
        json.dump(learned_weights, f, indent=2)
    logger.info(f"Saved ensemble weights → {weights_path}")

    uniform_thresh_path = os.path.join(args.output_dir, "uniform_thresholds.json")
    with open(uniform_thresh_path, "w") as f:
        json.dump(uniform_thresholds, f, indent=2)
    logger.info(f"Saved uniform thresholds → {uniform_thresh_path}")

    comparison = {
        "bert_tuned": metrics_bert_tuned,
        "llama_tuned": metrics_llama_tuned,
        "ensemble_uniform": metrics_ens_uniform,
        "ensemble_learned": metrics_ens_learned,
    }
    comparison_path = os.path.join(args.output_dir, "val_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved comparison report → {comparison_path}")

    # ── 10. Summary table ────────────────────────
    logger.info("")
    logger.info("═══ MACRO SUMMARY ═══")
    logger.info(f"  {'Method':<30s} {'Macro-F1':>9s} {'Micro-F1':>9s} {'M-AUC':>9s} {'M-PR-AUC':>9s}")
    for name, m in [
        ("BERT tuned", metrics_bert_tuned),
        ("LLaMA tuned", metrics_llama_tuned),
        ("Ensemble uniform", metrics_ens_uniform),
        ("Ensemble learned", metrics_ens_learned),
    ]:
        logger.info(
            f"  {name:<30s} {m['macro_f1']:9.4f} {m['micro_f1']:9.4f} "
            f"{m['macro_auc']:9.4f} {m['macro_pr_auc']:9.4f}"
        )

    logger.info("")
    logger.info("═══ PER-LABEL LEARNED WEIGHTS ═══")
    logger.info(f"  {'Label':15s} {'w_bert':>7s} {'w_llama':>8s} {'thresh':>7s} {'F2':>7s} {'F1':>7s}")
    for label in LABELS:
        lw = learned_weights[label]
        logger.info(
            f"  {label:15s} {lw['w_bert']:7.2f} {lw['w_llama']:8.2f} "
            f"{lw['threshold']:7.2f} {lw['f2']:7.4f} {lw['f1']:7.4f}"
        )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()