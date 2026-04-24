"""
eval_pipeline.py
----------------
End-to-end evaluation of the DetoxiGuard LangGraph pipeline.

Three test groups (all drawn from the HELD-OUT test set):
  1. test_toxic    — toxic samples from test_split.csv (stratified by label)
  2. test_clean    — clean samples from test_split.csv
  3. error_analysis — FP/FN boundary cases from ensemble error analysis

Runs each sample through the full detect→revise→re-score pipeline and
produces aggregate metrics + a per-sample detail log.

Usage (on HPC, inside Singularity with GPU):
    export OPENAI_API_KEY=sk-...
    python agent/eval_pipeline.py \
        --test_csv      data/test_split.csv \
        --error_csv     outputs/ensemble/error_analysis.csv \
        --bert_ckpt     outputs/bert_final/best_checkpoint \
        --llama_ckpt    outputs/llama_lora/best_checkpoint \
        --ensemble_dir  outputs/ensemble \
        --n_toxic 50 --n_clean 50 \
        --output_dir    outputs/pipeline_eval

Outputs:
    pipeline_eval_details.json   – per-sample record
    pipeline_eval_summary.json   – aggregate metrics
    pipeline_eval_samples.csv    – sampled test set (reproducibility)
    pipeline_eval.log            – full log

Co-authored by Ruide Yin and Yanfu Wang
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Path setup (mirrors pipeline.py conventions) ─────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_CLASSIFIER_DIR = os.path.join(_REPO_ROOT, "classifier")
if _CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, _CLASSIFIER_DIR)

from classifier.ensemble import LABELS, load_ensemble, predict
from agent.pipeline import init_ensemble, run_pipeline

logger = logging.getLogger(__name__)

SEED = 49458345  # team seed for reproducibility


# ──────────────────────────────────────────────
# 1. Test-set construction
# ──────────────────────────────────────────────

def sample_test_set(
    test_csv: str,
    n_toxic: int = 50,
    n_clean: int = 50,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Stratified sample from test_split.csv.
    Returns a DataFrame with columns: comment_text, any_toxic, source,
    plus the 6 ground-truth label columns and pred_* columns.
    """
    df = pd.read_csv(test_csv)
    df["comment_text"] = df["comment_text"].fillna("").astype(str)

    logger.info(f"Loaded {len(df):,} test samples. Running ensemble predict ...")
    texts = df["comment_text"].tolist()
    probs, preds = predict(texts, batch_size=64)

    for i, label in enumerate(LABELS):
        df[f"pred_{label}"] = preds[:, i]
    df["any_toxic"] = preds.any(axis=1).astype(int)

    toxic_df = df[df["any_toxic"] == 1].copy()
    clean_df = df[df["any_toxic"] == 0].copy()
    logger.info(f"Test toxic: {len(toxic_df):,} | Test clean: {len(clean_df):,}")

    rng = np.random.RandomState(seed)

    # ── Toxic: ensure rare-label coverage ──
    rare_labels = ["threat", "identity_hate", "severe_toxic"]
    selected_indices = set()
    for label in rare_labels:
        pool = toxic_df[toxic_df[f"pred_{label}"] == 1]
        if len(pool) > 0:
            n_pick = min(5, len(pool))
            picked = pool.sample(n=n_pick, random_state=rng)
            selected_indices.update(picked.index.tolist())

    remaining_needed = max(0, n_toxic - len(selected_indices))
    remaining_pool = toxic_df[~toxic_df.index.isin(selected_indices)]
    if remaining_needed > 0 and len(remaining_pool) > 0:
        n_fill = min(remaining_needed, len(remaining_pool))
        filled = remaining_pool.sample(n=n_fill, random_state=rng)
        selected_indices.update(filled.index.tolist())

    toxic_sample = toxic_df.loc[list(selected_indices)].copy()
    toxic_sample["source"] = "test_toxic"

    # ── Clean: random sample ──
    n_clean_actual = min(n_clean, len(clean_df))
    clean_sample = clean_df.sample(n=n_clean_actual, random_state=rng).copy()
    clean_sample["source"] = "test_clean"

    result = pd.concat([toxic_sample, clean_sample], ignore_index=True)

    logger.info("Test toxic sample label coverage:")
    for label in LABELS:
        count = int(toxic_sample[f"pred_{label}"].sum())
        logger.info(f"  {label:15s}: {count}")

    return result


def load_error_analysis(error_csv: str) -> pd.DataFrame:
    """
    Load error_analysis.csv (columns: label, error_type, prob, threshold, text).
    Returns a DataFrame aligned with the test-set format:
      comment_text, any_toxic, source, error_type, error_label
    """
    df = pd.read_csv(error_csv)
    df = df.rename(columns={"text": "comment_text"})
    df["comment_text"] = df["comment_text"].fillna("").astype(str)

    # Deduplicate — the same text can appear under multiple labels
    df = df.drop_duplicates(subset=["comment_text", "error_type"]).reset_index(drop=True)

    # For FP: the ensemble wrongly flagged it → pipeline WILL trigger revise
    #   (any_toxic = 1 from the ensemble's perspective)
    # For FN: the ensemble missed it → pipeline will NOT trigger revise
    #   (any_toxic = 0 from the ensemble's perspective)
    df["any_toxic"] = (df["error_type"] == "FP").astype(int)
    df["source"] = "error_" + df["error_type"].str.lower()   # error_fp / error_fn
    df["error_label"] = df["label"]

    logger.info(
        f"Loaded error_analysis: {len(df)} samples "
        f"(FP={int((df['error_type']=='FP').sum())}, "
        f"FN={int((df['error_type']=='FN').sum())})"
    )

    return df


def build_test_set(
    test_csv: str,
    error_csv: str | None,
    n_toxic: int,
    n_clean: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Combine test samples + error analysis into a single shuffled test set."""
    parts = [sample_test_set(test_csv, n_toxic, n_clean, seed)]

    if error_csv and os.path.exists(error_csv):
        parts.append(load_error_analysis(error_csv))
    elif error_csv:
        logger.warning(f"Error analysis CSV not found: {error_csv}, skipping.")

    combined = pd.concat(parts, ignore_index=True)

    # Fill missing columns so all rows have the same schema
    for col in ["source", "error_type", "error_label"]:
        if col not in combined.columns:
            combined[col] = ""
    combined["source"] = combined["source"].fillna("")
    combined["error_type"] = combined["error_type"].fillna("")
    combined["error_label"] = combined["error_label"].fillna("")

    # Shuffle
    rng = np.random.RandomState(seed)
    combined = combined.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    # Summary
    logger.info("Test set composition:")
    for src, count in combined["source"].value_counts().items():
        logger.info(f"  {src:15s}: {count}")
    logger.info(f"  {'TOTAL':15s}: {len(combined)}")

    return combined


# ──────────────────────────────────────────────
# 2. Run pipeline on each sample
# ──────────────────────────────────────────────

def evaluate_pipeline(
    test_df: pd.DataFrame,
    max_iterations: int = 5,
) -> list[dict]:
    """Run each sample through the pipeline and collect detailed results."""
    results = []
    total = len(test_df)

    for idx, row in test_df.iterrows():
        text = row["comment_text"]
        source = row.get("source", "")
        is_initially_toxic = bool(row["any_toxic"])

        # Ground-truth labels from the dataset (may be absent for error_analysis rows)
        gt_labels = {}
        for label in LABELS:
            if label in row and pd.notna(row[label]):
                try:
                    gt_labels[label] = int(row[label])
                except (ValueError, TypeError):
                    pass

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Sample {idx+1}/{total} | source={source} | "
            f"initially_toxic={is_initially_toxic}"
        )
        logger.info(f"Text preview: {text[:120]}...")

        t0 = time.time()
        try:
            state = run_pipeline(text, max_iterations=max_iterations)
        except Exception as exc:
            logger.error(f"Pipeline error on sample {idx}: {exc}")
            results.append({
                "sample_id": int(idx),
                "source": source,
                "text_preview": text[:200],
                "initial_is_toxic": is_initially_toxic,
                "error": str(exc),
                "outcome": "error",
            })
            continue
        elapsed = time.time() - t0

        # Determine outcome
        if not is_initially_toxic:
            outcome = "clean_pass"
        elif state.get("gave_up", False) or state["final_output"].startswith("Sorry, your comment"):
            outcome = "fallback"
        elif not state["is_toxic"]:
            outcome = "corrected"
        else:
            outcome = "fallback"

        # Collect triggered labels from revision_history
        history = state.get("revision_history", [])
        initial_triggered = history[0].get("triggered_labels", {}) if history else {}
        final_triggered = history[-1].get("triggered_labels", {}) if history else {}

        record = {
            "sample_id": int(idx),
            "source": source,
            "error_type": row.get("error_type", ""),
            "error_label": row.get("error_label", ""),
            "text_preview": text[:200],
            "initial_is_toxic": is_initially_toxic,
            "gt_labels": gt_labels,
            "initial_triggered_labels": initial_triggered,
            "final_triggered_labels": final_triggered,
            "num_iterations": state["iteration"],
            "final_is_toxic": state["is_toxic"],
            "outcome": outcome,
            "was_modified": state["was_modified"],
            "final_output_preview": state["final_output"][:200],
            "elapsed_seconds": round(elapsed, 2),
        }
        results.append(record)

        logger.info(
            f"  outcome={outcome} | iterations={state['iteration']} | "
            f"elapsed={elapsed:.1f}s"
        )

    return results


# ──────────────────────────────────────────────
# 3. Aggregate metrics
# ──────────────────────────────────────────────

def _group_metrics(results: list[dict], group_name: str) -> dict:
    """Compute metrics for a group of results that are initially toxic (or FP)."""
    valid = [r for r in results if r.get("outcome") != "error"]
    if not valid:
        return {"n_samples": 0}

    n = len(valid)
    n_corrected = sum(1 for r in valid if r["outcome"] == "corrected")
    n_fallback = sum(1 for r in valid if r["outcome"] == "fallback")
    avg_iter = float(np.mean([r["num_iterations"] for r in valid]))

    iter_dist = defaultdict(int)
    for r in valid:
        iter_dist[r["num_iterations"]] += 1

    # Per-label correction
    per_label = {}
    for label in LABELS:
        triggered = [r for r in valid if label in r.get("initial_triggered_labels", {})]
        if triggered:
            fixed = sum(1 for r in triggered if label not in r.get("final_triggered_labels", {}))
            per_label[label] = {
                "initially_triggered": len(triggered),
                "corrected": fixed,
                "correction_rate": round(fixed / len(triggered), 4),
            }
        else:
            per_label[label] = {"initially_triggered": 0, "corrected": 0, "correction_rate": None}

    return {
        "n_samples": n,
        "correction_success_rate": round(n_corrected / n, 4) if n else None,
        "fallback_rate": round(n_fallback / n, 4) if n else None,
        "avg_iterations": round(avg_iter, 2),
        "iteration_distribution": dict(sorted(iter_dist.items())),
        "per_label_correction": per_label,
    }


def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate metrics across all groups."""

    # ── Split by source ──
    test_toxic = [r for r in results if r["source"] == "test_toxic"]
    test_clean = [r for r in results if r["source"] == "test_clean"]
    error_fp   = [r for r in results if r["source"] == "error_fp"]
    error_fn   = [r for r in results if r["source"] == "error_fn"]

    # All toxic-flagged samples combined (test_toxic + error_fp)
    all_toxic = [r for r in results if r["initial_is_toxic"] and r.get("outcome") != "error"]

    # ── Test toxic group ──
    test_toxic_metrics = _group_metrics(test_toxic, "test_toxic")

    # ── Error FP group (ensemble wrongly flagged → pipeline tries to revise) ──
    error_fp_metrics = _group_metrics(error_fp, "error_fp")

    # ── Combined toxic (test_toxic + error_fp) ──
    combined_toxic_metrics = _group_metrics(all_toxic, "all_toxic")

    # ── Test clean group ──
    clean_valid = [r for r in test_clean if r.get("outcome") != "error"]
    n_clean = len(clean_valid)
    n_preserved = sum(1 for r in clean_valid if not r["was_modified"])

    # ── Error FN group (ensemble missed → pipeline won't catch either) ──
    fn_valid = [r for r in error_fn if r.get("outcome") != "error"]
    n_fn = len(fn_valid)
    # How many did the pipeline actually catch? (any_toxic was 0 based on
    # original ensemble pred, but re-scoring in the pipeline may differ
    # if the text happens to trigger on a different label)
    fn_caught = sum(1 for r in fn_valid if r.get("was_modified", False))

    errors = [r for r in results if r.get("outcome") == "error"]

    summary = {
        "total_samples": len(results),
        "errors": len(errors),

        "test_toxic": test_toxic_metrics,
        "error_fp": error_fp_metrics,
        "combined_toxic": combined_toxic_metrics,

        "test_clean": {
            "n_samples": n_clean,
            "preservation_rate": round(n_preserved / n_clean, 4) if n_clean else None,
            "false_trigger_rate": round((n_clean - n_preserved) / n_clean, 4) if n_clean else None,
        },

        "error_fn": {
            "n_samples": n_fn,
            "caught_by_pipeline": fn_caught,
            "miss_rate": round((n_fn - fn_caught) / n_fn, 4) if n_fn else None,
            "note": (
                "These are ensemble false negatives — the classifier missed them, "
                "so the pipeline is expected to miss them too. Reported as a "
                "system limitation."
            ),
        },
    }
    return summary


def print_summary(summary: dict):
    """Pretty-print the evaluation summary."""
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE EVALUATION SUMMARY")
    logger.info("=" * 60)

    for group_key, group_label in [
        ("test_toxic",     "Test-set toxic"),
        ("error_fp",       "Error-analysis FP (false positives)"),
        ("combined_toxic", "All toxic (test + error FP)"),
    ]:
        g = summary[group_key]
        if g["n_samples"] == 0:
            logger.info(f"\n{group_label}: (no samples)")
            continue
        logger.info(f"\n{group_label} ({g['n_samples']} samples):")
        logger.info(f"  Correction success rate : {g['correction_success_rate']}")
        logger.info(f"  Fallback rate           : {g['fallback_rate']}")
        logger.info(f"  Avg iterations          : {g['avg_iterations']}")
        logger.info(f"  Iteration distribution  : {g['iteration_distribution']}")
        if "per_label_correction" in g:
            logger.info("  Per-label correction:")
            for label, stats in g["per_label_correction"].items():
                rate = stats["correction_rate"]
                rate_str = f"{rate:.2%}" if rate is not None else "N/A"
                logger.info(
                    f"    {label:15s}: "
                    f"{stats['corrected']}/{stats['initially_triggered']} = {rate_str}"
                )

    cg = summary["test_clean"]
    logger.info(f"\nTest-set clean ({cg['n_samples']} samples):")
    logger.info(f"  Preservation rate       : {cg['preservation_rate']}")
    logger.info(f"  False trigger rate      : {cg['false_trigger_rate']}")

    fn = summary["error_fn"]
    logger.info(f"\nError-analysis FN ({fn['n_samples']} samples):")
    logger.info(f"  Caught by pipeline      : {fn['caught_by_pipeline']}")
    logger.info(f"  Miss rate               : {fn['miss_rate']}")

    logger.info(f"\nTotal time: {summary.get('total_elapsed_seconds', '?')}s")
    logger.info(f"Errors: {summary['errors']}")


# ──────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="End-to-end pipeline evaluation")
    p.add_argument("--test_csv", type=str, default="data/test_split.csv",
                   help="Held-out test CSV (from split.py)")
    p.add_argument("--error_csv", type=str, default="outputs/ensemble/error_analysis.csv",
                   help="Error analysis CSV from ensemble evaluation (optional)")
    p.add_argument("--bert_ckpt", type=str, default="outputs/bert_final/best_checkpoint")
    p.add_argument("--bert_base", type=str, default="bert-base-uncased")
    p.add_argument("--llama_ckpt", type=str, default="outputs/llama_lora/best_checkpoint")
    p.add_argument("--llama_base", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--ensemble_dir", type=str, default="outputs/ensemble")
    p.add_argument("--n_toxic", type=int, default=50,
                   help="Number of toxic samples from test set")
    p.add_argument("--n_clean", type=int, default=50,
                   help="Number of clean samples from test set")
    p.add_argument("--max_iterations", type=int, default=5,
                   help="Max revision iterations per sample")
    p.add_argument("--output_dir", type=str, default="outputs/pipeline_eval")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Logging ──
    log_path = os.path.join(args.output_dir, "pipeline_eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # ── Env check ──
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Export it before running.")
        sys.exit(1)

    import torch
    device = os.environ.get("DETOXIGUARD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load ensemble (shared by sampling + pipeline) ──
    logger.info("Loading ensemble for scoring ...")
    init_ensemble(
        bert_ckpt=args.bert_ckpt,
        bert_base=args.bert_base,
        llama_ckpt=args.llama_ckpt,
        llama_base=args.llama_base,
        ensemble_dir=args.ensemble_dir,
        device=device,
    )

    # ── Build test set ──
    logger.info("Building test set ...")
    test_df = build_test_set(
        test_csv=args.test_csv,
        error_csv=args.error_csv,
        n_toxic=args.n_toxic,
        n_clean=args.n_clean,
    )
    logger.info(f"Test set: {len(test_df)} samples")

    # Save sampled test set for reproducibility
    test_csv_path = os.path.join(args.output_dir, "pipeline_eval_samples.csv")
    test_df.to_csv(test_csv_path, index=False)
    logger.info(f"Saved test samples to {test_csv_path}")

    # ── Run evaluation ──
    logger.info("Starting pipeline evaluation ...")
    t_start = time.time()
    results = evaluate_pipeline(test_df, max_iterations=args.max_iterations)
    t_total = time.time() - t_start
    logger.info(f"Evaluation complete in {t_total:.1f}s")

    # ── Save details ──
    details_path = os.path.join(args.output_dir, "pipeline_eval_details.json")
    with open(details_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved per-sample details to {details_path}")

    # ── Compute & save summary ──
    summary = compute_summary(results)
    summary["total_elapsed_seconds"] = round(t_total, 1)
    summary["config"] = {
        "n_toxic": args.n_toxic,
        "n_clean": args.n_clean,
        "max_iterations": args.max_iterations,
        "seed": SEED,
        "test_csv": args.test_csv,
        "error_csv": args.error_csv,
    }

    summary_path = os.path.join(args.output_dir, "pipeline_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    print_summary(summary)


if __name__ == "__main__":
    main()