"""
eval_pipeline_vis.py
---------------
Visualize pipeline evaluation results from eval_pipeline.py outputs.
Reads pipeline_eval_summary.json and pipeline_eval_details.json,
produces publication-quality figures for Milestone 3 / EMNLP SRW paper.

Runs locally (no GPU needed).

Usage:
    python evaluation/vis_pipeline.py \
        --summary  outputs/pipeline_eval/pipeline_eval_summary.json \
        --details  outputs/pipeline_eval/pipeline_eval_details.json \
        --output_dir outputs/pipeline_eval

Author: Ruide Yin
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LABEL_DISPLAY = ["Toxic", "Severe\nToxic", "Obscene", "Threat", "Insult", "Identity\nHate"]
DPI = 150

# Color palette — consistent with eval_ensemble style
C_PRIMARY   = "#2563EB"   
C_SECONDARY = "#10B981"   
C_ACCENT    = "#F59E0B"   
C_DANGER    = "#EF4444"   
C_PASS      = "#60A5FA"   
C_GRAY      = "#6B7280"
C_LIGHT_BG  = "#F8FAFC"


def load_data(summary_path, details_path):
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    with open(details_path, encoding="utf-8") as f:
        details = json.load(f)
    return summary, details


# ──────────────────────────────────────────────
# 1. Per-label correction rate (horizontal bar)
# ──────────────────────────────────────────────

def plot_per_label_correction(summary, save_path):
    """Horizontal bar chart: per-label correction rate for combined toxic group."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    combined = summary["combined_toxic"]["per_label_correction"]

    rates = []
    triggered = []
    for label in LABELS:
        stats = combined[label]
        rates.append(stats["correction_rate"] * 100 if stats["correction_rate"] is not None else 0)
        triggered.append(stats["initially_triggered"])

    y_pos = np.arange(len(LABELS))
    bars = ax.barh(y_pos, rates, height=0.6, color=C_PRIMARY, edgecolor="white", linewidth=0.5)

    # Add rate + count annotations
    for i, (bar, rate, n) in enumerate(zip(bars, rates, triggered)):
        ax.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", va="center", ha="right", fontsize=11,
                fontweight="bold", color="white")
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"({n} samples)", va="center", ha="left", fontsize=9, color=C_GRAY)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(LABEL_DISPLAY, fontsize=11)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Correction Rate (%)", fontsize=12)
    ax.set_title("Per-Label Correction Rate — Pipeline (All Toxic Samples)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 2. Iteration distribution (bar chart)
# ──────────────────────────────────────────────

def plot_iteration_distribution(summary, save_path):
    """Bar chart: how many samples needed 1, 2, 3, ... iterations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    combined = summary["combined_toxic"]
    iter_dist = combined["iteration_distribution"]

    # Convert string keys to int (JSON keys are strings)
    iters = sorted(int(k) for k in iter_dist.keys())
    counts = [iter_dist[str(i)] for i in iters]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    bars = ax.bar(iters, counts, width=0.7, color=C_PRIMARY, edgecolor="white", linewidth=0.5)

    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Number of Revision Iterations", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Iteration Distribution — Pipeline Revision Loop", fontsize=13, fontweight="bold")
    ax.set_xticks(iters)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add avg line
    avg = combined["avg_iterations"]
    ax.axvline(avg, color=C_DANGER, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(avg + 0.15, ax.get_ylim()[1] * 0.9, f"avg = {avg:.2f}",
            fontsize=10, color=C_DANGER, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 3. Group-level summary (grouped bar)
# ──────────────────────────────────────────────

def plot_group_summary(summary, save_path):
    """Grouped bar chart comparing correction rate, preservation rate, etc."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")

    groups = []
    rates = []
    colors = []
    labels_text = []

    # Test toxic
    tt = summary["test_toxic"]
    groups.append("Test Toxic")
    rates.append(tt["correction_success_rate"] * 100)
    colors.append(C_PRIMARY)
    labels_text.append(f'{tt["correction_success_rate"]*100:.1f}%\n(n={tt["n_samples"]})')

    # Error FP
    efp = summary["error_fp"]
    groups.append("Error-Analysis\nFP")
    rates.append(efp["correction_success_rate"] * 100)
    colors.append(C_SECONDARY)
    labels_text.append(f'{efp["correction_success_rate"]*100:.1f}%\n(n={efp["n_samples"]})')

    # Combined toxic
    ct = summary["combined_toxic"]
    groups.append("All Toxic\n(Combined)")
    rates.append(ct["correction_success_rate"] * 100)
    colors.append(C_ACCENT)
    labels_text.append(f'{ct["correction_success_rate"]*100:.1f}%\n(n={ct["n_samples"]})')

    # Clean preservation
    tc = summary["test_clean"]
    groups.append("Clean\nPreservation")
    rates.append(tc["preservation_rate"] * 100)
    colors.append("#8B5CF6")  # purple
    labels_text.append(f'{tc["preservation_rate"]*100:.1f}%\n(n={tc["n_samples"]})')

    # FN catch rate
    fn = summary["error_fn"]
    catch_rate = (1.0 - fn["miss_rate"]) * 100 if fn["miss_rate"] is not None else 0
    groups.append("FN Catch\nRate")
    rates.append(catch_rate)
    colors.append(C_DANGER)
    labels_text.append(f'{catch_rate:.1f}%\n(n={fn["n_samples"]})')

    x_pos = np.arange(len(groups))
    bars = ax.bar(x_pos, rates, width=0.6, color=colors, edgecolor="white", linewidth=0.5)

    for bar, label in zip(bars, labels_text):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Pipeline Performance by Test Group", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(100, color=C_GRAY, linestyle=":", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 4. Outcome breakdown (stacked bar per group)
# ──────────────────────────────────────────────

def plot_outcome_breakdown(details, save_path):
    """Stacked horizontal bar: corrected / fallback / passthrough / error per source group."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("white")

    source_order = ["test_toxic", "error_fp", "test_clean", "error_fn"]
    source_labels = ["Test Toxic", "Error FP", "Test Clean", "Error FN"]

    outcome_colors = {
        "corrected":   C_SECONDARY,
        "clean_pass":  C_PASS,
        "passthrough": C_PRIMARY,
        "fallback":    C_DANGER,
        "error":       C_GRAY,
    }
    outcome_order = ["corrected", "clean_pass", "passthrough", "fallback", "error"]

    # Count outcomes per source
    data = {s: {o: 0 for o in outcome_order} for s in source_order}
    for r in details:
        src = r["source"]
        out = r["outcome"]
        if src in data and out in data[src]:
            data[src][out] += 1

    y_pos = np.arange(len(source_order))
    left = np.zeros(len(source_order))

    for outcome in outcome_order:
        counts = [data[s][outcome] for s in source_order]
        if sum(counts) == 0:
            continue
        ax.barh(y_pos, counts, left=left, height=0.6,
                color=outcome_colors[outcome], edgecolor="white", linewidth=0.5,
                label=outcome.capitalize())
        # Label non-zero segments
        for i, c in enumerate(counts):
            if c > 0:
                ax.text(left[i] + c / 2, y_pos[i], str(c),
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white" if c > 3 else "black")
        left += counts

    ax.set_yticks(y_pos)
    ax.set_yticklabels(source_labels, fontsize=11)
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_title("Outcome Breakdown by Test Group", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 5. Iterations vs. outcome (box/strip plot)
# ──────────────────────────────────────────────

def plot_iterations_by_source(details, save_path):
    """Strip plot: iteration count per sample, colored by source group."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    source_order = ["test_toxic", "error_fp", "error_fn"]
    source_labels = ["Test Toxic", "Error FP", "Error FN"]
    source_colors = {
        "test_toxic": C_PRIMARY,
        "error_fp":   C_SECONDARY,
        "error_fn":   C_ACCENT,
    }

    for idx, src in enumerate(source_order):
        samples = [r for r in details if r["source"] == src and r.get("outcome") != "error"]
        if not samples:
            continue
        iters = [r["num_iterations"] for r in samples]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(iters))
        ax.scatter([idx + j for j in jitter], iters,
                   c=source_colors[src], alpha=0.6, s=30, edgecolors="white", linewidth=0.3)

        # Box stats
        median = np.median(iters)
        ax.plot([idx - 0.25, idx + 0.25], [median, median],
                color="black", linewidth=2, zorder=5)

    ax.set_xticks(range(len(source_order)))
    ax.set_xticklabels(source_labels, fontsize=11)
    ax.set_ylabel("Revision Iterations", fontsize=12)
    ax.set_title("Revision Iterations per Sample by Group", fontsize=13, fontweight="bold")
    ax.set_yticks(range(0, 7))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize pipeline evaluation results")
    p.add_argument("--summary", type=str, default="outputs/pipeline_eval/pipeline_eval_summary.json")
    p.add_argument("--details", type=str, default="outputs/pipeline_eval/pipeline_eval_details.json")
    p.add_argument("--output_dir", type=str, default="outputs/pipeline_eval")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading pipeline evaluation data ...")
    summary, details = load_data(args.summary, args.details)

    print("Generating plots ...")
    plot_per_label_correction(summary, os.path.join(args.output_dir, "per_label_correction.png"))
    plot_iteration_distribution(summary, os.path.join(args.output_dir, "iteration_distribution.png"))
    plot_group_summary(summary, os.path.join(args.output_dir, "group_summary.png"))
    plot_outcome_breakdown(details, os.path.join(args.output_dir, "outcome_breakdown.png"))
    plot_iterations_by_source(details, os.path.join(args.output_dir, "iterations_by_source.png"))

    print(f"\nAll plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()