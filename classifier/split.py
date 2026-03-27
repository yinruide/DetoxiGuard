"""
split_data.py
-------------
Data splitting script for the DetoxiGuard project.
Produces a single, reproducible train/val split that all team members
(BERT, LLaMA, ensemble) share.

Output:
    data/train_split.csv   – training rows  (90 %)
    data/val_split.csv     – validation rows (10 %)

Both files keep the original columns:
    id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

Usage:
    python split_data.py --raw_csv data/train.csv --output_dir data --val_ratio 0.1
"""

import argparse
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Reproducibility seed derived from team member university IDs:
# 18903824 (Yin) + 15638934 (Wang) + 14915587 (Zhao)
TEAM_SEED = 49458345


def main():
    parser = argparse.ArgumentParser(description="Split Jigsaw data into train/val")
    parser.add_argument("--raw_csv",    type=str, default="data/train.csv",
                        help="Path to the original Kaggle train.csv")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory to write train_split.csv and val_split.csv")
    parser.add_argument("--val_ratio",  type=float, default=0.1,
                        help="Fraction of data reserved for validation")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────
    df = pd.read_csv(args.raw_csv)
    print(f"Loaded {len(df):,} rows from {args.raw_csv}")

    # ── Multi-label stratified split ──────────────
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=args.val_ratio, random_state=TEAM_SEED
    )
    train_idx, val_idx = next(msss.split(df, df[LABELS]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    # ── Save ──────────────────────────────────────
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train_split.csv")
    val_path   = os.path.join(args.output_dir, "val_split.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # ── Summary ───────────────────────────────────
    print(f"\nTrain: {len(train_df):,}  →  {train_path}")
    print(f"Val:   {len(val_df):,}  →  {val_path}")
    print(f"\nPer-label positive counts:")
    print(f"  {'Label':<15s} {'Train':>7s} {'Val':>7s} {'Val %':>7s}")
    for label in LABELS:
        tr_pos = int(train_df[label].sum())
        va_pos = int(val_df[label].sum())
        ratio  = va_pos / (tr_pos + va_pos) * 100
        print(f"  {label:<15s} {tr_pos:>7,d} {va_pos:>7,d} {ratio:>6.1f}%")


if __name__ == "__main__":
    main()