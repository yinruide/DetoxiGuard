"""
split_data.py
-------------
Data splitting script for the DetoxiGuard project.
Produces a single, reproducible three-way split that all team members
(BERT, LLaMA, ensemble, pipeline evaluation) share.

Output:
    data/train_split.csv   – training rows      (80 %)
    data/val_split.csv     – validation rows     (10 %)  → ensemble grid search + early stopping
    data/test_split.csv    – held-out test rows  (10 %)  → agent pipeline evaluation

All files keep the original columns:
    id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

Usage:
    python split_data.py --raw_csv data/train.csv --output_dir data
"""

import argparse
import os

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Reproducibility seed derived from team member university IDs:
# 18903824 (Yin) + 15638934 (Wang) + 14915587 (Zhao)
TEAM_SEED = 49458345


def main():
    parser = argparse.ArgumentParser(description="Split Jigsaw data into train/val/test")
    parser.add_argument("--raw_csv",    type=str, default="data/train.csv",
                        help="Path to the original Kaggle train.csv")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory to write train/val/test CSVs")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────
    df = pd.read_csv(args.raw_csv)
    print(f"Loaded {len(df):,} rows from {args.raw_csv}")

    # ── First split: 80 % train, 20 % remaining ──
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=TEAM_SEED
    )
    train_idx, remaining_idx = next(msss1.split(df, df[LABELS]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    remaining_df = df.iloc[remaining_idx].reset_index(drop=True)

    # ── Second split: remaining 50/50 → val 10 %, test 10 % ──
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=TEAM_SEED
    )
    val_idx, test_idx = next(msss2.split(remaining_df, remaining_df[LABELS]))

    val_df  = remaining_df.iloc[val_idx].reset_index(drop=True)
    test_df = remaining_df.iloc[test_idx].reset_index(drop=True)

    # ── Save ──────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train_split.csv")
    val_path   = os.path.join(args.output_dir, "val_split.csv")
    test_path  = os.path.join(args.output_dir, "test_split.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # ── Summary ───────────────────────────────────
    print(f"\nTrain: {len(train_df):>7,d}  →  {train_path}")
    print(f"Val:   {len(val_df):>7,d}  →  {val_path}")
    print(f"Test:  {len(test_df):>7,d}  →  {test_path}")

    print(f"\nPer-label positive counts:")
    print(f"  {'Label':<15s} {'Train':>7s} {'Val':>7s} {'Test':>7s} {'Val%':>6s} {'Test%':>6s}")
    for label in LABELS:
        tr = int(train_df[label].sum())
        va = int(val_df[label].sum())
        te = int(test_df[label].sum())
        total = tr + va + te
        print(
            f"  {label:<15s} {tr:>7,d} {va:>7,d} {te:>7,d} "
            f"{va / total * 100:>5.1f}% {te / total * 100:>5.1f}%"
        )


if __name__ == "__main__":
    main()