"""
train_bert.py
-------------
Fine-tune BERT for multi-label toxic comment classification.

This file is intentionally interface-aligned with train_llama.py so that a
future ensemble.py can call both models with the same contract.

Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate

Dependencies:
    pip install transformers accelerate scikit-learn pandas torch iterstrat

Usage:
    python train_bert.py --model_name bert-base-uncased \
                         --train_csv data/train.csv \
                         --output_dir outputs/bert \
                         --epochs 5 --batch_size 32
"""

import argparse
import functools
import json
import logging
import math
import os
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


# ──────────────────────────────────────────────
# Config & Logging
# ──────────────────────────────────────────────

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
NUM_LABELS = len(LABELS)

# Reproducibility seed derived from team member university IDs:
# 18903824 (Yin) + 15638934 (Wang) + 14915587 (Zhao)
TEAM_SEED = 49458345

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class ToxicCommentDataset(Dataset):
    """Tokenizes comments and returns input tensors + multi-label targets."""

    def __init__(self, texts, labels, tokenizer, max_length: int = 512):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding=False,              # defer padding to collate_fn
            truncation=True,
            return_tensors="pt",
        )
        label_vec = torch.tensor(
            self.labels.iloc[idx].values.astype(float), dtype=torch.float32
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_vec,
        }


def dynamic_collate_fn(batch, tokenizer):
    """
    Pad each batch to the longest sequence in that batch instead of a fixed
    max_length. This keeps the data pipeline aligned with train_llama.py.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for dynamic padding.")

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Right-padding is the standard choice for BERT-style encoders.
        input_ids_list.append(
            torch.cat([
                item["input_ids"],
                torch.full((pad_len,), pad_token_id, dtype=torch.long),
            ])
        )
        attention_mask_list.append(
            torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ])
        )
        labels_list.append(item["labels"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


# ──────────────────────────────────────────────
# Class-imbalance utilities
# ──────────────────────────────────────────────


def compute_pos_weight(label_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss to up-weight minority classes.

    pos_weight_i = num_negatives_i / num_positives_i
    """
    counts = label_df.sum(axis=0).values.astype(float)
    n = len(label_df)
    pos_weight = (n - counts) / np.clip(counts, 1, None)
    logger.info("pos_weight per label:")
    for label, w in zip(LABELS, pos_weight):
        logger.info(f"  {label:15s}: {w:.2f}")
    return torch.tensor(pos_weight, dtype=torch.float32)


# ──────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────


def build_bert_model(model_name: str):
    """
    Load a BERT-style encoder model with a multi-label classification head.

    This intentionally mirrors the role of build_lora_model() in train_llama.py,
    except here we fine-tune the full encoder rather than attaching LoRA.
    """
    logger.info(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    if tokenizer.pad_token is None:
        # This is unlikely for BERT, but keeps the code robust if the user swaps
        # to another encoder checkpoint later.
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────


def find_optimal_thresholds(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    search_range: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Find the threshold that maximises F1 for each label independently.
    """
    if search_range is None:
        search_range = np.arange(0.05, 0.96, 0.01)

    best_thresholds: dict[str, float] = {}
    for i, label in enumerate(LABELS):
        best_f1, best_t = 0.0, 0.5
        for t in search_range:
            preds = (all_probs[:, i] >= t).astype(int)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        best_thresholds[label] = best_t
        logger.info(f"  optimal threshold for {label:15s}: {best_t:.2f}  (F1={best_f1:.4f})")

    return best_thresholds



def compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    thresholds: dict[str, float] | None = None,
):
    """
    Per-label and macro F1, ROC-AUC, PR-AUC, precision, and recall.
    """
    default_t = 0.5
    all_preds = np.zeros_like(all_probs, dtype=int)
    for i, label in enumerate(LABELS):
        t = thresholds[label] if thresholds else default_t
        all_preds[:, i] = (all_probs[:, i] >= t).astype(int)

    per_label_f1 = {}
    per_label_auc = {}
    per_label_pr_auc = {}
    per_label_precision = {}
    per_label_recall = {}

    for i, label in enumerate(LABELS):
        per_label_f1[label] = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        per_label_precision[label] = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        per_label_recall[label] = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        if all_labels[:, i].sum() > 0:
            per_label_auc[label] = roc_auc_score(all_labels[:, i], all_probs[:, i])
            per_label_pr_auc[label] = average_precision_score(all_labels[:, i], all_probs[:, i])
        else:
            per_label_auc[label] = float("nan")
            per_label_pr_auc[label] = float("nan")

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_auc = np.nanmean(list(per_label_auc.values()))
    macro_pr_auc = np.nanmean(list(per_label_pr_auc.values()))

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
# Train / Eval loops
# ──────────────────────────────────────────────


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    loss_fn,
    grad_accum_steps=1,
):
    model.train()
    total_loss = 0.0

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        window_start = step - step % grad_accum_steps
        actual_accum = min(grad_accum_steps, len(loader) - window_start)

        amp_ctx = autocast(device_type="cuda", enabled=(device.type == "cuda"))
        with amp_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = loss_fn(outputs.logits, labels) / actual_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * actual_accum

        if (step + 1) % 100 == 0:
            logger.info(f"  step {step+1}/{len(loader)} | loss {loss.item() * actual_accum:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, thresholds: dict[str, float] | None = None):
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        amp_ctx = autocast(device_type="cuda", enabled=(device.type == "cuda"))
        with amp_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = loss_fn(outputs.logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(outputs.logits).cpu().float().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_probs, thresholds)

    return avg_loss, metrics, all_probs, all_labels


# ──────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────

class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter = 0

    def step(self, score: float) -> bool:
        """Return True if training should stop."""
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered (no improvement for {self.patience} epochs)")
            return True
        logger.info(f"  EarlyStopping: {self.counter}/{self.patience} epochs without improvement")
        return False


# ──────────────────────────────────────────────
# Inference helper (used by ensemble.py)
# ──────────────────────────────────────────────


def load_trained_model(checkpoint_dir: str, base_model_name: str, device: str = "cuda"):
    """
    Reload the fine-tuned BERT model for inference.
    Called by ensemble.py to get BERT predictions.

    Signature is intentionally the same as train_llama.py:
        load_trained_model(checkpoint_dir, base_model_name, device="cuda")

    For BERT we reload directly from checkpoint_dir. base_model_name is kept in
    the signature so both models can be called through one shared interface.
    """
    tokenizer_source = checkpoint_dir if os.path.exists(checkpoint_dir) else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=os.path.exists(tokenizer_source),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        local_files_only=True,
    )

    resolved_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(resolved_device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """
    Return sigmoid probabilities of shape (N, 6) for a list of raw text strings.
    Interface contract for ensemble.py:
        probs = predict(model, tokenizer, texts)   # float32 array, values in [0,1]
    """
    device = next(model.parameters()).device
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        amp_ctx = autocast(device_type="cuda", enabled=(device.type == "cuda"))
        with amp_ctx:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs = torch.sigmoid(logits).cpu().float().numpy()
        all_probs.append(probs)

    if not all_probs:
        return np.empty((0, NUM_LABELS), dtype=np.float32)
    return np.concatenate(all_probs, axis=0).astype(np.float32)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for toxic comment classification")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/bert")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Hard cap on token length; actual padding is dynamic per batch")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=TEAM_SEED)
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = main process only, safest on HPC)")
    return parser.parse_args()



def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    num_gpus = torch.cuda.device_count()
    logger.info(f"GPUs available: {num_gpus}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load & split data (multi-label stratified) ──
    logger.info("Loading dataset ...")
    df = pd.read_csv(args.train_csv)

    required_columns = {"comment_text", *LABELS}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {args.train_csv}: {sorted(missing_columns)}")

    df["comment_text"] = df["comment_text"].fillna("")

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=args.val_split, random_state=args.seed
    )
    train_idx, val_idx = next(msss.split(df, df[LABELS]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    logger.info(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    # ── 2. Build model & tokenizer ───────────
    model, tokenizer = build_bert_model(args.model_name)
    model = model.to(device)

    # ── 3. Compute class-imbalance weights ───
    pos_weight = compute_pos_weight(train_df[LABELS]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── 4. Build datasets & loaders ──────────
    train_dataset = ToxicCommentDataset(
        train_df["comment_text"], train_df[LABELS], tokenizer, args.max_length
    )
    val_dataset = ToxicCommentDataset(
        val_df["comment_text"], val_df[LABELS], tokenizer, args.max_length
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=functools.partial(dynamic_collate_fn, tokenizer=tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=functools.partial(dynamic_collate_fn, tokenizer=tokenizer),
    )

    # ── 5. Optimizer & scheduler ─────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── 6. Training loop with early stopping ─
    best_macro_f1 = -1.0
    best_epoch = 0
    early_stopper = EarlyStopping(patience=args.patience)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"═══ Epoch {epoch}/{args.epochs} ═══")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            loss_fn,
            grad_accum_steps=args.grad_accum_steps,
        )
        val_loss, metrics, _, _ = evaluate(model, val_loader, device, loss_fn)

        logger.info(
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | Macro-AUC: {metrics['macro_auc']:.4f} | "
            f"Macro-PR-AUC: {metrics['macro_pr_auc']:.4f}"
        )
        for label in LABELS:
            logger.info(
                f"  {label:15s}  F1={metrics['per_label_f1'][label]:.4f}  "
                f"P={metrics['per_label_precision'][label]:.4f}  "
                f"R={metrics['per_label_recall'][label]:.4f}  "
                f"AUC={metrics['per_label_auc'][label]:.4f}  "
                f"PR-AUC={metrics['per_label_pr_auc'][label]:.4f}"
            )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")
            model.save_pretrained(best_ckpt_dir)
            tokenizer.save_pretrained(best_ckpt_dir)
            with open(os.path.join(best_ckpt_dir, "label_order.json"), "w") as f:
                json.dump({"labels": LABELS}, f, indent=2)
            logger.info(f"  ✓ New best model saved (macro-F1={best_macro_f1:.4f})")

        if early_stopper.step(metrics["macro_f1"]):
            break

    logger.info(f"Training complete. Best macro-F1={best_macro_f1:.4f} at epoch {best_epoch}.")

    # ── 7. Find optimal per-label thresholds on val set ──
    best_ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")
    logger.info(f"Reloading best checkpoint from {best_ckpt_dir} for threshold search ...")
    model, _ = load_trained_model(best_ckpt_dir, args.model_name, device=str(device))

    logger.info("═══ Searching for optimal per-label thresholds on validation set ═══")
    _, _, val_probs, val_labels = evaluate(model, val_loader, device, loss_fn)
    optimal_thresholds = find_optimal_thresholds(val_labels, val_probs)

    logger.info("═══ Val metrics with optimal thresholds ═══")
    tuned_metrics = compute_metrics(val_labels, val_probs, optimal_thresholds)
    logger.info(
        f"Macro-F1: {tuned_metrics['macro_f1']:.4f} | "
        f"Micro-F1: {tuned_metrics['micro_f1']:.4f} | "
        f"Macro-AUC: {tuned_metrics['macro_auc']:.4f} | "
        f"Macro-PR-AUC: {tuned_metrics['macro_pr_auc']:.4f}"
    )
    for label in LABELS:
        logger.info(
            f"  {label:15s}  t={optimal_thresholds[label]:.2f}  "
            f"F1={tuned_metrics['per_label_f1'][label]:.4f}  "
            f"P={tuned_metrics['per_label_precision'][label]:.4f}  "
            f"R={tuned_metrics['per_label_recall'][label]:.4f}"
        )

    thresh_path = os.path.join(best_ckpt_dir, "optimal_thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(optimal_thresholds, f, indent=2)
    logger.info(f"Optimal thresholds saved to {thresh_path}")


if __name__ == "__main__":
    main()
