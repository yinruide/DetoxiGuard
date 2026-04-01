"""
train_llama.py
--------------
Fine-tune LLaMA with LoRA for multi-label toxic comment classification.

Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate

Dependencies:
    pip install transformers peft accelerate bitsandbytes datasets scikit-learn pandas torch iterstrat

Prerequisite:
    python split_data.py          # run once to generate train_split.csv & val_split.csv

Usage:
    python train_llama.py --model_name meta-llama/Llama-3.2-1B
                          --train_csv data/train_split.csv
                          --val_csv   data/val_split.csv
                          --output_dir outputs/llama_lora
                          --epochs 5 --batch_size 64
"""

import argparse
import functools
import json
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
import logging

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
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         label_vec,
        }


def dynamic_collate_fn(batch, tokenizer):
    """
    Pad each batch to the longest sequence in that batch instead of a
    fixed max_length.  This saves memory and compute on short-comment
    batches while still respecting the per-sample truncation cap set in
    __getitem__.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), item["input_ids"]])
        )
        attention_mask_list.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), item["attention_mask"]])
        )
        labels_list.append(item["labels"])

    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels":         torch.stack(labels_list),
    }


# ──────────────────────────────────────────────
# Class-imbalance utilities
# ──────────────────────────────────────────────

def compute_pos_weight(label_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss to up-weight minority classes.

    pos_weight_i = num_negatives_i / num_positives_i

    This makes the loss treat each positive example as if there were
    pos_weight many of them, directly counteracting class imbalance.
    """
    counts = label_df.sum(axis=0).values.astype(float)          # (6,)
    n = len(label_df)
    pos_weight = (n - counts) / np.clip(counts, 1, None)        # avoid div-by-zero
    logger.info("pos_weight per label:")
    for label, w in zip(LABELS, pos_weight):
        logger.info(f"  {label:15s}: {w:.2f}")
    return torch.tensor(pos_weight, dtype=torch.float32)


# ──────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────

def build_lora_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    """
    Load a LLaMA base model and attach a LoRA adapter + classification head.

    LoRA is applied to all four attention projection matrices (Q, K, V, O)
    for maximum expressiveness while keeping trainable parameter count low.
    """
    logger.info(f"Loading base model: {model_name}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def find_optimal_thresholds(all_labels: np.ndarray, all_probs: np.ndarray,
                            search_range: np.ndarray | None = None) -> dict[str, float]:
    """
    Find the threshold that maximises F1 for each label independently.

    Sweeps candidate thresholds over the validation predictions and picks
    the one that yields the highest per-label F1.  This is far more
    appropriate than a fixed 0.5 cutoff when classes are heavily imbalanced,
    because rare-class probability distributions tend to be shifted well
    below 0.5.

    Returns:
        dict mapping label name -> best threshold (float)
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


def compute_metrics(all_labels: np.ndarray, all_probs: np.ndarray,
                    thresholds: dict[str, float] | None = None):
    """
    Per-label and macro F1, ROC-AUC, PR-AUC, precision, and recall.

    If *thresholds* is provided, each label uses its own optimised cutoff;
    otherwise a default of 0.5 is used for every label.
    """
    default_t = 0.5
    all_preds = np.zeros_like(all_probs, dtype=int)
    for i, label in enumerate(LABELS):
        t = thresholds[label] if thresholds else default_t
        all_preds[:, i] = (all_probs[:, i] >= t).astype(int)

    per_label_f1  = {}
    per_label_auc = {}
    per_label_pr_auc    = {}
    per_label_precision = {}
    per_label_recall    = {}
    for i, label in enumerate(LABELS):
        per_label_f1[label]  = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        per_label_precision[label] = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        per_label_recall[label]    = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        if all_labels[:, i].sum() > 0:
            per_label_auc[label]    = roc_auc_score(all_labels[:, i], all_probs[:, i])
            per_label_pr_auc[label] = average_precision_score(all_labels[:, i], all_probs[:, i])
        else:
            per_label_auc[label]    = float("nan")
            per_label_pr_auc[label] = float("nan")

    macro_f1  = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    micro_f1  = f1_score(all_labels, all_preds, average="micro",  zero_division=0)
    macro_auc    = np.nanmean(list(per_label_auc.values()))
    macro_pr_auc = np.nanmean(list(per_label_pr_auc.values()))

    return {
        "macro_f1":          macro_f1,
        "micro_f1":          micro_f1,
        "macro_auc":         macro_auc,
        "macro_pr_auc":      macro_pr_auc,
        "per_label_f1":      per_label_f1,
        "per_label_auc":     per_label_auc,
        "per_label_pr_auc":  per_label_pr_auc,
        "per_label_precision": per_label_precision,
        "per_label_recall":    per_label_recall,
    }


# ──────────────────────────────────────────────
# Train / Eval loops
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn,
                    grad_accum_steps=1, rank=0):
    model.train()
    total_loss = 0.0

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Number of micro-steps in the current accumulation window
        # (may be < grad_accum_steps for the last window of the epoch)
        window_start = step - step % grad_accum_steps
        actual_accum = min(grad_accum_steps, len(loader) - window_start)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = loss_fn(outputs.logits, labels) / actual_accum

        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * actual_accum

        if (step + 1) % 100 == 0 and rank == 0:
            logger.info(f"  step {step+1}/{len(loader)} | loss {loss.item() * actual_accum:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, thresholds: dict[str, float] | None = None):
    model.eval()
    all_labels = []
    all_probs  = []
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = loss_fn(outputs.logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(outputs.logits).cpu().float().numpy()   # (B, 6)
        all_probs.append(probs)
        all_labels.append(labels.cpu().float().numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss   = total_loss / len(loader)
    metrics    = compute_metrics(all_labels, all_probs, thresholds)

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
    Reload the fine-tuned LoRA model for inference.
    Called by ensemble.py to get LLaMA predictions.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        dtype=torch.bfloat16,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], batch_size: int = 32,
            max_length: int = 512) -> np.ndarray:
    """
    Return sigmoid probabilities of shape (N, 6) for a list of raw text strings.
    Interface contract for ensemble.py:
        probs = predict(model, tokenizer, texts)   # float32 array, values in [0,1]
    """
    device = next(model.parameters()).device
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.sigmoid(logits).cpu().float().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA+LoRA for toxic comment classification")
    parser.add_argument("--model_name",  type=str, default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--train_csv",   type=str, default="data/train_split.csv",
                        help="Pre-split training CSV (from split_data.py)")
    parser.add_argument("--val_csv",     type=str, default="data/val_split.csv",
                        help="Pre-split validation CSV (from split_data.py)")
    parser.add_argument("--output_dir",  type=str, default="outputs/llama")
    parser.add_argument("--max_length",  type=int, default=512,
                        help="Hard cap on token length; actual padding is dynamic per batch")
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=32,
                        help="Per-GPU batch size (effective = batch_size * num_gpus)")
    parser.add_argument("--grad_accum_steps", type=int, default=None,
                        help="Gradient accumulation steps (overrides auto-compute from target_effective_batch)")
    parser.add_argument("--target_effective_batch", type=int, default=64,
                        help="Target effective batch size; grad_accum_steps auto-computed if not explicitly set")
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--lora_r",      type=int, default=8)
    parser.add_argument("--lora_alpha",  type=int, default=16)
    parser.add_argument("--lora_dropout",type=float, default=0.1)
    parser.add_argument("--seed",        type=int, default=TEAM_SEED)
    parser.add_argument("--patience",    type=int, default=2,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers (0 = main process only, safest on HPC)")
    return parser.parse_args()


def _worker(rank, world_size, args):
    """DDP worker function. Each process runs this with its own rank."""
    is_distributed = world_size > 1

    if is_distributed:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if rank == 0:
        logger.info(f"Device: {device}")
        logger.info(f"GPUs available: {world_size}")
        os.makedirs(args.output_dir, exist_ok=True)

    # Compute grad_accum_steps
    if args.grad_accum_steps is not None:
        grad_accum_steps = args.grad_accum_steps
    else:
        grad_accum_steps = max(1, args.target_effective_batch // (args.batch_size * world_size))

    if rank == 0:
        logger.info(
            f"Effective batch size: {args.batch_size} * {world_size} GPUs * {grad_accum_steps} accum = "
            f"{args.batch_size * world_size * grad_accum_steps}"
        )

    # ── 1. Load pre-split data ─────────────────
    if rank == 0:
        logger.info("Loading pre-split datasets ...")
    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)
    if rank == 0:
        logger.info(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    # ── 2. Build model & tokenizer ───────────
    model, tokenizer = build_lora_model(
        args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

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

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=functools.partial(dynamic_collate_fn, tokenizer=tokenizer),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=functools.partial(dynamic_collate_fn, tokenizer=tokenizer),
    )

    # ── 5. Optimizer & scheduler ─────────────
    # AdamW already provides momentum via beta1 (default 0.9) and adaptive
    # learning rates via beta2 (default 0.999).  This subsumes classical
    # SGD+momentum and is the standard choice for LLM fine-tuning.
    raw_model = model.module if is_distributed else model
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, raw_model.parameters()),
        lr=args.lr, weight_decay=0.01,
        betas=(0.9, 0.999),             # beta1 = momentum coefficient
    )
    total_steps   = math.ceil(len(train_loader) / grad_accum_steps) * args.epochs
    warmup_steps  = int(0.10 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── 6. Training loop with early stopping ─
    best_macro_f1 = 0.0
    best_epoch    = 0
    early_stopper = EarlyStopping(patience=args.patience)

    for epoch in range(1, args.epochs + 1):
        if rank == 0:
            logger.info(f"═══ Epoch {epoch}/{args.epochs} ═══")

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn,
            grad_accum_steps=grad_accum_steps, rank=rank,
        )

        # Only rank 0 runs evaluation, saves checkpoints, and checks early stopping
        if rank == 0:
            eval_model = raw_model
            val_loss, metrics, _, _ = evaluate(eval_model, val_loader, device, loss_fn)

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
                best_epoch    = epoch
                raw_model.save_pretrained(os.path.join(args.output_dir, "best_checkpoint"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best_checkpoint"))
                logger.info(f"  ✓ New best model saved (macro-F1={best_macro_f1:.4f})")

            should_stop = early_stopper.step(metrics["macro_f1"])
        else:
            should_stop = False

        # Broadcast early stopping decision to all ranks
        if is_distributed:
            dist.barrier()
        if is_distributed:
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

        if should_stop:
            break

    if rank == 0:
        logger.info(f"Training complete. Best macro-F1={best_macro_f1:.4f} at epoch {best_epoch}.")

        # ── 7. Find optimal per-label thresholds on val set ──
        best_ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")
        logger.info(f"Reloading best checkpoint from {best_ckpt_dir} for threshold search ...")
        base_for_thresh = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=NUM_LABELS,
            problem_type="multi_label_classification", dtype=torch.bfloat16,
        )
        base_for_thresh.config.pad_token_id = tokenizer.eos_token_id
        model_for_thresh = PeftModel.from_pretrained(base_for_thresh, best_ckpt_dir)
        model_for_thresh = model_for_thresh.to(device)
        model_for_thresh.eval()

        logger.info("═══ Searching for optimal per-label thresholds on validation set ═══")
        _, _, val_probs, val_labels = evaluate(model_for_thresh, val_loader, device, loss_fn)
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

        thresh_path = os.path.join(args.output_dir, "best_checkpoint", "optimal_thresholds.json")
        with open(thresh_path, "w") as f:
            json.dump(optimal_thresholds, f, indent=2)
        logger.info(f"Optimal thresholds saved to {thresh_path}")

    if is_distributed:
        dist.destroy_process_group()


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        torch.multiprocessing.spawn(
            _worker,
            args=(num_gpus, args),
            nprocs=num_gpus,
            join=True,
        )
    else:
        _worker(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()