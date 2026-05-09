# DetoxiGuard

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![NYU DS-301](https://img.shields.io/badge/NYU-DS--301%20Spring%202026-purple)

**Multi-Label Toxic Comment Classification & Agentic LLM Guardrail System**

DetoxiGuard is an end-to-end content moderation pipeline that combines fine-tuned BERT and LLaMA classifiers in a weighted ensemble, then wraps them in a LangGraph-based detect-and-correct agent loop. The system flags toxic user input across six categories and automatically rewrites flagged content using GPT-4o, iterating until the output passes the classifier or a maximum iteration count is reached.

Course project for Advanced Topics in Data Science: Deep Learning and Intro to LLM @ NYU, Spring 2026.    

**Authors**: Ruide Yin, Yanfu Wang, Langyue Zhao     

---

### Highlights

- **AUC-ROC 0.989 · Macro-Recall 0.884** — learned ensemble with recall-prioritized thresholds (6 labels, n = 22,355)
- **99.4% correction rate** on flagged toxic inputs; 87.9% resolved in a single GPT-4o iteration
- Zero false triggers on clean inputs; 0.6% fallback rate
- Test-set performance consistent with validation (Macro-F1 0.671 vs 0.672, Macro-AUC 0.990 vs 0.985)

---

## Architecture

```
User Input
    |
    v
+-------------------------+
|   Ensemble Classifier   |
|  BERT + LLaMA-3.2-1B   |
|  (per-label weighted)   |
+------------+------------+
             |
        +----v----+  Toxic   +--------------+
        | Toxic?  |--------->| GPT-4o       |
        +----+----+          | Rewrite Node |
             | Clean         +------+-------+
             v                      |
       +-----------+                |  re-score
       | Finalize  |<---------------+  (loop <= 5)
       +-----------+
             |
             v
       Clean Output
```

The LangGraph state machine implements three-way routing after each scoring step: **finalize** (clean), **revise** (still toxic, iterations remaining), or **fallback** (max iterations exhausted).

---

## Results

### Ensemble Classifier — Four-Group Comparison (Validation, n = 22,355)

All four groups use the same F2-optimized threshold search over [0.05, 0.50] for fair comparison. F2 (β = 2) weights recall twice as heavily as precision, matching the guardrail use case where missing toxic content is costlier than over-flagging.

| Method | Macro-F1 | Micro-F1 | Macro-AUC | Macro-PR-AUC |
|--------|----------|----------|-----------|--------------|
| BERT tuned | 0.6555 | 0.7360 | 0.9825 | 0.7181 |
| LLaMA tuned | 0.6624 | 0.7329 | 0.9837 | 0.7141 |
| Ensemble (uniform w=0.5) | 0.6656 | 0.7418 | 0.9846 | 0.7436 |
| **Ensemble (learned)** | **0.6716** | **0.7437** | **0.9850** | **0.7434** |

The learned ensemble achieves **Macro-F2 = 0.7821** and **Macro-Recall = 0.8837** on the validation set.

### Per-Label Learned Ensemble Weights

| Label | w\_bert | w\_llama | Threshold | F2 | F1 |
|-------|---------|----------|-----------|------|------|
| toxic | 0.30 | 0.70 | 0.30 | 0.8605 | 0.7734 |
| severe\_toxic | 0.40 | 0.60 | 0.48 | 0.6623 | 0.5102 |
| obscene | 0.50 | 0.50 | 0.30 | 0.8671 | 0.7785 |
| threat | 0.15 | 0.85 | 0.32 | 0.7273 | 0.6292 |
| insult | 0.70 | 0.30 | 0.28 | 0.8393 | 0.7475 |
| identity\_hate | 0.30 | 0.70 | 0.14 | 0.7363 | 0.5905 |

The ensemble learns complementary weights: `insult` favors BERT (w=0.70), while `threat` and `identity_hate` rely more on LLaMA (w=0.85 and 0.70 respectively). Thresholds are kept low (0.14–0.48) to maximize recall.

### Per-Label Detail (Ensemble Learned, Validation)

| Label | Support | AUC-ROC | PR-AUC | F1 | Precision | Recall |
|-------|---------|---------|--------|------|-----------|--------|
| toxic | 2,139 | 0.9862 | 0.9008 | 0.7734 | 0.6618 | 0.9303 |
| severe\_toxic | 196 | 0.9922 | 0.4918 | 0.5102 | 0.3690 | 0.8265 |
| obscene | 1,214 | 0.9929 | 0.8958 | 0.7785 | 0.6653 | 0.9382 |
| threat | 69 | 0.9609 | 0.6639 | 0.6292 | 0.5138 | 0.8116 |
| insult | 1,130 | 0.9900 | 0.8417 | 0.7475 | 0.6322 | 0.9142 |
| identity\_hate | 211 | 0.9878 | 0.6665 | 0.5905 | 0.4439 | 0.8815 |
| **MACRO** | **4,959** | **0.9850** | **0.7434** | **0.6716** | **0.5477** | **0.8837** |

### Held-Out Test Set (n = 22,355)

| Label | Support | AUC-ROC | PR-AUC | F1 | Precision | Recall |
|-------|---------|---------|--------|------|-----------|--------|
| toxic | 2,138 | 0.9862 | 0.8964 | 0.7707 | 0.6604 | 0.9252 |
| severe\_toxic | 196 | 0.9870 | 0.4693 | 0.4867 | 0.3506 | 0.7959 |
| obscene | 1,214 | 0.9931 | 0.8959 | 0.7850 | 0.6694 | 0.9489 |
| threat | 69 | 0.9887 | 0.6234 | 0.5473 | 0.4167 | 0.7971 |
| insult | 1,131 | 0.9880 | 0.8185 | 0.7268 | 0.6140 | 0.8904 |
| identity\_hate | 212 | 0.9941 | 0.7216 | 0.5473 | 0.3958 | 0.8868 |
| **MACRO** | **4,960** | **0.9895** | **0.7375** | **0.6714** | **0.5736** | **0.8285** |

Test-set performance is consistent with validation (Macro-F1: 0.6714 vs 0.6716, Macro-AUC: 0.9895 vs 0.9850), confirming no overfitting during threshold/weight search.

### Agent Pipeline Evaluation (313 samples)

The pipeline evaluation uses a stratified mix of 4 sample groups: known-toxic comments, clean (non-toxic) comments, ensemble false positives, and ensemble false negatives.

| Source Group | Samples | Metric | Value |
|-------------|---------|--------|-------|
| Known toxic | 100 | Correction success rate | 99.0% |
| Known toxic | 100 | Average iterations | 1.13 |
| False positives (from error analysis) | 57 | Correction success rate | 100.0% |
| False positives (from error analysis) | 57 | Average iterations | 1.25 |
| **Combined toxic** | **157** | **Correction success rate** | **99.4%** |
| Combined toxic | 157 | Average iterations | 1.17 |
| Combined toxic | 157 | Fallback rate | 0.6% (1/157) |
| Clean (non-toxic) | 100 | Preservation rate (pass-through) | 100.0% |
| Clean (non-toxic) | 100 | False trigger rate | 0.0% |
| False negatives (from error analysis) | 56 | Caught by pipeline | 22/56 (39.3%) |
| **Overall (excl. FN)** | **257** | **Successful outcome rate** | **99.6% (256/257)** |

> **Note:** The 56 false-negative samples are ensemble misses — the classifier scored them below threshold, so the pipeline passes them through as clean. These are classifier-level limitations, not pipeline failures, and are excluded from the pipeline success rate (99.6% = 256/257). Of these 56, 22 were incidentally caught when GPT-4o rewrites for co-occurring flagged labels cleared previously missed labels during re-scoring. Across all 313 samples, the end-to-end rate is 88.8% (278/313).

**Key takeaways:**
- The pipeline corrects **99.4%** of flagged toxic inputs within 5 iterations, with most resolved in a **single iteration** (138/157 = 87.9%).
- Clean inputs pass through untouched — **zero false triggers**.
- For ensemble false negatives (samples the classifier missed), the pipeline catches 39.3% via re-scoring after GPT-4o rewrite — an unexpected bonus, since these were already missed by the classifier.
- Only 1 out of 157 toxic inputs fell back to the safety response (0.6% fallback rate).

#### Iteration Distribution (Combined Toxic)

| Iterations | Count | Percentage |
|------------|-------|------------|
| 1 | 138 | 87.9% |
| 2 | 13 | 8.3% |
| 3 | 5 | 3.2% |
| 5 (fallback) | 1 | 0.6% |

#### Per-Label Correction Rate (Combined Toxic, 157 samples)

| Label | Triggered | Corrected | Rate |
|-------|-----------|-----------|------|
| toxic | 157 | 156 | 99.4% |
| severe\_toxic | 54 | 54 | 100.0% |
| obscene | 121 | 121 | 100.0% |
| threat | 25 | 25 | 100.0% |
| insult | 116 | 116 | 100.0% |
| identity\_hate | 52 | 52 | 100.0% |

---

## Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data?select=jigsaw-toxic-comment-train.csv) — 223,549 Wikipedia talk-page comments with 6 binary labels:

`toxic` · `severe_toxic` · `obscene` · `threat` · `insult` · `identity_hate`

Data is split using iterative stratification (`MultilabelStratifiedShuffleSplit`) with a fixed team seed (`49458345`, derived from the sum of three team members' university IDs):

| Split | Rows | Usage |
|-------|------|-------|
| Train | 178,839 (80%) | Model fine-tuning |
| Validation | 22,355 (10%) | Ensemble weight search |
| Test | 22,355 (10%) | Held-out pipeline evaluation |

---

## Repository Structure

```
DetoxiGuard/
├── agent/
│   └── pipeline.py              # LangGraph detect-and-correct agent
├── classifier/
│   ├── ensemble.py              # Per-label (weight, threshold) grid search on F2
│   ├── split.py                 # Reproducible 80/10/10 stratified split
│   ├── train_bert.py            # BERT fine-tuning (full, single-GPU)
│   └── train_llama.py           # LLaMA-3.2-1B + LoRA fine-tuning (DDP, 2× A100)
├── data/
│   ├── train_split.csv          # Generated by split.py
│   ├── val_split.csv
│   └── test_split.csv
├── eda/
│   └── eda.ipynb                # Exploratory data analysis notebook
├── evaluation/
│   ├── eval.ipynb               # Evaluation notebook
│   ├── eval_bert.py             # BERT standalone evaluation
│   ├── eval_ensemble.py         # Ensemble evaluation (val + test)
│   ├── eval_llama.py            # LLaMA standalone evaluation
│   ├── eval_pipeline.py         # Agent pipeline evaluation (313 samples)
│   ├── eval_pipeline_vis.py     # Pipeline evaluation visualizations
│   └── test_pipeline.py         # Interactive pipeline demo
├── logs/                        # Training and evaluation logs
├── outputs/
│   ├── bert_final/best_checkpoint/
│   ├── llama/best_checkpoint/
│   ├── ensemble/                # ensemble_weights.json, val_comparison.json
│   ├── eval_bert/
│   ├── eval_ensemble/
│   ├── eval_llama/
│   └── eval_pipeline/
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Setup & Reproduction

### Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA support
- HuggingFace account with access to `meta-llama/Llama-3.2-1B` (gated model)
- OpenAI API key (for the GPT-4o rewrite node in the pipeline)

### Installation

```bash
git clone https://github.com/yinruide/DetoxiGuard.git
cd DetoxiGuard
pip install -r requirements.txt
```

Key dependencies: `transformers`, `peft`, `accelerate`, `langgraph`, `openai`, `scikit-learn`, `iterstrat`, `pandas`, `torch`

### Data Preparation

1. Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data?select=jigsaw-toxic-comment-train.csv) and place it in `data/`.
2. Generate the three-way split:

```bash
python classifier/split.py --raw_csv data/train.csv --output_dir data
```

### Training

**BERT**:

```bash
python classifier/train_bert.py \
    --model_name bert-base-uncased \
    --train_csv data/train_split.csv \
    --val_csv data/val_split.csv \
    --output_dir outputs/bert_final \
    --epochs 5 --batch_size 8 --grad_accum_steps 4
```

**LLaMA + LoRA**:

```bash
torchrun --nproc_per_node=2 classifier/train_llama.py \
    --model_name meta-llama/Llama-3.2-1B \
    --train_csv data/train_split.csv \
    --val_csv data/val_split.csv \
    --output_dir outputs/llama \
    --epochs 5 --batch_size 32
```

LLaMA uses LoRA (rank 8, α = 16) applied to query/value projections. Trainable parameters: 1.72M / 1.24B total (0.14%).

**Ensemble Weight Search:**

```bash
python classifier/ensemble.py \
    --bert_ckpt outputs/bert_final/best_checkpoint \
    --bert_base bert-base-uncased \
    --llama_ckpt outputs/llama/best_checkpoint \
    --llama_base meta-llama/Llama-3.2-1B \
    --val_csv data/val_split.csv \
    --output_dir outputs/ensemble
```

### Evaluation

```bash
# Individual model evaluation
python evaluation/eval_bert.py \
    --checkpoint outputs/bert_final/best_checkpoint \
    --base_model bert-base-uncased \
    --val_csv data/val_split.csv \
    --output_dir outputs/eval_bert

python evaluation/eval_llama.py \
    --checkpoint outputs/llama/best_checkpoint \
    --base_model meta-llama/Llama-3.2-1B \
    --val_csv data/val_split.csv \
    --output_dir outputs/eval_llama

# Ensemble evaluation (val + test)
python evaluation/eval_ensemble.py \
    --bert_ckpt outputs/bert_final/best_checkpoint \
    --bert_base bert-base-uncased \
    --llama_ckpt outputs/llama/best_checkpoint \
    --llama_base meta-llama/Llama-3.2-1B \
    --val_csv data/val_split.csv \
    --test_csv data/test_split.csv \
    --ensemble_dir outputs/ensemble \
    --output_dir outputs/eval_ensemble_results

# Agent pipeline evaluation
export OPENAI_API_KEY=sk-...
python evaluation/eval_pipeline.py \
    --bert_ckpt outputs/bert_final/best_checkpoint \
    --bert_base bert-base-uncased \
    --llama_ckpt outputs/llama/best_checkpoint \
    --llama_base meta-llama/Llama-3.2-1B \
    --ensemble_dir outputs/ensemble \
    --test_csv data/test_split.csv \
    --error_csv outputs/eval_ensemble_results/val/error_analysis.csv \
    --output_dir outputs/eval_pipeline
```

### Interactive Demo

```bash
export OPENAI_API_KEY=sk-...
python agent/pipeline.py
# Enter your text: <type a message>
```

---

## Design Decisions

- **F2 over F1:** The guardrail use case penalizes missed toxic content more than over-flagging. F2 (β = 2) weights recall 2× as heavily as precision. All threshold searches — including single-model baselines — use F2 with the same [0.05, 0.50] range for fair comparison.
- **Per-label weights:** Each toxicity label has its own BERT/LLaMA weight and decision threshold, learned independently via grid search. This lets the ensemble exploit each model's relative strengths per category.
- **Threshold cap at 0.50:** Prevents the optimizer from finding high-precision/low-recall thresholds that would defeat the guardrail purpose.
- **Raw text for transformers:** Both BERT and LLaMA receive raw, unpreprocessed text — no EDA augmentation or text cleaning — since pretrained tokenizers handle tokenization internally.
- **Iterative stratification:** `MultilabelStratifiedShuffleSplit` preserves per-label class ratios across all three splits, critical for heavily imbalanced minority labels (`threat`: 0.28%, `severe_toxic`: 0.79%).
- **Weighted BCE with pos\_weight:** Both models use class-frequency-inverse `pos_weight` in `BCEWithLogitsLoss` to counteract label imbalance during training.
- **Max 5 iterations with fallback:** The agent loop caps at 5 rewrite–rescore cycles. If the text is still flagged after 5 attempts, the pipeline returns a safe pre-defined response rather than surfacing partially sanitized content. This bounds latency and API cost while guaranteeing a safe output.

---

## Computation Environment

| Task | Resource |
|----------|------|
| Training (LLaMA) | NYU HPC, 2× NVIDIA A100 40GB GPU |
| Training (BERT) | Local machine, 1× NVIDIA RTX 4070 GPU |
| Evaluation | NYU HPC, 1× L4/A100 GPU |
| Pipeline eval (GPT-4o) | OpenAI API, 313 samples, 341s total, ~$0.20 |

---

## Team Contributions

| Member | Responsibility |
|--------|----------------|
| **Ruide Yin** | LLaMA + LoRA fine-tuning, ensemble weight search (joint with Wang), pipeline evaluation, project integration |
| **Yanfu Wang** | BERT fine-tuning, ensemble weight search (joint with Yin), BERT evaluation |
| **Langyue Zhao** | LangGraph agent pipeline (`pipeline.py`), presentation deck, pipeline demo, pipeline test |


---

## License

Code in this repository is licensed under the [Apache License 2.0](LICENSE).

Model weights derived from LLaMA-3.2-1B are subject to Meta's [Llama 3.2 Community License Agreement](https://ai.meta.com/llama/license/). The Jigsaw dataset is subject to its own [Kaggle competition rules](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/rules#7-competition-data).
