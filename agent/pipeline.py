"""
pipeline.py
-----------
LangGraph agent pipeline for user input sanitisation.

Flow: score user input -> if toxic, GPT revise -> re-score -> loop until clean or max iterations -> output
"""

import sys
import os
from typing import TypedDict

import openai
from langgraph.graph import StateGraph, START, END

# Allow imports from the project root so that `classifier.ensemble` is reachable
# regardless of where the script is executed from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ensemble.py internally does `import train_bert` / `import train_llama` (bare
# module names), so the classifier/ directory must also be on sys.path.
_CLASSIFIER_DIR = os.path.join(_REPO_ROOT, "classifier")
if _CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, _CLASSIFIER_DIR)

from classifier.ensemble import load_ensemble, predict, LABELS

# Set to False to use the real BERT+LLaMA ensemble scorer.
USE_MOCK_SCORER = False

# ──────────────────────────────────────────────
# OpenAI client (single instance, reused across nodes)
# ──────────────────────────────────────────────

_openai_client: openai.OpenAI | None = None


def _get_openai_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI()
    return _openai_client


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class GraphState(TypedDict):
    user_input: str
    current_text: str  # text being checked / rewritten; starts as user_input
    toxicity_probs: list[float]  # length 6, label order matches LABELS
    is_toxic: bool
    toxic_labels: dict[str, float]  # triggered labels with their probs
    iteration: int
    max_iterations: int  # default 3
    final_output: str


# ──────────────────────────────────────────────
# Ensemble initialisation
# ──────────────────────────────────────────────

def init_ensemble(
    bert_ckpt: str,
    bert_base: str,
    llama_ckpt: str,
    llama_base: str,
    ensemble_dir: str,
    device: str = "cpu",
) -> None:
    """Call once before running the pipeline to load both models."""
    if USE_MOCK_SCORER:
        print("[init_ensemble] USE_MOCK_SCORER=True, skipping model loading.")
        return
    weights_path = os.path.join(ensemble_dir, "ensemble_weights.json")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Ensemble weights not found at {weights_path}. "
            "Run `python classifier/ensemble.py` first to generate the weights file."
        )
    load_ensemble(bert_ckpt, bert_base, llama_ckpt, llama_base, ensemble_dir, device)


# ──────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────

def score_toxicity(state: GraphState) -> dict:
    """Score current_text using the BERT+LLaMA ensemble."""
    _MOCK_THRESHOLD = 0.5

    if USE_MOCK_SCORER:
        probs_row = [0.8, 0.1, 0.75, 0.0, 0.6, 0.0]
        triggered: dict[str, float] = {
            label: probs_row[i]
            for i, label in enumerate(LABELS)
            if probs_row[i] >= _MOCK_THRESHOLD
        }
        toxic = len(triggered) > 0
    else:
        probs, preds = predict([state["current_text"]])
        probs_row = probs[0].tolist()   # (6,)
        preds_row = preds[0]            # (6,) int
        toxic = bool(preds_row.any())
        triggered = {
            label: round(probs_row[i], 4)
            for i, label in enumerate(LABELS)
            if preds_row[i] == 1
        }

    print("[score_toxicity] scores:")
    for label, p in zip(LABELS, probs_row):
        flag = " !!!" if label in triggered else ""
        print(f"  {label:15s}: {p:.4f}{flag}")
    print(f"  is_toxic = {toxic}")
    if triggered:
        print(f"  toxic_labels = {triggered}")

    return {"toxicity_probs": probs_row, "is_toxic": toxic, "toxic_labels": triggered}


def revise_response(state: GraphState) -> dict:
    """Rewrite current_text to remove toxic content while preserving meaning."""
    new_iter = state["iteration"] + 1
    triggered = state.get("toxic_labels", {})

    if not triggered:
        print("[revise_response] no toxic labels triggered, skipping revision")
        return {"iteration": new_iter}

    trigger_str = ", ".join(f"{l}({p})" for l, p in triggered.items())

    client = _get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a content sanitisation assistant. Your task is to "
                    "rewrite text that has been flagged as toxic. Only remove or "
                    "rephrase the toxic parts — preserve the original meaning and "
                    "useful information as much as possible."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original text:\n{state['current_text']}\n\n"
                    f"Triggered toxic labels: {trigger_str}\n\n"
                    "Please only fix the flagged issues above. "
                    "Keep everything else intact."
                ),
            },
        ],
    )
    revised = resp.choices[0].message.content or state["current_text"]
    print(f"[revise_response] iteration {new_iter}: {revised!r}")
    return {"current_text": revised, "iteration": new_iter}


def finalize(state: GraphState) -> dict:
    """Copy current_text to final_output."""
    final = state["current_text"]
    print(f"[finalize] final_output: {final!r}")
    return {"final_output": final}


# ──────────────────────────────────────────────
# Routing
# ──────────────────────────────────────────────

def after_score(state: GraphState) -> str:
    """Decide whether to revise or finalize after scoring."""
    if not state["is_toxic"] or state["iteration"] >= state["max_iterations"]:
        return "finalize"
    return "revise_response"


# ──────────────────────────────────────────────
# Graph
# ──────────────────────────────────────────────

graph_builder = StateGraph(GraphState)

graph_builder.add_node("score_toxicity", score_toxicity)
graph_builder.add_node("revise_response", revise_response)
graph_builder.add_node("finalize", finalize)

graph_builder.add_edge(START, "score_toxicity")
graph_builder.add_conditional_edges("score_toxicity", after_score)
graph_builder.add_edge("revise_response", "score_toxicity")
graph_builder.add_edge("finalize", END)

graph = graph_builder.compile()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def run_pipeline(user_input: str) -> str:
    """Run the sanitisation pipeline, return the cleaned text."""
    initial_state: GraphState = {
        "user_input": user_input,
        "current_text": user_input,
        "toxicity_probs": [0.0] * 6,
        "is_toxic": False,
        "toxic_labels": {},
        "iteration": 0,
        "max_iterations": 3,
        "final_output": "",
    }
    result = graph.invoke(initial_state)
    return result["final_output"]


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("请先设置环境变量 OPENAI_API_KEY，例如：export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    init_ensemble(
        bert_ckpt=os.path.join(_REPO_ROOT, "outputs/bert_final/best_checkpoint"),
        bert_base="bert-base-uncased",
        llama_ckpt=os.path.join(_REPO_ROOT, "outputs/llama_lora/best_checkpoint"),
        llama_base="meta-llama/Llama-3.2-1B",
        ensemble_dir=os.path.join(_REPO_ROOT, "outputs/ensemble"),
        device="cpu",
    )
    user_input = input("请输入你的问题：")
    result = run_pipeline(user_input)
    print("\n=== Pipeline returned ===")
    print(result)
