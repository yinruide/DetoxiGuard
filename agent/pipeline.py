"""
pipeline.py
-----------
LangGraph agent pipeline for toxic comment detection and correction.

Flow: generate draft -> score toxicity -> if toxic, revise -> re-score -> loop
"""

import sys
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

# Allow imports from the project root so that `classifier.ensemble` is reachable
# regardless of where the script is executed from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from classifier.ensemble import load_ensemble, predict_probs

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class GraphState(TypedDict):
    user_input: str
    draft_response: str
    toxicity_probs: list[float]  # length 6, label order matches LABELS
    is_toxic: bool
    iteration: int
    max_iterations: int  # default 3
    system_threshold: float  # default 0.5
    final_response: str


# ──────────────────────────────────────────────
# Ensemble initialisation
# ──────────────────────────────────────────────

def init_ensemble(
    bert_ckpt: str,
    bert_base: str,
    llama_ckpt: str,
    llama_base: str,
    ensemble_dir: str,
    device: str = "cuda",
) -> None:
    """Call once before running the pipeline to load both models."""
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

def generate_response(state: GraphState) -> dict:
    """Mock: produce a deliberately toxic draft for testing."""
    draft = (
        "You're such an idiot, I hate you and everyone like you. "
        "Go kill yourself, you worthless piece of trash."
    )
    print(f"[generate_response] draft: {draft!r}")
    return {"draft_response": draft, "iteration": 0}


def score_toxicity(state: GraphState) -> dict:
    """Score the current draft_response using the BERT+LLaMA ensemble."""
    probs = predict_probs([state["draft_response"]])[0]  # shape (6,)
    probs_list = probs.tolist()
    threshold = state["system_threshold"]
    toxic = any(p >= threshold for p in probs_list)

    print("[score_toxicity] scores:")
    for label, p in zip(LABELS, probs_list):
        flag = " !!!" if p >= threshold else ""
        print(f"  {label:15s}: {p:.4f}{flag}")
    print(f"  is_toxic = {toxic}")

    return {"toxicity_probs": probs_list, "is_toxic": toxic}


def revise_response(state: GraphState) -> dict:
    """Mock: simulate a revision by prepending a tag."""
    new_iter = state["iteration"] + 1
    revised = f"[Revised v{new_iter}] {state['draft_response']}"
    print(f"[revise_response] iteration {new_iter}: {revised!r}")
    return {"draft_response": revised, "iteration": new_iter}


def finalize(state: GraphState) -> dict:
    """Copy the current draft to final_response."""
    final = state["draft_response"]
    print(f"[finalize] final_response: {final!r}")
    return {"final_response": final}


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

graph_builder.add_node("generate_response", generate_response)
graph_builder.add_node("score_toxicity", score_toxicity)
graph_builder.add_node("revise_response", revise_response)
graph_builder.add_node("finalize", finalize)

graph_builder.add_edge(START, "generate_response")
graph_builder.add_edge("generate_response", "score_toxicity")
graph_builder.add_conditional_edges("score_toxicity", after_score)
graph_builder.add_edge("revise_response", "score_toxicity")
graph_builder.add_edge("finalize", END)

graph = graph_builder.compile()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def run_pipeline(user_input: str) -> str:
    """Run the full detect-and-correct pipeline, return the final response."""
    initial_state: GraphState = {
        "user_input": user_input,
        "draft_response": "",
        "toxicity_probs": [0.0] * 6,
        "is_toxic": False,
        "iteration": 0,
        "max_iterations": 3,
        "system_threshold": 0.5,
        "final_response": "",
    }
    result = graph.invoke(initial_state)
    return result["final_response"]


if __name__ == "__main__":
    init_ensemble(
        bert_ckpt="outputs/bert_final/best_checkpoint",
        bert_base="bert-base-uncased",
        llama_ckpt="outputs/llama_lora/best_checkpoint",
        llama_base="meta-llama/Llama-3.2-1B",
        ensemble_dir="outputs/ensemble",
    )
    answer = run_pipeline("tell me something")
    print(f"\n=== Pipeline returned ===\n{answer}")
