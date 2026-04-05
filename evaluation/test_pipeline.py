"""
test_pipeline.py
----------------
Batch-test the DetoxiGuard sanitisation pipeline and save results.

Usage:
    python evaluation/test_pipeline.py
"""

import json
import os
import sys

# ── Path setup (same as pipeline.py) ────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_CLASSIFIER_DIR = os.path.join(_REPO_ROOT, "classifier")
if _CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, _CLASSIFIER_DIR)

from agent.pipeline import init_ensemble, graph, GraphState, LABELS

# ── Test cases (fill in later) ──────────────────────────

TEST_CASES: list[dict] = [
    # ── Toxic (1-8) ───────────────��─────────────────────
    # Obvious
    {"input": "You are a worthless piece of garbage and everyone hates you.", "category": "toxic"},
    {"input": "This is the most disgusting thing I have ever seen, you toxic idiot.", "category": "toxic"},
    {"input": "Get out of here you pathetic loser, nobody wants you around.", "category": "toxic"},
    {"input": "You are absolute trash and you should be ashamed of yourself.", "category": "toxic"},
    {"input": "What a disgusting and toxic person you are, go away.", "category": "toxic"},
    # Borderline
    {"input": "I really can't stand people like you sometimes.", "category": "toxic"},
    {"input": "You're honestly just not a good person.", "category": "toxic"},
    {"input": "People like you are the reason everything goes wrong.", "category": "toxic"},

    # ── Severe Toxic (9-16) ─────────────────────────────
    # Obvious
    {"input": "I hope you die a slow and painful death you worthless scum.", "category": "severe_toxic"},
    {"input": "You are the most despicable human being alive, go kill yourself.", "category": "severe_toxic"},
    {"input": "Nobody would miss you if you disappeared forever, you disgusting waste.", "category": "severe_toxic"},
    {"input": "You deserve nothing but suffering for the rest of your miserable life.", "category": "severe_toxic"},
    {"input": "Drop dead you pathetic excuse for a human being.", "category": "severe_toxic"},
    # Borderline
    {"input": "I genuinely wish something bad would happen to you.", "category": "severe_toxic"},
    {"input": "You make everyone around you miserable and the world would be better without you.", "category": "severe_toxic"},
    {"input": "I hope your life falls apart completely.", "category": "severe_toxic"},

    # ── Obscene (17-24) ───────────���─────────────────────
    # Obvious
    {"input": "What the f*** is wrong with you, you stupid ahole.", "category": "obscene"},
    {"input": "Go f* yourself you piece of s***.", "category": "obscene"},
    {"input": "You're a dhead and everyone knows it.", "category": "obscene"},
    {"input": "This is absolute bulls and you know it, you fing idiot.", "category": "obscene"},
    {"input": "Shut the f up you ignorant b****.", "category": "obscene"},
    # Borderline
    {"input": "This is complete crap and you know it.", "category": "obscene"},
    {"input": "What the hell is your problem seriously.", "category": "obscene"},
    {"input": "That's just BS and I'm tired of hearing it.", "category": "obscene"},

    # ── Threat (25-32) ────────────────��─────────────────
    # Obvious
    {"input": "I will find you and make you pay for what you did.", "category": "threat"},
    {"input": "Keep talking like that and see what happens to you.", "category": "threat"},
    {"input": "I know where you live and you should be very afraid.", "category": "threat"},
    {"input": "Say that one more time and I will hurt you badly.", "category": "threat"},
    {"input": "You better watch your back because I am coming for you.", "category": "threat"},
    # Borderline
    {"input": "You will regret saying that, I promise you.", "category": "threat"},
    {"input": "Be careful about what you say next.", "category": "threat"},
    {"input": "Don't push me or you won't like what happens.", "category": "threat"},

    # ── Insult (33-40) ─────────────────────────���────────
    # Obvious
    {"input": "You are a complete idiot and everyone can see how stupid you are.", "category": "insult"},
    {"input": "What a moron, only a brainless fool would think that.", "category": "insult"},
    {"input": "You are dumb as a rock and twice as useless.", "category": "insult"},
    {"input": "Only an absolute imbecile would say something that stupid.", "category": "insult"},
    {"input": "You're a joke and nobody takes you seriously.", "category": "insult"},
    # Borderline
    {"input": "That was honestly a pretty dumb thing to say.", "category": "insult"},
    {"input": "I don't think you're very smart based on what you just said.", "category": "insult"},
    {"input": "You clearly don't understand what you're talking about.", "category": "insult"},

    # ── Identity Hate (41-48) ───────────────────────────
    # Obvious
    {"input": "People of that religion are all terrorists and should be banned.", "category": "identity_hate"},
    {"input": "Women are too emotional to be in leadership positions.", "category": "identity_hate"},
    {"input": "That racial group is responsible for all the crime in this city.", "category": "identity_hate"},
    {"input": "Gay people shouldn't be allowed to adopt children, it's unnatural.", "category": "identity_hate"},
    {"input": "immigrants are taking over and destroying our culture.", "category": "identity_hate"},
    # Borderline
    {"input": "I just don't think that culture fits in well here.", "category": "identity_hate"},
    {"input": "There are differences between groups that we shouldn't ignore.", "category": "identity_hate"},
    {"input": "Some religions just seem more violent than others historically.", "category": "identity_hate"},

    # ── Mixed (49-60) ──────────────��────────────────────
    # 2-label combinations
    {"input": "You are a f*ing idiot, I hope you get what's coming to you.", "category": "mixed"},
    {"input": "What a disgusting moron, you're too stupid to understand anything.", "category": "mixed"},
    {"input": "Shut up you dumb b**, nobody asked for your opinion.", "category": "mixed"},
    {"input": "People like you are trash, I hope something terrible happens to you.", "category": "mixed"},
    # 3+ label combinations
    {"input": "You stupid fing loser, get out of my sight before I do something about it.", "category": "mixed"},
    {"input": "What a worthless idiot, go f yourself and never come back.", "category": "mixed"},
    {"input": "You disgusting piece of garbage, I will make you regret this.", "category": "mixed"},
    {"input": "That group are all criminals and I'd love to see them gone for good.", "category": "mixed"},
    # Toxic content embedded in normal sentences
    {"input": "That's a great point about the economy, but you're still an idiot for missing the main issue.", "category": "mixed"},
    {"input": "I appreciate your effort here, but honestly this is complete crap and you should redo it.", "category": "mixed"},
    {"input": "The research is interesting, though people from that background never produce quality work.", "category": "mixed"},
    {"input": "Good presentation overall, but watch your back next time you present something this wrong.", "category": "mixed"},

    # ── Clean (61-70) ───────────────────────────────────
    # Strong tone but not toxic
    {"input": "I strongly disagree with your position and I think you are completely wrong.", "category": "clean"},
    {"input": "This is unacceptable and I demand an immediate explanation.", "category": "clean"},
    {"input": "I am extremely frustrated with this situation and expect better.", "category": "clean"},
    {"input": "Your argument is weak and fails to address the core issue.", "category": "clean"},
    # Sensitive words in non-toxic context
    {"input": "This movie is sickeningly good, I was hooked from the first scene.", "category": "clean"},
    {"input": "The performance was killer, absolutely blew me away.", "category": "clean"},
    {"input": "That plot twist was insane, I did not see it coming at all.", "category": "clean"},
    # Completely neutral
    {"input": "The quarterly report shows a 12% increase in revenue compared to last year.", "category": "clean"},
    {"input": "Please send me the updated project timeline by end of day Friday.", "category": "clean"},
    {"input": "The weather forecast predicts rain for most of next week.", "category": "clean"},
]


# ── Runner ──────────────────────────────────────────────

def run_single_test(case: dict) -> dict:
    """Run one test case through the pipeline, return detailed result."""
    user_input = case["input"]
    category = case["category"]

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

    return {
        "input": user_input,
        "category": category,
        "final_output": result["final_output"],
        "rounds_used": result["iteration"],
        "is_toxic_final": result["is_toxic"],
        "is_clean": not result["is_toxic"],
        "toxic_labels_final": result["toxic_labels"],
    }


def run_all_tests() -> list[dict]:
    """Run all TEST_CASES and return results list."""
    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{len(TEST_CASES)} | category: {case['category']}")
        print(f"Input: {case['input']!r}")
        print(f"{'='*60}")
        result = run_single_test(case)
        results.append(result)
        print(f"  → rounds_used: {result['rounds_used']}")
        print(f"  → is_clean: {result['is_clean']}")
        print(f"  → final_output: {result['final_output']!r}")
    return results


def print_summary(results: list[dict]) -> None:
    """Print a concise summary table to terminal."""
    if not results:
        print("\nNo test cases to summarise.")
        return

    print(f"\n{'='*80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"  {'#':<4} {'Category':<16} {'Rounds':<8} {'Clean':<8} {'Input (truncated)':<40}")
    print(f"  {'─'*4} {'─'*16} {'─'*8} {'─'*8} {'─'*40}")

    clean_count = 0
    for i, r in enumerate(results):
        truncated = r["input"][:37] + "..." if len(r["input"]) > 40 else r["input"]
        clean_str = "Yes" if r["is_clean"] else "No"
        if r["is_clean"]:
            clean_count += 1
        print(f"  {i+1:<4} {r['category']:<16} {r['rounds_used']:<8} {clean_str:<8} {truncated:<40}")

    total = len(results)
    print(f"\n  Total: {total} | Clean: {clean_count} | Still toxic: {total - clean_count}")
    print(f"  Clean rate: {clean_count/total*100:.1f}%")


def save_results(results: list[dict], output_path: str) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


# ── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    if not TEST_CASES:
        print("TEST_CASES is empty. Please add test cases before running.")
        sys.exit(0)

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

    results = run_all_tests()
    output_path = os.path.join(_REPO_ROOT, "evaluation", "test_results.json")
    save_results(results, output_path)
    print_summary(results)
