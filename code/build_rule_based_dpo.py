"""
Build rule-based DPO preference pairs for 5 structured EuroEval Swedish tasks.

No GPU required — deterministic construction from ground-truth labels.

Tasks and rejected-response strategy:
  hellaswag-sv  letter answer  → next wrong letter in {a,b,c,d}
  mmlu-sv       letter answer  → next wrong letter in {a,b,c,d}
  swerec        sentiment      → most contrasting label
  scala-sv      acceptability  → opposite label (korrekt ↔ inkorrekt)
  suc3          NER string     → "Inga entiteter" if entities present,
                                 fabricated entity if no entities present

Output: JSONL in the same format as generate_dpo_rejected.py so that
train_dpo.py can read all files from one directory without changes.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    SYSTEM_PROMPTS,
    _extract_entities_from_bio,
    _format_acceptability,
    _format_commonsense,
    _format_knowledge,
    _format_ner,
    _format_sentiment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rejected-response constructors (one per task)
# ---------------------------------------------------------------------------

# Deterministic "next wrong letter": a→b, b→c, c→d, d→a
_NEXT_WRONG = {"a": "b", "b": "c", "c": "d", "d": "a"}

# Most contrasting sentiment for swerec
_SENTIMENT_CONTRAST = {
    "positive": "negative",
    "negative": "positive",
    "neutral":  "negative",
}

# Entity-type rotation for suc3 when we need to fabricate wrong NER
_TYPE_ROTATE = {"PER": "ORG", "ORG": "LOC", "LOC": "PER", "MISC": "PER"}


def _rejected_commonsense_knowledge(sample: dict) -> str:
    """Wrong letter: the next letter after the correct one (deterministic)."""
    return _NEXT_WRONG[sample["label"]]


def _rejected_sentiment(sample: dict) -> str:
    """Most contrasting sentiment label."""
    return _SENTIMENT_CONTRAST[sample["label"]]


def _rejected_acceptability(sample: dict) -> str:
    """Flip the binary label."""
    correct_sv = "korrekt" if sample["label"] == "correct" else "inkorrekt"
    return "inkorrekt" if correct_sv == "korrekt" else "korrekt"


def _rejected_ner(sample: dict) -> str:
    """
    If the sentence has entities: return 'Inga entiteter' (false negative).
    If it has no entities: fabricate an entity using the first capitalized
    token (likely a proper noun), rotating its type.
    """
    tokens = sample.get("tokens", [])
    labels = sample.get("labels", [])
    entities = _extract_entities_from_bio(tokens, labels)

    if entities:
        # Correct answer has entities → wrong answer claims none
        return "Inga entiteter"
    else:
        # Correct answer is "Inga entiteter" → wrong answer fabricates one.
        # Find the first capitalized token as a plausible (but wrong) span.
        fake_token = next(
            (t for t in tokens if t and t[0].isupper()),
            tokens[0] if tokens else "Stockholm",
        )
        return f"{fake_token} (PER)"


# ---------------------------------------------------------------------------
# Task registry: filename → (task_type, system_prompt, formatter, rejector)
# ---------------------------------------------------------------------------

RULE_BASED_TASKS = {
    "hellaswag-sv_train.json": {
        "task_type":     "commonsense",
        "system_prompt": SYSTEM_PROMPTS["commonsense"],
        "formatter":     _format_commonsense,
        "rejector":      _rejected_commonsense_knowledge,
    },
    "mmlu-sv_train.json": {
        "task_type":     "knowledge",
        "system_prompt": SYSTEM_PROMPTS["knowledge"],
        "formatter":     _format_knowledge,
        "rejector":      _rejected_commonsense_knowledge,
    },
    "swerec_train.json": {
        "task_type":     "sentiment",
        "system_prompt": SYSTEM_PROMPTS["sentiment"],
        "formatter":     _format_sentiment,
        "rejector":      _rejected_sentiment,
    },
    "scala-sv_train.json": {
        "task_type":     "acceptability",
        "system_prompt": SYSTEM_PROMPTS["acceptability"],
        "formatter":     _format_acceptability,
        "rejector":      _rejected_acceptability,
    },
    "suc3_train.json": {
        "task_type":     "ner",
        "system_prompt": SYSTEM_PROMPTS["ner"],
        "formatter":     _format_ner,
        "rejector":      _rejected_ner,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_pairs(data_dir: str, output_dir: str) -> None:
    data_path   = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0
    for filename, cfg in RULE_BASED_TASKS.items():
        filepath = data_path / filename
        if not filepath.exists():
            logger.warning(f"Source file not found, skipping: {filepath}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            samples = json.load(f).get("samples", [])

        out_stem  = filename.replace(".json", "")
        out_path  = output_path / f"{out_stem}_dpo_pairs.jsonl"

        n_written = n_identical = 0
        with open(out_path, "w", encoding="utf-8") as f_out:
            for idx, sample in enumerate(samples):
                user_msg, chosen  = cfg["formatter"](sample)
                rejected          = cfg["rejector"](sample)
                identical         = chosen.strip().lower() == rejected.strip().lower()
                if identical:
                    n_identical += 1

                record = {
                    "task":   cfg["task_type"],
                    "source": filename,
                    "sample_id": idx,
                    "prompt": [
                        {"role": "system", "content": cfg["system_prompt"]},
                        {"role": "user",   "content": user_msg},
                    ],
                    "chosen":   chosen,
                    "rejected": rejected,
                    "rejected_matches_chosen": identical,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

        logger.info(
            f"{filename}: {n_written} pairs written → {out_path.name} "
            f"({n_identical} identical, will be filtered at training time)"
        )
        total += n_written

    logger.info(f"Total rule-based pairs written: {total}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True, help="Swedish training JSON directory")
    parser.add_argument("--output-dir", required=True, help="Directory to write JSONL files")
    args = parser.parse_args()
    build_pairs(args.data_dir, args.output_dir)
