"""
Data loading and prompt formatting for EuroEval Swedish SFT.

Converts each of the 7 EuroEval task types into chat-formatted
instruction-response pairs matching the GRPO trainer's system prompts.
"""

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts — identical to GRPO trainer (_build_swedish_system_prompts)
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "sentiment": (
        "Du klassificerar sentiment i svenska texter. "
        "Svara med exakt ett ord: positiv, negativ, neutral eller blandad."
    ),
    "acceptability": (
        "Du bedömer grammatisk korrekthet i svenska meningar. "
        "Svara med exakt ett ord: korrekt eller inkorrekt."
    ),
    "ner": (
        "Du identifierar namngivna entiteter i svenska texter. "
        "Lista varje entitet med dess typ (PER, LOC, ORG). "
        "Om inga entiteter finns, skriv 'Inga entiteter'."
    ),
    "reading_comprehension": (
        "Du är en precis fråga-svar-assistent. Besvara frågan "
        "kort och korrekt baserat på den givna kontexten."
    ),
    "commonsense": (
        "Du löser frågor om sunt förnuft. Svara med enbart "
        "bokstaven för rätt alternativ (a, b, c eller d)."
    ),
    "knowledge": (
        "Du svarar på kunskapsfrågor. Svara med enbart "
        "bokstaven för rätt alternativ (a, b, c eller d)."
    ),
    "summarization": (
        "Du är en precis sammanfattare. Sammanfatta texten i två till tre "
        "meningar, håll dig till fakta och lägg inte till ny information."
    ),
}

# Map filename stem → (task_type key, formatter function name)
DATASET_REGISTRY: dict[str, str] = {
    "swerec_train.json": "sentiment",
    "scala-sv_train.json": "acceptability",
    "suc3_train.json": "ner",
    "scandiqa-sv_train.json": "reading_comprehension",
    "hellaswag-sv_train.json": "commonsense",
    "mmlu-sv_train.json": "knowledge",
    "swedn_train.json": "summarization",
}


# ---------------------------------------------------------------------------
# Per-task formatters: sample → (user_message, assistant_message)
# ---------------------------------------------------------------------------

def _format_sentiment(sample: dict) -> tuple[str, str]:
    text = sample.get("text", "")[:800]
    label = sample.get("label", "")
    user_msg = (
        "Klassificera sentimentet i följande recension som "
        "'positiv', 'negativ', 'neutral' eller 'blandad'. "
        "Svara med ett enda ord.\n\n"
        f"Recension: {text}\n\n"
        "Sentiment:"
    )
    return user_msg, label


def _format_acceptability(sample: dict) -> tuple[str, str]:
    text = sample.get("text", "")
    label = sample.get("label", "")  # "correct" / "incorrect" → Swedish
    label_sv = "korrekt" if label == "correct" else "inkorrekt"
    user_msg = (
        "Bedöm om följande mening är grammatiskt korrekt och naturlig "
        "på svenska. Svara med ett enda ord: 'korrekt' eller 'inkorrekt'.\n\n"
        f"Mening: {text}\n\n"
        "Bedömning:"
    )
    return user_msg, label_sv


def _format_ner(sample: dict) -> tuple[str, str]:
    text = sample.get("text", "")
    tokens = sample.get("tokens", [])
    labels = sample.get("labels", [])

    entities = _extract_entities_from_bio(tokens, labels)
    if entities:
        answer = "; ".join(f"{ent} ({typ})" for ent, typ in entities)
    else:
        answer = "Inga entiteter"

    user_msg = (
        "Identifiera alla namngivna entiteter i följande text. "
        "Lista varje entitet med dess typ (PER för person, LOC för plats, "
        "ORG för organisation). Om inga entiteter finns, skriv 'Inga entiteter'.\n\n"
        f"Text: {text}\n\n"
        "Entiteter:"
    )
    return user_msg, answer


def _format_reading_comprehension(sample: dict) -> tuple[str, str]:
    context = sample.get("context", "")[:800]
    question = sample.get("question", "")
    answers = sample.get("answers", {})
    answer_texts = answers.get("text", [])
    ground_truth = answer_texts[0] if answer_texts else ""

    user_msg = (
        f"Kontext: {context}\n\n"
        f"Fråga: {question}\n\n"
        "Svara kort baserat på kontexten."
    )
    return user_msg, ground_truth


def _format_commonsense(sample: dict) -> tuple[str, str]:
    text = sample.get("text", "")
    label = sample.get("label", "")
    user_msg = (
        f"{text}\n\n"
        "Välj rätt svarsalternativ. Svara med enbart bokstaven (a, b, c eller d):"
    )
    return user_msg, label


def _format_knowledge(sample: dict) -> tuple[str, str]:
    text = sample.get("text", "")
    label = sample.get("label", "")
    user_msg = (
        f"{text}\n\n"
        "Välj rätt svarsalternativ. Svara med enbart bokstaven (a, b, c eller d):"
    )
    return user_msg, label


def _format_summarization(sample: dict) -> tuple[str, str]:
    article = sample.get("text", "")[:1500]
    target = sample.get("target_text", "")
    user_msg = (
        "Sammanfatta följande text i två till tre meningar.\n\n"
        f"Text: {article}\n\n"
        "Sammanfattning:"
    )
    return user_msg, target


# ---------------------------------------------------------------------------
# BIO entity extraction (from GRPO trainer)
# ---------------------------------------------------------------------------

def _extract_entities_from_bio(
    tokens: list[str], labels: list[str]
) -> list[tuple[str, str]]:
    entities = []
    current_tokens: list[str] = []
    current_type: str | None = None

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
            current_tokens = [token]
            current_type = label[2:]
        elif label.startswith("I-") and current_tokens:
            current_tokens.append(token)
        else:
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
                current_tokens = []
                current_type = None

    if current_tokens:
        entities.append((" ".join(current_tokens), current_type))

    return entities


# ---------------------------------------------------------------------------
# Formatter dispatch
# ---------------------------------------------------------------------------

FORMATTERS = {
    "sentiment": _format_sentiment,
    "acceptability": _format_acceptability,
    "ner": _format_ner,
    "reading_comprehension": _format_reading_comprehension,
    "commonsense": _format_commonsense,
    "knowledge": _format_knowledge,
    "summarization": _format_summarization,
}


def _sample_to_messages(
    sample: dict[str, Any], task_type: str
) -> list[dict[str, str]]:
    """Convert a raw sample into a chat messages list."""
    formatter = FORMATTERS[task_type]
    user_msg, assistant_msg = formatter(sample)
    system_prompt = SYSTEM_PROMPTS[task_type]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_format_dataset(data_dir: str) -> Dataset:
    """Load all EuroEval Swedish JSON files and format as chat messages.

    Returns a HuggingFace Dataset with a single 'messages' column,
    ready for tokenization with a chat template.
    """
    data_path = Path(data_dir)
    all_messages: list[list[dict[str, str]]] = []

    for filename, task_type in DATASET_REGISTRY.items():
        filepath = data_path / filename
        if not filepath.exists():
            logger.warning(f"Dataset not found, skipping: {filepath}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = data.get("samples", [])
        count = 0
        for sample in samples:
            messages = _sample_to_messages(sample, task_type)
            all_messages.append(messages)
            count += 1

        logger.info(f"Loaded {count} samples from {filename} ({task_type})")

    logger.info(f"Total training examples: {len(all_messages)}")

    return Dataset.from_dict({"messages": all_messages})
