"""
Generate 'rejected' responses for DPO using Qwen3-0.6B as a weak model.

Covers the two open-ended tasks where rule-based negatives are not meaningful:
  - scandiqa-sv  (reading comprehension / extractive QA)
  - swedn        (news summarization)

For each sample we generate one response from the weak model. The correct
ground-truth answer becomes 'chosen'; the weak model output becomes 'rejected'.

Output (JSONL, one record per sample):
  {
    "task":           str,          # "reading_comprehension" | "summarization"
    "source":         str,          # original filename
    "sample_id":      int,
    "prompt":         list[dict],   # [system, user] messages — input to DPOTrainer
    "chosen":         str,          # ground-truth answer
    "rejected":       str,          # Qwen3-0.6B generation
    "rejected_matches_chosen": bool # flag: model happened to be correct
  }
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse prompt formatters and system prompts from SFT data loader
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (
    SYSTEM_PROMPTS,
    _format_reading_comprehension,
    _format_summarization,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASKS = {
    "scandiqa-sv_train.json": {
        "task_type": "reading_comprehension",
        "formatter": _format_reading_comprehension,
        "system_prompt": SYSTEM_PROMPTS["reading_comprehension"],
    },
    "swedn_train.json": {
        "task_type": "summarization",
        "formatter": _format_summarization,
        "system_prompt": SYSTEM_PROMPTS["summarization"],
    },
}

# Generation params: high enough temperature to get non-trivial outputs
# but not so high that the output is pure noise
GEN_KWARGS = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> block from generated text if present."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _build_prompt_messages(system_prompt: str, user_msg: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[list[dict]],
    batch_size: int = 8,
) -> list[str]:
    """Tokenize a list of chat prompts, generate, return decoded responses."""
    all_responses = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]

        # Apply Qwen3 chat template with thinking disabled
        texts = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for msgs in batch
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                **GEN_KWARGS,
            )

        # Decode only the newly generated tokens (after the prompt)
        prompt_lengths = inputs["input_ids"].shape[1]
        for j, output in enumerate(outputs):
            new_tokens = output[prompt_lengths:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_responses.append(_strip_thinking(decoded))

        logger.info(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)}")

    return all_responses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate DPO rejected responses with weak model")
    parser.add_argument("--model-path", required=True, help="Path to Qwen3-0.6B weights")
    parser.add_argument("--data-dir", required=True, help="Directory with Swedish training JSONs")
    parser.add_argument("--output-dir", required=True, help="Where to write JSONL output files")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load weak model
    # ------------------------------------------------------------------
    logger.info(f"Loading weak model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batch generation

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.eval()

    logger.info(f"Weak model parameters: {model.num_parameters():,}")

    # ------------------------------------------------------------------
    # Process each task
    # ------------------------------------------------------------------
    total_written = 0

    for filename, task_cfg in TASKS.items():
        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning(f"Skipping missing file: {filepath}")
            continue

        task_type = task_cfg["task_type"]
        formatter = task_cfg["formatter"]
        system_prompt = task_cfg["system_prompt"]

        logger.info(f"Processing {filename} ({task_type})...")

        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        samples = raw.get("samples", [])

        # Build prompts and collect ground-truth chosen answers
        prompt_messages = []
        chosen_answers = []
        for sample in samples:
            user_msg, ground_truth = formatter(sample)
            prompt_messages.append(_build_prompt_messages(system_prompt, user_msg))
            chosen_answers.append(ground_truth)

        # Generate rejected responses in batches
        logger.info(f"  Generating {len(samples)} rejected responses (batch={args.batch_size})...")
        rejected_answers = generate_batch(model, tokenizer, prompt_messages, args.batch_size)

        # Write JSONL output
        out_stem = filename.replace(".json", "")
        out_path = output_dir / f"{out_stem}_dpo_pairs.jsonl"

        n_correct = 0
        with open(out_path, "w", encoding="utf-8") as f_out:
            for idx, (msgs, chosen, rejected) in enumerate(
                zip(prompt_messages, chosen_answers, rejected_answers)
            ):
                # Prompt is just the [system, user] turns (no assistant turn)
                prompt_only = [m for m in msgs if m["role"] != "assistant"]

                matches = chosen.strip().lower() == rejected.strip().lower()
                if matches:
                    n_correct += 1

                record = {
                    "task": task_type,
                    "source": filename,
                    "sample_id": idx,
                    "prompt": prompt_only,
                    "chosen": chosen,
                    "rejected": rejected,
                    "rejected_matches_chosen": matches,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"  Wrote {len(samples)} pairs to {out_path}")
        logger.info(f"  Weak model was accidentally correct on {n_correct}/{len(samples)} samples")
        total_written += len(samples)

    logger.info(f"Done. Total pairs written: {total_written}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
