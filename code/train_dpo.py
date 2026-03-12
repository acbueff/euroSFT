"""
DPO trainer for Qwen3-1.7B on EuroEval Swedish.

Direct Preference Optimization using weak-model-generated preference pairs
for the two open-ended tasks (reading comprehension, summarization).

Policy:    Qwen3-1.7B + LoRA adapters (trainable)
Reference: Qwen3-1.7B base weights (frozen — shared automatically by TRL
           when peft_config is provided, so only one model is loaded)

Target: Leonardo HPC (A100, SLURM, offline compute nodes).
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config_dpo.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dpo_dataset(data_dir: str, filenames: list[str]) -> Dataset:
    """Load JSONL preference pairs and convert to TRL conversational format.

    Returns a Dataset with three columns:
      - prompt:   list of {role, content} dicts (system + user turns)
      - chosen:   list of {role, content} dict  (assistant turn, correct)
      - rejected: list of {role, content} dict  (assistant turn, weak model)

    Pairs where rejected accidentally equals chosen are filtered out.
    """
    data_path = Path(data_dir)
    records = []

    for filename in filenames:
        filepath = data_path / filename
        if not filepath.exists():
            logger.warning(f"DPO data file not found, skipping: {filepath}")
            continue

        n_loaded = n_filtered = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())

                # Skip any pairs where the weak model accidentally got it right
                if item.get("rejected_matches_chosen", False):
                    n_filtered += 1
                    continue

                records.append({
                    "prompt": item["prompt"],  # [system, user] messages
                    "chosen": [{"role": "assistant", "content": item["chosen"]}],
                    "rejected": [{"role": "assistant", "content": item["rejected"]}],
                })
                n_loaded += 1

        logger.info(f"Loaded {n_loaded} pairs from {filename} (filtered {n_filtered} identical pairs)")

    logger.info(f"Total DPO pairs: {len(records)}")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config(Path(__file__).parent / "config_dpo.yaml")

    model_cfg  = config["model"]
    data_cfg   = config["data"]
    lora_cfg   = config["lora"]
    dpo_cfg    = config["dpo"]
    train_cfg  = config["training"]

    # Allow env-var overrides for Leonardo paths
    model_path = os.environ.get("MODEL_PATH") or model_cfg.get("local_path") or model_cfg["name"]
    data_dir   = os.environ.get("DPO_DATA_DIR") or data_cfg["dpo_data_dir"]
    output_dir = os.environ.get("OUTPUT_DIR") or train_cfg["output_dir"]

    logger.info(f"Model:      {model_path}")
    logger.info(f"DPO data:   {data_dir}")
    logger.info(f"Output:     {output_dir}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for DPO batch processing

    # ------------------------------------------------------------------
    # Base model (policy — LoRA adapters will be added by DPOTrainer)
    # ------------------------------------------------------------------
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    logger.info(f"Base model parameters: {model.num_parameters():,}")

    # ------------------------------------------------------------------
    # LoRA config
    # When peft_config is passed to DPOTrainer, TRL automatically:
    #   - Wraps the policy model with LoRA (trainable)
    #   - Uses the unwrapped base weights as the frozen reference model
    # This avoids loading a second copy of the 1.7B model in memory.
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info(f"LoRA: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info("Loading DPO preference pairs...")
    train_dataset = load_dpo_dataset(data_dir, data_cfg["files"])
    logger.info(f"DPO training pairs: {len(train_dataset)}")

    train_dataset = train_dataset.shuffle(seed=train_cfg["seed"])

    # ------------------------------------------------------------------
    # DPO Config
    # ------------------------------------------------------------------
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=dpo_cfg["beta"],
        loss_type=dpo_cfg["loss_type"],
        max_length=dpo_cfg["max_length"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        seed=train_cfg["seed"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,  # must be False for DPO
        report_to="none",
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,} ({100*trainable/model.num_parameters():.2f}%)")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting DPO training...")
    train_result = trainer.train()

    # ------------------------------------------------------------------
    # Save LoRA adapter
    # ------------------------------------------------------------------
    logger.info(f"Saving DPO adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("DPO training complete.")
    logger.info(f"  Loss:    {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"  Runtime: {metrics.get('train_runtime', 0):.0f}s")
    logger.info(f"  Adapter: {output_dir}")


if __name__ == "__main__":
    main()
