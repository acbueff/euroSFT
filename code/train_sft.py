"""
SFT baseline trainer for Qwen3-1.7B on EuroEval Swedish.

Pure supervised fine-tuning — no distillation, no LoRA.
Designed for direct comparison against the GRPO-trained model.

Target: Leonardo HPC (A100, SLURM, offline compute nodes).
"""

import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data_loader import load_and_format_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config(Path(__file__).parent / "config.yaml")

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Allow env-var overrides for Leonardo paths
    model_path = os.environ.get("MODEL_PATH") or model_cfg.get("local_path") or model_cfg["name"]
    data_dir = os.environ.get("DATA_DIR") or data_cfg["train_dir"]
    output_dir = os.environ.get("OUTPUT_DIR") or train_cfg["output_dir"]

    logger.info(f"Model: {model_path}")
    logger.info(f"Data:  {data_dir}")
    logger.info(f"Output: {output_dir}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    logger.info(f"Model parameters: {model.num_parameters():,}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info("Loading and formatting training data...")
    train_dataset = load_and_format_dataset(data_dir)
    logger.info(f"Training examples: {len(train_dataset)}")

    # Shuffle the dataset (mixing tasks uniformly)
    train_dataset = train_dataset.shuffle(seed=train_cfg["seed"])

    # ------------------------------------------------------------------
    # SFT Config
    # ------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_seq_length=train_cfg["max_seq_length"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        seed=train_cfg["seed"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=True,
        report_to="none",
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting SFT training...")
    train_result = trainer.train()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Training complete.")
    logger.info(f"  Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"  Runtime: {metrics.get('train_runtime', 0):.0f}s")


if __name__ == "__main__":
    main()
