"""
LoRA SFT trainer for Qwen3-1.7B on EuroEval Swedish.

Low-rank adapter fine-tuning — base model weights are frozen,
only LoRA adapter weights are trained (~10-20M parameters).

Target: Leonardo HPC (A100, SLURM, offline compute nodes).
"""

import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data_loader import load_and_format_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config_lora.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config(Path(__file__).parent / "config_lora.yaml")

    model_cfg = config["model"]
    data_cfg = config["data"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    # Allow env-var overrides for Leonardo paths
    model_path = os.environ.get("MODEL_PATH") or model_cfg.get("local_path") or model_cfg["name"]
    data_dir = os.environ.get("DATA_DIR") or data_cfg["train_dir"]
    output_dir = os.environ.get("OUTPUT_DIR") or train_cfg["output_dir"]

    logger.info(f"Model:  {model_path}")
    logger.info(f"Data:   {data_dir}")
    logger.info(f"Output: {output_dir}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Base model (loaded in bfloat16; LoRA keeps it frozen)
    # ------------------------------------------------------------------
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    total_params = model.num_parameters()
    logger.info(f"Base model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # LoRA config
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info(
        f"LoRA config: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}, "
        f"target_modules={lora_cfg['target_modules']}"
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info("Loading and formatting training data...")
    train_dataset = load_and_format_dataset(data_dir)
    logger.info(f"Training examples: {len(train_dataset)}")

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
        max_length=train_cfg["max_seq_length"],
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
    # Trainer (peft_config triggers automatic LoRA wrapping)
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Log trainable vs frozen parameter counts
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,} ({100 * trainable / total_params:.2f}% of base model)")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting LoRA SFT training...")
    train_result = trainer.train()

    # ------------------------------------------------------------------
    # Save adapter weights only (not the full base model)
    # ------------------------------------------------------------------
    logger.info(f"Saving LoRA adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Training complete.")
    logger.info(f"  Loss:    {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"  Runtime: {metrics.get('train_runtime', 0):.0f}s")
    logger.info(f"  Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
