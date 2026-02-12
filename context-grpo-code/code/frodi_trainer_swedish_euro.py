"""
FRÓÐI Trainer - SWEDISH EUROEVAL + SELF-PLAY ADAPTATION

Implements GRPO RL with tasks sourced from EuroEval Swedish training sets
and adaptive self-play tasks generated from FineWeb corpus text.

TRAINING DATA (EuroEval Swedish):
  - swerec (sentiment classification)
  - scala-sv (linguistic acceptability)
  - suc3 (named entity recognition)
  - scandiqa-sv (reading comprehension)
  - hellaswag-sv (common-sense reasoning)
  - mmlu-sv (knowledge)

SELF-PLAY: Once overall EuroEval accuracy >= 70% and at least one eligible
type is mastered, synthetic tasks are generated from FineWeb corpus text
for mastered types (sentiment, acceptability, NER, reading comprehension).
Self-play ratio scales with the number of eligible mastered types (max 50%).

ADAPTIVE WEIGHTING: Unmastered task types (accuracy < 80%) get 3x sampling
weight; mastered types get 1x.  This focuses training on weak areas.

REWARD (three-way routing):
  - EuroEval + deterministic type (sentiment, acceptability, commonsense,
    knowledge) + gold answer -> string matching (no LLM judge).
  - EuroEval + generative type (NER, RC) + gold answer -> LLM judge with gold.
  - Self-play (any type) -> LLM judge with correct_example from EuroEval.
"""

import torch
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import time
import logging
import json
import csv
import os
import copy
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers import get_linear_schedule_with_warmup
from datasets import Dataset

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator

from .task_buffer import TaskBuffer, AdaptiveWeightSchedulerSimple, Task

logger = logging.getLogger(__name__)


def _extract_texts_from_corpus(corpus_data) -> List[str]:
    """
    Extract text list from a corpus that could be a Dataset object or a plain dict.

    Args:
        corpus_data: Either a HuggingFace Dataset, a dict with 'text' key, or None

    Returns:
        List of text strings
    """
    if corpus_data is None:
        return []

    # Try as HuggingFace Dataset first
    if hasattr(corpus_data, '__getitem__') and hasattr(corpus_data, '__len__'):
        try:
            # Dataset objects support column access like corpus_data['text']
            return list(corpus_data['text'])
        except (KeyError, TypeError):
            pass

    # Try as plain dict
    if isinstance(corpus_data, dict):
        return corpus_data.get('text', [])

    return []


# === EuroEval Dataset Configuration ===

EUROEVAL_DATASETS = {
    "swerec": {
        "file": "swerec_train.json",
        "task_type": "sentiment",
        "description": "Swedish sentiment classification",
    },
    "scala-sv": {
        "file": "scala-sv_train.json",
        "task_type": "acceptability",
        "description": "Swedish linguistic acceptability",
    },
    "suc3": {
        "file": "suc3_train.json",
        "task_type": "ner",
        "description": "Swedish named entity recognition",
    },
    "scandiqa-sv": {
        "file": "scandiqa-sv_train.json",
        "task_type": "reading_comprehension",
        "description": "Swedish reading comprehension",
    },
    "hellaswag-sv": {
        "file": "hellaswag-sv_train.json",
        "task_type": "commonsense",
        "description": "Swedish common-sense reasoning",
    },
    "mmlu-sv": {
        "file": "mmlu-sv_train.json",
        "task_type": "knowledge",
        "description": "Swedish knowledge (MMLU)",
    },
}

# Per-task mastery: a task type is considered "mastered" when its accuracy
# exceeds this value.  Mastered types are down-weighted (1x) in the task
# buffer; unmastered types are up-weighted (3x) to focus training effort.
TASK_MASTERY_THRESHOLD = 0.80
# Minimum number of evaluated samples per task type before mastery is assessed.
MIN_SAMPLES_FOR_MASTERY = 20
# Core EuroEval task types.
CORE_TASK_TYPES = ["sentiment", "acceptability", "ner", "reading_comprehension", "commonsense"]

# Self-play configuration: synthetic tasks from FineWeb corpus for mastered types.
SELFPLAY_UNLOCK_THRESHOLD = 0.70       # Overall EuroEval accuracy needed to unlock self-play
SELFPLAY_ELIGIBLE_TYPES = {"sentiment", "acceptability", "ner", "reading_comprehension"}
MIN_EUROEVAL_RATIO = 0.50               # Task buffer always >= 50% real EuroEval
MAX_SELFPLAY_RATIO = 0.50               # Self-play at most 50% of buffer
MAX_CORPUS_TEXTS = 50000                # Cap on monolingual texts kept in memory


def _extract_entities_from_bio(tokens: List[str], labels: List[str]) -> List[Tuple[str, str]]:
    """Extract entity spans from BIO-tagged tokens.

    Returns list of (entity_text, entity_type) tuples.
    """
    entities = []
    current_entity_tokens = []
    current_type = None

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            # Save previous entity if exists
            if current_entity_tokens:
                entities.append((" ".join(current_entity_tokens), current_type))
            current_entity_tokens = [token]
            current_type = label[2:]
        elif label.startswith("I-") and current_entity_tokens:
            current_entity_tokens.append(token)
        else:
            if current_entity_tokens:
                entities.append((" ".join(current_entity_tokens), current_type))
                current_entity_tokens = []
                current_type = None

    # Don't forget the last entity
    if current_entity_tokens:
        entities.append((" ".join(current_entity_tokens), current_type))

    return entities


def load_euroeval_training_data(data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load EuroEval training data from JSON files.

    Args:
        data_dir: Path to the sv/ directory containing EuroEval training JSONs

    Returns:
        Dictionary mapping dataset name to list of samples
    """
    euroeval_data = {}
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"EuroEval data directory not found: {data_dir}")
        return euroeval_data

    for dataset_name, config in EUROEVAL_DATASETS.items():
        file_path = data_path / config["file"]
        if not file_path.exists():
            logger.warning(f"EuroEval dataset not found: {file_path}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            samples = data.get("samples", [])
            euroeval_data[dataset_name] = samples
            logger.info(
                f"Loaded {len(samples)} samples from {dataset_name} "
                f"({config['task_type']})"
            )
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")

    total = sum(len(v) for v in euroeval_data.values())
    logger.info(f"Total EuroEval training samples loaded: {total}")
    return euroeval_data


class SwedishKnowledgeDistillation:
    """
    Two-Stage Knowledge Distillation for Swedish Language Acquisition
    
    Stage 1 (CPT): Continued Pre-Training
        - ~5-10B tokens of Swedish text
        - 1 epoch (see every token once)
        - 85% Swedish + 15% English anchor
        - Higher learning rate (1e-4) for rewiring embeddings
    
    Stage 2 (SFT): Supervised Fine-Tuning
        - ~50k-100k instruction samples
        - 2-3 epochs
        - Data mix: 50% native Swedish QA, 30% translated instructions, 20% parallel translation
        - Temperature 3.0 for soft labels (dark knowledge)
        - ANTI-FORGETTING: Replay buffer + diverse tasks + KL regularization
    """
    
    def __init__(
        self,
        student_model: AutoModelForCausalLM,
        student_tokenizer: AutoTokenizer,
        teacher_model: AutoModelForCausalLM,
        teacher_tokenizer: AutoTokenizer,
        config: Dict[str, Any],
        model_config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.config = config
        self.model_config = model_config
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        
        # Ensure pad tokens are set
        self._ensure_pad_token(self.student_tokenizer)
        self._ensure_pad_token(self.teacher_tokenizer)
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'outputs/distillation_swedish'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache directory
        self.cache_dir = Path(config.get('cache_dir', self.output_dir / 'cache'))
        if config.get('cache_teacher_logits', False):
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Teacher logits caching enabled at: {self.cache_dir}")

        # Metrics tracking
        self.metrics = {
            'cpt_loss': [],
            'sft_loss': [],
            'kl_loss': [],
            'ce_loss': [],
            'replay_loss': []
        }
        
        # Reference model for KL regularization (created during SFT)
        self.reference_model = None
        
        logger.info("Swedish Knowledge Distillation initialized")
        student_path = model_config.get('student')
        teacher_path = model_config.get('teacher')
        logger.info(f"Student model: {student_path} ({'local' if student_path.startswith('/') else 'HF'})")
        logger.info(f"Teacher model: {teacher_path} ({'local' if teacher_path.startswith('/') else 'HF'})")
    
    def _ensure_pad_token(self, tokenizer: AutoTokenizer):
        """Ensure tokenizer has a pad token"""
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
    
    def _get_teacher_device(self) -> torch.device:
        """Get the device of the teacher model"""
        try:
            return next(self.teacher_model.parameters()).device
        except Exception:
            return self.device
    
    def _generate_teacher_output(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        enable_thinking: bool = False
    ) -> str:
        """Generate output from teacher model (single sample)"""
        # For backward compatibility or single calls, though batched is preferred
        return self._generate_teacher_outputs_batched([prompt], max_new_tokens)[0]

    def _generate_teacher_outputs_batched(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 256, 
        batch_size: int = None
    ) -> List[str]:
        """
        Batch generate outputs from teacher model.
        Much faster than sequential generation.
        """
        teacher_device = self._get_teacher_device()
        
        # Get batch size from config if not provided
        if batch_size is None:
            batch_size = self.config.get('teacher_generation', {}).get('batch_size', 8)
            
        use_cache = self.config.get('teacher_generation', {}).get('use_cache', True)
        
        all_outputs = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Apply chat template to each prompt in the batch
            batch_input_texts = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                try:
                    # Use chat template if available
                    text = self.teacher_tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                except Exception:
                    # Fallback
                    text = prompt
                batch_input_texts.append(text)
                
            # Tokenize batch
            encoding = self.teacher_tokenizer(
                batch_input_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            ).to(teacher_device)
            
            # Generate
            with torch.no_grad():
                outputs = self.teacher_model.generate(
                    **encoding,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic
                    use_cache=use_cache, # Use KV cache for speed
                    pad_token_id=self.teacher_tokenizer.eos_token_id
                )
            
            # Decode
            for j, output in enumerate(outputs):
                # Skip input tokens
                input_len = encoding['input_ids'].shape[1]
                # Be careful if output length is shorter than input (shouldn't happen with generate)
                generated_tokens = output[input_len:]
                text = self.teacher_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                all_outputs.append(text.strip())
            
            # Clean up memory
            del encoding, outputs
            torch.cuda.empty_cache()
            
        return all_outputs
    
    def run_cpt_stage(self, corpora: Dict[str, Dataset]):
        """
        Stage 1: Continued Pre-Training (CPT)
        
        Train on Swedish monolingual data with English anchor to maintain reasoning.
        """
        logger.info("=" * 50)
        logger.info("STAGE 1: CONTINUED PRE-TRAINING (CPT)")
        logger.info("=" * 50)
        
        cpt_config = self.config.get('stage1_cpt', {})
        epochs = cpt_config.get('epochs', 1)
        batch_size = cpt_config.get('batch_size', 4)
        max_length = cpt_config.get('max_length', 1024)
        grad_accum = cpt_config.get('gradient_accumulation_steps', 16)
        lr = float(cpt_config.get('learning_rate', 1e-4))
        swedish_ratio = cpt_config.get('swedish_ratio', 0.85)
        
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Max length: {max_length}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Swedish ratio: {swedish_ratio}")
        
        # Setup optimizer with higher LR for language adaptation
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        # Get Swedish and English texts from Dataset objects or dicts
        swedish_texts = _extract_texts_from_corpus(corpora.get('sv'))
        english_texts = _extract_texts_from_corpus(corpora.get('en'))
        
        if not swedish_texts:
            logger.error("No Swedish texts found for CPT!")
            return
        
        logger.info(f"Swedish texts: {len(swedish_texts)}")
        logger.info(f"English anchor texts: {len(english_texts)}")
        
        # Create mixed dataset
        total_samples = len(swedish_texts)
        n_english = int(total_samples * (1 - swedish_ratio))
        
        # Sample English texts if we have fewer than needed
        if english_texts and len(english_texts) < n_english:
            english_sample = english_texts * (n_english // len(english_texts) + 1)
            english_sample = english_sample[:n_english]
        elif english_texts:
            english_sample = random.sample(english_texts, min(n_english, len(english_texts)))
        else:
            english_sample = []
        
        # Combine and shuffle
        all_texts = [(text, 'sv') for text in swedish_texts] + [(text, 'en') for text in english_sample]
        random.shuffle(all_texts)
        
        logger.info(f"Total CPT samples: {len(all_texts)}")
        
        # Training loop
        self.student_model.train()
        global_step = 0
        running_loss = 0.0
        
        for epoch in range(epochs):
            logger.info(f"CPT Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(all_texts), batch_size):

                if (global_step // grad_accum) >= 10000: # After 10000 steps, we should exit the loop
                    logger.info(f"Reached 10000 steps, exiting CPT loop")
                    break

                batch_texts = [t[0] for t in all_texts[i:i + batch_size]]
                
                # Tokenize batch
                inputs = self.student_tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Forward pass (standard language modeling)
                outputs = self.student_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['input_ids'],
                    use_cache=False
                )
                
                loss = outputs.loss / grad_accum
                loss.backward()
                
                running_loss += loss.item() * grad_accum
                epoch_loss += loss.item() * grad_accum
                n_batches += 1
                global_step += 1
                
                # Optimizer step
                if global_step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if (global_step // grad_accum) % 100 == 0:
                        avg_loss = running_loss / (grad_accum * 100)
                        logger.info(f"Step {global_step // grad_accum}, Loss: {avg_loss:.4f}")
                        self.metrics['cpt_loss'].append(avg_loss)
                        running_loss = 0.0
            
            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
            

            if (global_step // grad_accum) >= 10000: # After 10000 steps, we should exit the loop
                logger.info(f"Reached 10000 steps, exiting CPT loop")
                #save the CPT checkpoint
                checkpoint_path = self.output_dir / f'cpt_checkpoint_epoch_{epoch + 1}'
                self.student_model.save_pretrained(checkpoint_path)
                self.student_tokenizer.save_pretrained(checkpoint_path)
                logger.info(f"Saved CPT checkpoint: {checkpoint_path}")
                
                break
            
        
        logger.info("CPT Stage completed!")
    
    def run_sft_stage(self, seed_tasks: Dataset, corpora: Dict[str, Dataset]):
        """
        Stage 2: Supervised Fine-Tuning (SFT)
        
        Train on instruction data with teacher soft labels.
        Data mix: 50% native Swedish QA, 30% translated, 20% parallel translation
        
        ANTI-FORGETTING MEASURES:
        - Replay buffer with pretraining data (10-20% of each batch)
        - Lower learning rate with warmup
        - KL regularization to reference model
        - Diverse task types (QA, MMLU-style, reasoning, translation)
        """
        logger.info("=" * 50)
        logger.info("STAGE 2: SUPERVISED FINE-TUNING (SFT)")
        logger.info("=" * 50)
        
        sft_config = self.config.get('stage2_sft', {})
        epochs = sft_config.get('epochs', 3)
        batch_size = sft_config.get('batch_size', 8)
        max_length = sft_config.get('max_length', 512)
        grad_accum = sft_config.get('gradient_accumulation_steps', 8)
        lr = float(sft_config.get('learning_rate', 5e-5))
        warmup_ratio = float(sft_config.get('warmup_ratio', 0.1))
        
        # Distillation parameters
        temperature = float(self.config.get('temperature', 3.0))
        alpha_ce = float(self.config.get('alpha_ce', 0.5))  # Increased CE weight
        alpha_kl = float(self.config.get('alpha_kl', 0.5))  # Balanced KL weight
        
        # Anti-forgetting parameters
        replay_ratio = float(sft_config.get('replay_ratio', 0.10))  # 10% replay data
        kl_reg_weight = float(sft_config.get('kl_regularization_weight', 0.05))
        kl_reg_interval = int(sft_config.get('kl_reg_interval', 4))  # Compute KL every N steps
        
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Warmup ratio: {warmup_ratio}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Alpha CE: {alpha_ce}, Alpha KL: {alpha_kl}")
        logger.info(f"Replay ratio: {replay_ratio}")
        logger.info(f"KL regularization weight: {kl_reg_weight}")
        logger.info(f"KL regularization interval: every {kl_reg_interval} steps")
        
        # Prepare instruction data (EXPANDED with diverse tasks)
        instruction_data = self._prepare_instruction_data_diverse(seed_tasks, corpora)
        
        if not instruction_data:
            logger.error("No instruction data prepared for SFT!")
            return
        
        logger.info(f"Total SFT samples: {len(instruction_data)}")
        
        # Prepare replay buffer (pretraining data for anti-forgetting)
        # Reduced size to save memory
        replay_buffer = self._prepare_replay_buffer(corpora, max_samples=2000)
        logger.info(f"Replay buffer size: {len(replay_buffer)}")
        
        # Clear CUDA cache after data preparation
        torch.cuda.empty_cache()
        
        # Create reference model for KL regularization (copy of current student)
        if kl_reg_weight > 0:
            logger.info("Creating reference model for KL regularization...")
            self.reference_model = copy.deepcopy(self.student_model)
            self.reference_model.eval()
            for p in self.reference_model.parameters():
                p.requires_grad = False
        
        # Calculate total steps for scheduler
        total_steps = (len(instruction_data) // batch_size) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        # Setup optimizer with lower LR
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        # Learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        # Training loop
        self.student_model.train()
        global_step = 0
        
        for epoch in range(epochs):
            logger.info(f"SFT Epoch {epoch + 1}/{epochs}")
            random.shuffle(instruction_data)
            
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_ce = 0.0
            epoch_replay = 0.0
            n_batches = 0
            
            for i in range(0, len(instruction_data), batch_size):
                batch = instruction_data[i:i + batch_size]
                
                # Compute SFT loss (CE only, no teacher KL for stability)
                batch_loss, kl_loss, ce_loss = self._sft_step_stable(
                    batch, max_length, temperature, alpha_ce, alpha_kl
                )
                
                if batch_loss is None:
                    continue
                
                total_batch_loss = batch_loss
                
                # Add replay loss (anti-forgetting) - skip some batches to save memory
                replay_loss_val = 0.0
                if replay_buffer and replay_ratio > 0 and global_step % 2 == 0:  # Every other step
                    n_replay = max(1, int(batch_size * replay_ratio))
                    replay_samples = random.sample(replay_buffer, min(n_replay, len(replay_buffer)))
                    replay_loss = self._compute_replay_loss(replay_samples, max_length)
                    if replay_loss is not None:
                        total_batch_loss = total_batch_loss + replay_ratio * replay_loss
                        replay_loss_val = replay_loss.item()
                        del replay_loss
                
                # Add KL regularization to reference model (only every kl_reg_interval steps, skip first batch)
                if (self.reference_model is not None and kl_reg_weight > 0 and 
                    global_step > 0 and global_step % kl_reg_interval == 0):
                    # Clear cache before KL computation
                    torch.cuda.empty_cache()
                    kl_reg_loss = self._compute_kl_regularization(batch, max_length)
                    if kl_reg_loss is not None:
                        total_batch_loss = total_batch_loss + kl_reg_weight * kl_reg_loss
                        del kl_reg_loss
                
                (total_batch_loss / grad_accum).backward()
                
                # Clean up to free memory
                del total_batch_loss
                torch.cuda.empty_cache()
                
                epoch_loss += batch_loss.item()
                epoch_kl += kl_loss
                epoch_ce += ce_loss
                epoch_replay += replay_loss_val
                n_batches += 1
                global_step += 1
                
                # Optimizer step
                if global_step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    if (global_step // grad_accum) % 10 == 0:
                        avg_loss = epoch_loss / max(n_batches, 1)
                        avg_kl = epoch_kl / max(n_batches, 1)
                        avg_ce = epoch_ce / max(n_batches, 1)
                        avg_replay = epoch_replay / max(n_batches, 1)
                        current_lr = scheduler.get_last_lr()[0]
                        logger.info(f"Step {global_step // grad_accum}, Loss: {avg_loss:.4f}, "
                                  f"KL: {avg_kl:.4f}, CE: {avg_ce:.4f}, Replay: {avg_replay:.4f}, LR: {current_lr:.2e}")
                        self.metrics['sft_loss'].append(avg_loss)
                        self.metrics['kl_loss'].append(avg_kl)
                        self.metrics['ce_loss'].append(avg_ce)
                        self.metrics['replay_loss'].append(avg_replay)
            
            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = self.output_dir / f'sft_checkpoint_epoch_{epoch + 1}'
            self.student_model.save_pretrained(checkpoint_path)
            self.student_tokenizer.save_pretrained(checkpoint_path)
            logger.info(f"Saved SFT checkpoint: {checkpoint_path}")
        
        # Cleanup
        if self.reference_model is not None:
            del self.reference_model
            self.reference_model = None
        torch.cuda.empty_cache()

        logger.info("SFT Stage completed!")
    
    def _prepare_replay_buffer(self, corpora: Dict[str, Dataset], max_samples: int = 5000) -> List[str]:
        """
        Prepare replay buffer with pretraining data for anti-forgetting.
        Mix of Swedish and English texts.
        """
        replay_data = []
        
        # Get Swedish texts
        swedish_texts = _extract_texts_from_corpus(corpora.get('sv'))
        if swedish_texts:
            n_swedish = int(max_samples * 0.7)  # 70% Swedish
            selected = random.sample(swedish_texts, min(n_swedish, len(swedish_texts)))
            replay_data.extend(selected)
        
        # Get English texts
        english_texts = _extract_texts_from_corpus(corpora.get('en'))
        if english_texts:
            n_english = int(max_samples * 0.3)  # 30% English
            selected = random.sample(english_texts, min(n_english, len(english_texts)))
            replay_data.extend(selected)
        
        random.shuffle(replay_data)
        return replay_data[:max_samples]
    
    def _compute_replay_loss(self, texts: List[str], max_length: int) -> Optional[torch.Tensor]:
        """Compute language modeling loss on replay buffer (anti-forgetting)"""
        if not texts:
            return None
        
        inputs = self.student_tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        outputs = self.student_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['input_ids'],
            use_cache=False
        )
        
        return outputs.loss
    
    def _compute_kl_regularization(self, batch: List[Dict], max_length: int) -> Optional[torch.Tensor]:
        """
        Compute KL divergence between current model and reference model.
        This prevents the model from drifting too far from its original distribution.
        
        MEMORY OPTIMIZED: Only compute on first sample, use reduced sequence length.
        """
        if self.reference_model is None:
            return None
        
        # Use only first sample to reduce memory
        sample = batch[0]
        full_text = f"{sample['prompt']}\n{sample['response']}"
        
        # Use shorter max_length for KL computation
        kl_max_length = min(max_length, 256)
        
        inputs = self.student_tokenizer(
            [full_text],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=kl_max_length
        ).to(self.device)
        
        # Get reference model logits FIRST (no gradients needed)
        with torch.no_grad():
            ref_outputs = self.reference_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                use_cache=False
            )
            ref_probs = F.softmax(ref_outputs.logits, dim=-1)
            # Delete ref_outputs to free memory
            del ref_outputs
        
        # Get current model logits (needs gradients)
        current_outputs = self.student_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            use_cache=False
        )
        current_log_probs = F.log_softmax(current_outputs.logits, dim=-1)
        
        # KL divergence (reference is target, current is prediction)
        kl_loss = F.kl_div(
            current_log_probs,
            ref_probs,
            reduction='batchmean'
        )
        
        # Clean up
        del inputs, ref_probs, current_log_probs, current_outputs
        
        return kl_loss

    def _prepare_instruction_data_diverse(
        self,
        seed_tasks: Dataset,
        corpora: Dict[str, Dataset]
    ) -> List[Dict[str, str]]:
        """
        Prepare DIVERSE instruction data for SFT to prevent catastrophic forgetting.
        
        Includes:
        - Native Swedish QA/reasoning
        - Translation tasks
        - MMLU-style multiple choice (Swedish)
        - Reading comprehension
        - General knowledge questions
        - Summarization
        
        OPTIMIZED: Uses batched generation for teacher outputs.
        """
        instruction_data = []
        logger.info("Preparing diverse instruction data (anti-forgetting)...")
        
        # --- Group 1: Native Swedish QA from seed tasks ---
        native_tasks = []
        native_prompts = []
        
        for task in seed_tasks:
            task_type = task.get('task_type', '')
            if task_type == 'qa' and task.get('language') == 'sv':
                prompt = f"Kontext: {task.get('context', '')}\n\nFråga: {task.get('question', '')}"
                native_tasks.append({'task': task, 'prompt': prompt, 'type': 'native_qa', 'max_tokens': 160})
                native_prompts.append(prompt)
            
            elif task_type == 'reasoning' and task.get('language') == 'sv':
                prompt = f"Premiss: {task.get('premise', '')}\n\nVad är slutsatsen?"
                native_tasks.append({'task': task, 'prompt': prompt, 'type': 'native_reasoning', 'max_tokens': 100})
                native_prompts.append(prompt)

        # Batched generation for Native QA
        if native_prompts:
            logger.info(f"Generating answers for {len(native_prompts)} native tasks...")
            responses = self._generate_teacher_outputs_batched(native_prompts, max_new_tokens=160)
            
            for i, task_info in enumerate(native_tasks):
                teacher_response = responses[i]
                original_task = task_info['task']
                fallback = original_task.get('answer') or original_task.get('conclusion') or ''
                
                instruction_data.append({
                    'id': f"native_{i}",
                    'prompt': task_info['prompt'],
                    'response': teacher_response or fallback,
                    'type': task_info['type']
                })

        # --- Group 2: Translation tasks ---
        trans_tasks = []
        trans_prompts = []
        
        for task in seed_tasks:
            if task.get('task_type') == 'translation':
                source_lang = task.get('source_lang', 'en')
                target_lang = task.get('target_lang', 'sv')
                
                if target_lang == 'sv':
                    prompt = f"Översätt till svenska:\n{task.get('source_text', '')}"
                else:
                    prompt = f"Translate to English:\n{task.get('source_text', '')}"
                
                trans_tasks.append({'task': task, 'prompt': prompt})
                trans_prompts.append(prompt)

        if trans_prompts:
            logger.info(f"Generating translations for {len(trans_prompts)} tasks...")
            responses = self._generate_teacher_outputs_batched(trans_prompts, max_new_tokens=256)
            
            for i, task_info in enumerate(trans_tasks):
                teacher_response = responses[i]
                original_task = task_info['task']
                fallback = original_task.get('target_text', '')
                
                instruction_data.append({
                    'id': f"trans_{i}",
                    'prompt': task_info['prompt'],
                    'response': teacher_response or fallback,
                    'type': 'translation'
                })

        # --- Group 3: Generated Swedish QA from monolingual data ---
        swedish_texts = _extract_texts_from_corpus(corpora.get('sv'))
        n_additional = min(2000, len(swedish_texts))  # Reduced from 5000 to save memory/time
        
        if swedish_texts:
            logger.info(f"Generating {n_additional} synthetic QA pairs...")
            selected_texts = random.sample(swedish_texts, n_additional)
            
            # Step 3a: Generate Questions
            question_prompts = []
            valid_indices = []
            
            for i, text in enumerate(selected_texts):
                if len(text) > 100:
                    prompt = f"Skapa en fråga om följande text:\n\n{text[:500]}"
                    question_prompts.append(prompt)
                    valid_indices.append(i)
            
            if question_prompts:
                questions = self._generate_teacher_outputs_batched(question_prompts, max_new_tokens=64)
                
                # Step 3b: Generate Answers
                answer_prompts = []
                valid_qa_indices = []
                
                for i, question in enumerate(questions):
                    if question:
                        orig_text_idx = valid_indices[i]
                        text = selected_texts[orig_text_idx]
                        prompt = f"Kontext: {text[:500]}\n\nFråga: {question}"
                        answer_prompts.append(prompt)
                        valid_qa_indices.append(i)
                
                if answer_prompts:
                    answers = self._generate_teacher_outputs_batched(answer_prompts, max_new_tokens=160)
                    
                    for i, answer in enumerate(answers):
                        if answer:
                            prompt = answer_prompts[i]
                            instruction_data.append({
                                'id': f"gen_qa_{i}",
                                'prompt': prompt,
                                'response': answer,
                                'type': 'generated_qa'
                            })

        # --- Group 4: MMLU-style Multiple Choice (Swedish) ---
        # Generate diverse knowledge questions to maintain reasoning capability
        mmlu_prompts = self._generate_mmlu_style_prompts(swedish_texts[:1000] if swedish_texts else [])
        if mmlu_prompts:
            logger.info(f"Generating {len(mmlu_prompts)} MMLU-style questions...")
            mmlu_responses = self._generate_teacher_outputs_batched(mmlu_prompts, max_new_tokens=100)
            
            for i, (prompt, response) in enumerate(zip(mmlu_prompts, mmlu_responses)):
                if response:
                    instruction_data.append({
                        'id': f"mmlu_{i}",
                        'prompt': prompt,
                        'response': response,
                        'type': 'mmlu_style'
                    })

        # --- Group 5: Summarization tasks ---
        if swedish_texts:
            n_summary = min(500, len(swedish_texts))  # Reduced from 1000
            logger.info(f"Generating {n_summary} summarization tasks...")
            summary_texts = random.sample(swedish_texts, n_summary)
            summary_prompts = []
            
            for text in summary_texts:
                if len(text) > 200:
                    prompt = f"Sammanfatta följande text i 2-3 meningar:\n\n{text[:800]}"
                    summary_prompts.append(prompt)
            
            if summary_prompts:
                summaries = self._generate_teacher_outputs_batched(summary_prompts, max_new_tokens=150)
                
                for i, (prompt, summary) in enumerate(zip(summary_prompts, summaries)):
                    if summary:
                        instruction_data.append({
                            'id': f"summary_{i}",
                            'prompt': prompt,
                            'response': summary,
                            'type': 'summarization'
                        })

        # --- Group 6: Simple instruction following (maintains format) ---
        simple_instructions = self._generate_simple_instructions()
        if simple_instructions:
            logger.info(f"Generating {len(simple_instructions)} simple instruction responses...")
            simple_prompts = [item['prompt'] for item in simple_instructions]
            simple_responses = self._generate_teacher_outputs_batched(simple_prompts, max_new_tokens=200)
            
            for i, (item, response) in enumerate(zip(simple_instructions, simple_responses)):
                if response:
                    instruction_data.append({
                        'id': f"simple_{i}",
                        'prompt': item['prompt'],
                        'response': response,
                        'type': 'simple_instruction'
                    })
        
        logger.info(f"Prepared {len(instruction_data)} diverse instruction samples")
        
        # Log distribution
        type_counts = {}
        for item in instruction_data:
            t = item.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        logger.info(f"Task distribution: {type_counts}")
        
        return instruction_data
    
    def _generate_mmlu_style_prompts(self, texts: List[str]) -> List[str]:
        """Generate MMLU-style multiple choice questions in Swedish"""
        prompts = []
        
        # Template questions covering various domains
        templates = [
            "Fråga: Vad är huvudstaden i Sverige?\nA) Oslo\nB) Stockholm\nC) Köpenhamn\nD) Helsingfors\n\nSvara med endast bokstaven (A, B, C eller D).",
            "Fråga: Vilket år grundades Sverige som nation?\nA) 1523\nB) 1809\nC) 1905\nD) 1397\n\nSvara med endast bokstaven (A, B, C eller D).",
            "Fråga: Vilken svensk uppfinnare uppfann dynamiten?\nA) Carl von Linné\nB) Alfred Nobel\nC) Anders Celsius\nD) Emanuel Swedenborg\n\nSvara med endast bokstaven (A, B, C eller D).",
            "Fråga: Vad är 15 + 27?\nA) 32\nB) 42\nC) 52\nD) 62\n\nSvara med endast bokstaven (A, B, C eller D).",
            "Fråga: Vilket av följande är ett däggdjur?\nA) Lax\nB) Krokodil\nC) Delfin\nD) Örn\n\nSvara med endast bokstaven (A, B, C eller D).",
        ]
        
        # Add templates
        prompts.extend(templates)
        
        # Generate context-based questions from texts
        for text in texts[:200]:  # Limit to avoid too many
            if len(text) > 100:
                prompt = f"Baserat på följande text, vilket påstående är korrekt?\n\nText: {text[:300]}\n\nA) Texten handlar om vetenskap\nB) Texten handlar om historia\nC) Texten handlar om kultur\nD) Texten handlar om ekonomi\n\nSvara med endast bokstaven (A, B, C eller D) och förklara kort varför."
                prompts.append(prompt)
        
        return prompts
    
    def _generate_simple_instructions(self) -> List[Dict[str, str]]:
        """Generate simple instruction-following tasks"""
        instructions = [
            {"prompt": "Skriv en kort hälsning på svenska."},
            {"prompt": "Förklara vad en dator är för ett barn."},
            {"prompt": "Lista tre svenska städer."},
            {"prompt": "Vad är skillnaden mellan en sjö och ett hav?"},
            {"prompt": "Beskriv hur man kokar ett ägg."},
            {"prompt": "Vad betyder ordet 'demokrati'?"},
            {"prompt": "Nämn tre svenska traditioner."},
            {"prompt": "Förklara varför himlen är blå."},
            {"prompt": "Vad är fotosyntesen?"},
            {"prompt": "Beskriv en typisk svensk frukost."},
            {"prompt": "What is the capital of France?"},  # English for diversity
            {"prompt": "Explain what machine learning is in simple terms."},
            {"prompt": "List three programming languages."},
            {"prompt": "What is the difference between a planet and a star?"},
            {"prompt": "Describe how a bicycle works."},
        ]
        return instructions * 10  # Repeat for more samples
    
    def _sft_step_stable(
        self,
        batch: List[Dict[str, str]],
        max_length: int,
        temperature: float,
        alpha_ce: float,
        alpha_kl: float
    ) -> Tuple[Optional[torch.Tensor], float, float]:
        """
        Stable SFT training step.
        
        Key changes for stability:
        - Proper handling of vocab size mismatch
        - Clamping of logits to prevent inf/nan
        - Proper KL scaling
        """
        
        # Prepare batch data
        prompts = [item['prompt'] for item in batch]
        responses = [item['response'] for item in batch]
        full_texts = [f"{p}\n{r}" for p, r in zip(prompts, responses)]
        
        # Tokenize for student
        student_inputs = self.student_tokenizer(
            full_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Create labels (mask prompt tokens)
        labels = student_inputs['input_ids'].clone()
        for i, prompt in enumerate(prompts):
            prompt_tokens = self.student_tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=max_length
            )['input_ids']
            prompt_len = prompt_tokens.shape[1]
            labels[i, :prompt_len] = -100
        
        # Student forward pass
        student_outputs = self.student_model(
            input_ids=student_inputs['input_ids'],
            attention_mask=student_inputs['attention_mask'],
            labels=labels,
            use_cache=False
        )
        ce_loss = student_outputs.loss
        student_logits = student_outputs.logits
        
        # Teacher forward pass
        teacher_device = self._get_teacher_device()
        teacher_inputs = self.teacher_tokenizer(
            full_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(teacher_device)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_inputs['input_ids'],
                attention_mask=teacher_inputs['attention_mask'],
                use_cache=False
            )
            teacher_logits = teacher_outputs.logits.to(self.device)
        
        # Align sequence lengths
        min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_seq_len, :]
        teacher_logits = teacher_logits[:, :min_seq_len, :]
        
        # Align vocab sizes (CRITICAL for different tokenizers)
        min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits = student_logits[:, :, :min_vocab]
        teacher_logits = teacher_logits[:, :, :min_vocab]
        
        # Clamp logits to prevent numerical issues
        student_logits = torch.clamp(student_logits, min=-100, max=100)
        teacher_logits = torch.clamp(teacher_logits, min=-100, max=100)
        
        # Create mask for valid (non-padding) positions
        # Only compute KL on positions where we have valid labels
        valid_mask = (labels[:, :min_seq_len] != -100).unsqueeze(-1).float()
        
        # KL divergence loss with temperature (only on valid positions)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # Element-wise KL
        kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction='none')
        
        # Apply mask and compute mean
        kl_masked = kl_per_token * valid_mask
        n_valid = valid_mask.sum() + 1e-8
        kl_loss = (kl_masked.sum() / n_valid) * (temperature ** 2)
        
        # Clamp KL loss to prevent explosion
        kl_loss = torch.clamp(kl_loss, max=100.0)
        
        # Combined loss
        total_loss = alpha_ce * ce_loss + alpha_kl * kl_loss
        
        # Store values before cleanup
        kl_val = float(kl_loss.item())
        ce_val = float(ce_loss.item())
        
        # Clean up intermediate tensors
        del student_inputs, teacher_inputs, labels, student_logits, teacher_logits
        del student_log_probs, teacher_probs, kl_per_token, kl_masked, valid_mask
        
        return total_loss, kl_val, ce_val


class FrodiTrainerSwedish:
    """
    FRÓÐI Trainer for Swedish - GRPO with EuroEval + Adaptive Self-Play

    Tasks sourced from EuroEval Swedish training sets with adaptive self-play
    from FineWeb corpus text.  Unmastered types get 3x sampling weight.
    Self-play unlocks at 70% overall accuracy for mastered eligible types.
    """

    def __init__(
        self,
        student_model: AutoModelForCausalLM,
        student_tokenizer: AutoTokenizer,
        teacher_model: AutoModelForCausalLM,  # This is the judge model for GRPO
        teacher_tokenizer: AutoTokenizer,
        config: Dict[str, Any],
        device: str = "cuda",
        euroeval_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.config = config

        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.teacher_model = teacher_model  # Judge model
        self.teacher_tokenizer = teacher_tokenizer

        # EuroEval training data (primary task source)
        self.euroeval_data = euroeval_data or {}
        self._euroeval_indices = {}  # Track sampling position per dataset
        for ds_name in self.euroeval_data:
            self._euroeval_indices[ds_name] = 0

        # Per-task accuracy tracking for mastery-weighted EuroEval sampling
        self.task_accuracy = {}  # task_type -> {"correct": int, "total": int}
        for ds_config in EUROEVAL_DATASETS.values():
            tt = ds_config["task_type"]
            self.task_accuracy[tt] = {"correct": 0, "total": 0}
        self.mastered_tasks: set = set()   # task types with accuracy >= TASK_MASTERY_THRESHOLD

        # Self-play state
        self.selfplay_enabled = False
        self.selfplay_ratio = 0.0
        self.selfplay_eligible_mastered: set = set()
        self._monolingual_texts: List[str] = []

        # Separate accuracy tracking for self-play tasks
        self.selfplay_accuracy: Dict[str, Dict[str, int]] = {}
        for tt in SELFPLAY_ELIGIBLE_TYPES:
            self.selfplay_accuracy[tt] = {"correct": 0, "total": 0}

        # Reward tracking by source (rolling window for paper plots)
        self.reward_tracker: Dict[str, Dict[str, Any]] = {
            "euroeval": {"count": 0, "recent": []},
            "selfplay": {"count": 0, "recent": []},
        }

        # Ensure pad tokens
        self._ensure_pad_token(self.student_tokenizer)
        self._ensure_pad_token(self.teacher_tokenizer)

        # Build Swedish system prompts
        self.default_system_prompts = self._build_swedish_system_prompts()
        
        # Task buffer
        self.task_buffer = TaskBuffer(
            max_size=config.get('curriculum', {}).get('max_buffer_size', 1000),
            target_success_rate=config.get('curriculum', {}).get('target_success_rate', 0.7)
        )
        
        # Reward weights
        rl_config = config.get('rl', {})
        reward_weights = copy.deepcopy(rl_config.get('reward_weights', {})) or {}
        self.weight_scheduler = AdaptiveWeightSchedulerSimple(reward_weights)
        
        # Initialize Swedish reward model (EuroEval-aware version)
        from ..rl_loop.reward_swedish_euro import TeacherStudentSimpleRewardSwedish
        
        reward_clip_min = float(rl_config.get('reward_clip_min', -1.0))
        reward_clip_max = float(rl_config.get('reward_clip_max', 1.0))
        reward_norm_momentum = float(rl_config.get('reward_norm_momentum', 0.1))
        reward_weights['reward_clip_min'] = reward_clip_min
        reward_weights['reward_clip_max'] = reward_clip_max
        reward_weights['reward_norm_momentum'] = reward_norm_momentum
        
        self.reward_model = TeacherStudentSimpleRewardSwedish(
            student_tokenizer=student_tokenizer,
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            reward_weights=reward_weights,
            device=device,
        )
        
        # Disable KV cache for training (saves memory, required for gradient checkpointing)
        try:
            self.student_model.config.use_cache = False
            self.teacher_model.config.use_cache = False
            if hasattr(self.student_model, 'generation_config'):
                self.student_model.generation_config.use_cache = False
            if hasattr(self.teacher_model, 'generation_config'):
                self.teacher_model.generation_config.use_cache = False
            logger.info("Disabled KV caching for all models")
        except Exception as e:
            logger.warning(f"Could not disable caching: {e}")
        
        # Enable gradient checkpointing to reduce memory (CRITICAL for large batches)
        if config.get('optimization', {}).get('gradient_checkpointing', True):
            try:
                self.student_model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing for student model")
            except Exception as e:
                logger.warning(f"Gradient checkpointing failed: {e}")
        
        # Optimizer
        lr = float(rl_config.get('learning_rate', 5e-6))
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=lr,
            eps=float(config.get('optimization', {}).get('adam_eps', 1e-5))
        )
        
        # Training state
        self.iteration = 0
        self.training_history = []
        self.start_time = None
        
        # Output directories
        self.output_dir = Path(rl_config.get('output_dir', 'outputs/self_play_rl_swedish'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.metrics_dir = self.output_dir / 'metrics'
        
        for dir_path in [self.checkpoint_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Reference model
        self.reference_model = None
        
        # Evaluation and checkpoint tracking
        eval_config = config.get('evaluation', {})
        self.eval_interval = eval_config.get('eval_interval', 100)
        self.euroeval_results_file = self.metrics_dir / 'euroeval_grpo_progress.jsonl'
        
        # Best model tracking (based on training reward, EuroEval runs post-training)
        self.best_training_reward = float('-inf')
        self.best_model_iteration = 0
        self.best_model_dir = self.output_dir / 'best_model'
        self.current_checkpoint_dir = self.output_dir / 'current_checkpoint'
        self.eval_history: List[Dict[str, Any]] = []
        
        # EuroEval configuration (for post-training evaluation)
        self.euroeval_cache_dir = "/proj/berzelius-aiics-real/users/x_anbue/euroeval/.euroeval_cache"
        self.euroeval_container = "/proj/berzelius-aiics-real/users/x_anbue/euroeval.sif"
        
        logger.info("Swedish FRÓÐI GRPO Trainer initialized")
    
    def _ensure_pad_token(self, tokenizer: AutoTokenizer):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
    
    def _build_swedish_system_prompts(self) -> Dict[str, Dict[str, str]]:
        """Build Swedish-specific system prompts for all task types"""
        prompts_cfg = self.config.get('prompts', {})

        return {
            # EuroEval task prompts
            'sentiment': {
                'student': (
                    "Du klassificerar sentiment i svenska texter. "
                    "Svara med exakt ett ord: positiv, negativ, neutral eller blandad."
                ),
            },
            'acceptability': {
                'student': (
                    "Du bedömer grammatisk korrekthet i svenska meningar. "
                    "Svara med exakt ett ord: korrekt eller inkorrekt."
                ),
            },
            'ner': {
                'student': (
                    "Du identifierar namngivna entiteter i svenska texter. "
                    "Lista varje entitet med dess typ (PER, LOC, ORG). "
                    "Om inga entiteter finns, skriv 'Inga entiteter'."
                ),
            },
            'reading_comprehension': {
                'student': (
                    "Du är en precis fråga-svar-assistent. Besvara frågan "
                    "kort och korrekt baserat på den givna kontexten."
                ),
            },
            'commonsense': {
                'student': (
                    "Du löser frågor om sunt förnuft. Svara med enbart "
                    "bokstaven för rätt alternativ (a, b, c eller d)."
                ),
            },
            'knowledge': {
                'student': (
                    "Du svarar på kunskapsfrågor. Svara med enbart "
                    "bokstaven för rätt alternativ (a, b, c eller d)."
                ),
            },
        }
    
    def run_euroeval_evaluation(self, model_path: str, iteration: int) -> Optional[Dict[str, Any]]:
        """
        Run EuroEval benchmark on a model checkpoint for Swedish validation.
        
        NOTE: This method is designed to be called from post-training evaluation,
        not during training (since GRPO runs in env.sif but EuroEval needs euroeval.sif).
        During training, we only save checkpoints and register them for later evaluation.
        
        Args:
            model_path: Path to the model checkpoint
            iteration: Current training iteration
            
        Returns:
            Dictionary with evaluation results or None if evaluation failed
        """
        logger.info(f"Running EuroEval Swedish validation at iteration {iteration}")
        
        # Create a unique results file for this evaluation
        eval_results_dir = self.metrics_dir / 'euroeval_runs'
        eval_results_dir.mkdir(exist_ok=True)
        results_file = eval_results_dir / f'euroeval_iter_{iteration}.jsonl'
        
        try:
            # Use Python API if euroeval is available (only works if running in euroeval container)
            from euroeval import Benchmarker
            
            benchmarker = Benchmarker(
                results_path=str(results_file),
                cache_dir=self.euroeval_cache_dir
            )
            
            # Run benchmark on Swedish validation split with reduced iterations for speed
            results = benchmarker.benchmark(
                model=model_path,
                language="sv",
                device="cuda",
                evaluate_val_split=True,
                trust_remote_code=True,
                save_results=True,
                verbose=False,
                num_iterations=1,  # Single iteration for speed during training
                batch_size=16
            )
            
            logger.info(f"EuroEval completed for iteration {iteration}")
            return self._parse_euroeval_results(results_file, iteration)
                
        except ImportError:
            logger.warning(
                f"EuroEval not available in current environment. "
                f"Checkpoint saved at {model_path} - run post-training evaluation."
            )
            return None
        except Exception as e:
            logger.error(f"EuroEval evaluation failed at iteration {iteration}: {e}")
            return None
    
    def register_checkpoint_for_eval(self, checkpoint_path: str, iteration: int):
        """
        Register a checkpoint for EuroEval evaluation.
        
        If submit_eval_jobs is enabled, this will submit an sbatch job to run
        EuroEval on the checkpoint in parallel using euroeval.sif container.
        Otherwise, it just logs the checkpoint for post-training evaluation.
        """
        eval_queue_file = self.metrics_dir / 'pending_eval_checkpoints.jsonl'
        
        entry = {
            'checkpoint_path': str(checkpoint_path),
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'evaluated': False
        }
        
        with open(eval_queue_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Registered checkpoint for evaluation: {checkpoint_path} (iteration {iteration})")
        
        # Submit evaluation job if enabled
        submit_eval_jobs = self.config.get('evaluation', {}).get('submit_eval_jobs', True)
        if submit_eval_jobs:
            self._submit_euroeval_job(checkpoint_path, iteration)
    
    def _submit_euroeval_job(self, checkpoint_path: str, iteration: int):
        """
        Submit an sbatch job to run EuroEval on a checkpoint.
        
        This allows parallel evaluation during training by submitting a separate
        job that uses euroeval.sif container while training continues with env.sif.
        
        IMPORTANT: To avoid race conditions where the rolling checkpoint gets
        overwritten before evaluation starts, we copy the checkpoint to an
        iteration-specific directory first.
        """
        eval_script = Path("/home/x_anbue/frodi/scripts/run_single_euroeval.sh")
        
        if not eval_script.exists():
            logger.warning(f"EuroEval script not found: {eval_script}")
            return
        
        # Create iteration-specific checkpoint directory to avoid race conditions
        # The rolling checkpoint may be overwritten before the eval job runs
        eval_checkpoint_dir = self.output_dir / 'eval_checkpoints' / f'iter_{iteration}'
        
        try:
            # Copy checkpoint to iteration-specific directory
            if eval_checkpoint_dir.exists():
                shutil.rmtree(eval_checkpoint_dir)
            
            eval_checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(checkpoint_path, eval_checkpoint_dir)
            logger.info(f"Copied checkpoint to {eval_checkpoint_dir} for evaluation")
            
            # Use the copied checkpoint path for evaluation
            eval_checkpoint_path = str(eval_checkpoint_dir)
            
        except Exception as e:
            logger.warning(f"Failed to copy checkpoint for evaluation: {e}")
            # Fall back to original path (may cause race condition)
            eval_checkpoint_path = checkpoint_path
        
        try:
            # Build sbatch command with environment variables
            cmd = [
                'sbatch',
                f'--export=CHECKPOINT_PATH={eval_checkpoint_path},ITERATION={iteration}',
                f'--job-name=euroeval-iter{iteration}',
                str(eval_script)
            ]
            
            # Also pass HF_TOKEN if available
            hf_token = os.environ.get('HF_TOKEN', '')
            if hf_token:
                cmd[1] = f'--export=CHECKPOINT_PATH={eval_checkpoint_path},ITERATION={iteration},HF_TOKEN={hf_token}'
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Extract job ID from sbatch output (format: "Submitted batch job 12345")
                job_id = result.stdout.strip().split()[-1] if result.stdout else 'unknown'
                logger.info(f"Submitted EuroEval job {job_id} for iteration {iteration}")
                
                # Log the submission
                job_log_file = self.metrics_dir / 'submitted_eval_jobs.jsonl'
                job_entry = {
                    'job_id': job_id,
                    'checkpoint_path': eval_checkpoint_path,
                    'iteration': iteration,
                    'submitted_at': datetime.now().isoformat()
                }
                with open(job_log_file, 'a') as f:
                    f.write(json.dumps(job_entry) + '\n')
            else:
                logger.warning(f"Failed to submit EuroEval job: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning("sbatch command timed out")
        except FileNotFoundError:
            logger.warning("sbatch command not found - not running on SLURM cluster?")
        except Exception as e:
            logger.warning(f"Failed to submit EuroEval job: {e}")
    
    def _parse_euroeval_results(
        self, 
        results_file: Path, 
        iteration: int
    ) -> Optional[Dict[str, Any]]:
        """
        Parse EuroEval results from JSONL file and compute aggregate scores.
        
        Swedish datasets and their primary metrics:
        - swerec: test_mcc (sentiment classification)
        - scala-sv: test_mcc (linguistic acceptability)
        - suc3: test_micro_f1_no_misc (NER)
        - scandiqa-sv: test_f1 (reading comprehension)
        - swedn: test_bertscore (summarization)
        - mmlu-sv: test_mcc (knowledge)
        - hellaswag-sv: test_mcc (reasoning)
        """
        swedish_metrics = {
            'swerec': 'test_mcc',
            'scala-sv': 'test_mcc',
            'suc3': 'test_micro_f1_no_misc',
            'scandiqa-sv': 'test_f1',
            'swedn': 'test_bertscore',
            'mmlu-sv': 'test_mcc',
            'hellaswag-sv': 'test_mcc',
        }
        
        # Try to find results file (euroeval may add suffix)
        possible_files = [
            results_file,
            results_file.parent / 'euroeval_benchmark_results.jsonl'
        ]
        
        results_path = None
        for path in possible_files:
            if path.exists():
                results_path = path
                break
        
        if not results_path:
            logger.warning(f"No EuroEval results file found")
            return None
        
        scores = {}
        
        try:
            with open(results_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    
                    # Filter for Swedish
                    if 'sv' not in record.get('dataset_languages', []):
                        continue
                    
                    dataset = record.get('dataset', '')
                    if dataset in swedish_metrics:
                        metric_key = swedish_metrics[dataset]
                        value = record.get('results', {}).get('total', {}).get(metric_key)
                        if value is not None:
                            scores[dataset] = float(value)
        
        except Exception as e:
            logger.error(f"Failed to parse EuroEval results: {e}")
            return None
        
        if not scores:
            logger.warning("No Swedish benchmark scores found in results")
            return None
        
        # Compute aggregate score (mean of all metrics)
        aggregate_score = np.mean(list(scores.values()))
        
        result = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'aggregate_score': float(aggregate_score),
            'scores': scores,
            'num_datasets': len(scores)
        }
        
        # Log to progress file
        with open(self.euroeval_results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        self.eval_history.append(result)
        
        logger.info(f"EuroEval iteration {iteration}: aggregate={aggregate_score:.2f}, scores={scores}")
        
        return result
    
    
    def _safe_log_softmax(self, logits: torch.Tensor, clamp_min: float = -20.0) -> torch.Tensor:
        """Numerically stable log softmax with clamping (in-place optimization)."""
        # In-place operations to save memory
        logits.nan_to_num_(nan=-10.0, neginf=-10.0, posinf=10.0)
        logits.clamp_(min=-50.0, max=50.0)
        return F.log_softmax(logits, dim=-1)
    
    def _get_active_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Return only active (positive) reward weights."""
        active = {}
        for k, v in (weights or {}).items():
            try:
                fv = float(v)
                if fv > 0.0:
                    active[k] = fv
            except (TypeError, ValueError):
                continue
        return active
    
    def _organize_rewards_by_component(self, rewards: List[Dict]) -> Dict[str, List[float]]:
        """Organize rewards by component for adaptive weight scheduler."""
        organized = {}
        for reward_dict in rewards:
            for component, value in reward_dict.items():
                if component == "reward_method":
                    continue
                if component not in organized:
                    organized[component] = []
                organized[component].append(float(value))
        return organized
    
    def _compute_pretraining_loss(self, corpora: Dict[str, Dataset], use_autocast: bool = True) -> torch.Tensor:
        """Compute language modeling loss on pretraining data to maintain fluency."""
        texts = []
        n_samples = self.config.get('rl', {}).get('pretraining_samples_per_batch', 2)
        
        # Get actual device of student model (handles device_map models)
        student_device = self._get_student_device()
        
        # Efficiently sample from corpora without loading all texts
        sv_corpus = corpora.get('sv')
        en_corpus = corpora.get('en')
        
        # Sample Swedish texts
        if sv_corpus is not None:
            try:
                sv_len = len(sv_corpus)
                if sv_len > 0:
                    for _ in range(min(n_samples, sv_len)):
                        idx = random.randint(0, sv_len - 1)
                        texts.append(sv_corpus[idx]['text'])
            except Exception:
                swedish_texts = _extract_texts_from_corpus(sv_corpus)
                if swedish_texts:
                    texts.extend(random.sample(swedish_texts, min(n_samples, len(swedish_texts))))
        
        # Sample English texts if needed
        if len(texts) < n_samples and en_corpus is not None:
            remaining = n_samples - len(texts)
            try:
                en_len = len(en_corpus)
                if en_len > 0:
                    for _ in range(min(remaining, en_len)):
                        idx = random.randint(0, en_len - 1)
                        texts.append(en_corpus[idx]['text'])
            except Exception:
                english_texts = _extract_texts_from_corpus(en_corpus)
                if english_texts:
                    texts.extend(random.sample(english_texts, min(remaining, len(english_texts))))
        
        if not texts:
            return torch.tensor(0.0, device=student_device, requires_grad=True)
        
        max_length = self.config.get('rl', {}).get('max_length', 512)
        inputs = self.student_tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(student_device)
        
        # Use mixed precision for memory efficiency (same as German trainer)
        self.student_model.train()
        mp_enabled = torch.cuda.is_available() and use_autocast
        if mp_enabled:
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            autocast_ctx = torch.autocast(device_type='cuda', dtype=autocast_dtype, enabled=True)
        else:
            autocast_ctx = nullcontext()
        
        with autocast_ctx:
            outputs = self.student_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['input_ids'],
                use_cache=False
            )
        
        return outputs.loss

    def train_self_play_rl(
        self,
        monolingual_corpora: Optional[Dict[str, Dataset]] = None,
        n_iterations: int = 1000,
        start_iteration: int = 0
    ):
        """Run GRPO RL training with EuroEval + adaptive self-play.

        Tasks come primarily from EuroEval Swedish training sets.  Once
        overall accuracy reaches SELFPLAY_UNLOCK_THRESHOLD and at least one
        eligible type is mastered, self-play tasks are generated from
        monolingual corpus text.

        Args:
            monolingual_corpora: Corpora for pretraining loss AND self-play
                task generation (texts are extracted and capped at
                MAX_CORPUS_TEXTS).
            n_iterations: Total number of iterations to train.
            start_iteration: Iteration to resume from (0 to start fresh).
        """
        # Log EuroEval data availability
        if self.euroeval_data:
            total_samples = sum(len(v) for v in self.euroeval_data.values())
            logger.info(
                f"EuroEval training data: {len(self.euroeval_data)} datasets, "
                f"{total_samples} total samples"
            )
            for ds_name, samples in self.euroeval_data.items():
                ds_cfg = EUROEVAL_DATASETS.get(ds_name, {})
                logger.info(
                    f"  {ds_name}: {len(samples)} samples ({ds_cfg.get('task_type', '?')})"
                )
        else:
            logger.error("No EuroEval data loaded — cannot train without EuroEval tasks")
            raise ValueError("EuroEval training data is required but none was loaded")

        # Extract corpus texts for self-play task generation (capped)
        if monolingual_corpora:
            all_texts: List[str] = []
            for corpus_name, corpus_data in monolingual_corpora.items():
                texts = _extract_texts_from_corpus(corpus_data)
                all_texts.extend(texts)
                if len(all_texts) >= MAX_CORPUS_TEXTS:
                    break
            self._monolingual_texts = all_texts[:MAX_CORPUS_TEXTS]
            logger.info(
                f"Extracted {len(self._monolingual_texts)} corpus texts for "
                f"self-play (cap={MAX_CORPUS_TEXTS})"
            )
        else:
            self._monolingual_texts = []
            logger.info("No monolingual corpora — self-play will be disabled")

        logger.info("Starting Swedish GRPO EuroEval-Aligned Training")
        if start_iteration > 0:
            logger.info(f"Resuming from iteration {start_iteration}")
            logger.info(f"Will train from iteration {start_iteration + 1} to {n_iterations}")
        logger.info(f"EuroEval evaluation every {self.eval_interval} iterations")
        self.start_time = time.time()
        
        # GPU warmup to ensure high power utilization from the start
        # Skip if disabled in config or if we want faster startup
        warmup_enabled = self.config.get('optimization', {}).get('gpu_warmup_enabled', True)
        warmup_duration = self.config.get('optimization', {}).get('gpu_warmup_duration', 30)  # Reduced default
        
        if warmup_enabled:
            logger.info("Performing GPU warmup to ensure high power utilization...")
            try:
                self._gpu_warmup(duration_seconds=warmup_duration)
            except RuntimeError as e:
                logger.error(f"GPU warmup failed critically: {e}")
                logger.info("Attempting to continue without warmup...")
                # Try to reset CUDA state
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            logger.info("GPU warmup disabled, skipping...")
        
        # Ensure CUDA is in clean state before creating reference model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        rl_config = self.config.get('rl', {})
        
        # Create reference model (same as German trainer)
        logger.info("Creating reference model...")
        self.reference_model = self._create_reference_model()
        logger.info("Reference model created successfully")
        batch_size = rl_config.get('batch_size', 8)
        grad_accum = rl_config.get('gradient_accumulation_steps', 4)
        grpo_group_size = rl_config.get('grpo_group_size', 4)
        log_interval = self.config.get('logging', {}).get('log_interval', 5)  # Log every N iterations
        
        # KL control parameters - now with adaptive control
        self.current_kl_coef = float(rl_config.get('init_kl_coef', rl_config.get('kl_penalty_coef', 0.08)))
        self.adap_kl_ctrl = bool(rl_config.get('adap_kl_ctrl', True))
        self.kl_target = float(rl_config.get('target', 6.0))
        self.kl_decay = float(rl_config.get('kl_decay', 0.8))
        self.kl_growth = float(rl_config.get('kl_growth', 1.5))
        self.kl_coef_min = float(rl_config.get('kl_coef_min', 0.001))
        self.kl_coef_max = float(rl_config.get('kl_coef_max', 0.5))
        self.ref_update_interval = int(rl_config.get('ref_update_interval', 50))  # Update reference less frequently
        
        # Track KL history for early warning
        self.kl_history = []
        self.negative_kl_warnings = 0
        
        # Gradient spike detection (German trainer approach)
        self.grad_spike_counter = 0
        self.grad_spike_threshold = float(self.config.get('optimization', {}).get('grad_spike_threshold', 10.0))
        self.grad_spike_up = int(self.config.get('optimization', {}).get('grad_spike_up', 3))
        self.grad_spike_down = int(self.config.get('optimization', {}).get('grad_spike_down', 1))
        
        # Mixed precision control from config (CRITICAL for memory efficiency)
        self.use_mixed_precision = bool(self.config.get('optimization', {}).get('mixed_precision', True))
        
        # Storage-efficient checkpointing:
        # - current_checkpoint: rolling checkpoint (overwritten each save)
        # - best_model: best by training reward (updated when reward improves)
        # - final_model: saved at end of training
        # Post-training: EuroEval runs only on best_model and final_model
        
        self.current_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        
        accumulated_loss = 0.0
        iteration_start_time = time.time()
        
        # Calculate starting point for training loop
        actual_start = start_iteration + 1  # Resume from next iteration after checkpoint
        remaining_iterations = n_iterations - start_iteration
        
        logger.info(f"Starting training loop: {n_iterations} total iterations, batch_size={batch_size}, grad_accum={grad_accum}")
        if start_iteration > 0:
            logger.info(f"Resuming: starting at iteration {actual_start}, {remaining_iterations} iterations remaining")
        
        for iteration in range(actual_start, n_iterations + 1):
            self.iteration = iteration
            iter_step_start = time.time()
            
            # Log iteration start for every iteration (for debugging hangs)
            if iteration <= 5 or iteration % log_interval == 0:
                logger.info(f"=== Iteration {iteration} starting ===")
            
            # PERIODIC CHECKPOINT SAVING (at START of iteration, before any processing)
            # This ensures checkpoints are saved even if the current iteration has gradient spikes
            # or other issues that cause `continue`. The checkpoint represents the model state
            # from all previous completed iterations.
            if iteration % self.eval_interval == 0 and iteration > 0:
                logger.info(f"=== Checkpoint update at iteration {iteration} ===")

                # Update rolling "current" checkpoint (overwrites previous)
                self._save_current_checkpoint(iteration)

                # Save persistent checkpoint (kept for post-hoc analysis)
                self._save_persistent_checkpoint(iteration)

                # Check if this is the best model by training reward (use last known reward)
                last_reward = getattr(self, '_last_avg_reward', float('-inf'))
                if last_reward > self.best_training_reward:
                    logger.info(
                        f"New best training reward: {last_reward:.4f} "
                        f"(previous: {self.best_training_reward:.4f})"
                    )
                    self.best_training_reward = last_reward
                    self.best_model_iteration = iteration
                    self._save_best_model(iteration, last_reward)

                # Log metrics for tracking
                self._log_training_metrics(iteration, accumulated_loss, last_reward)

                # Submit EuroEval job for the current checkpoint
                # This runs in parallel using euroeval.sif while training continues
                self.register_checkpoint_for_eval(str(self.current_checkpoint_dir), iteration)

                # Log evaluation status (shows results from parallel eval jobs)
                if iteration > self.eval_interval:  # Skip first checkpoint
                    self.log_evaluation_status()

                logger.info(f"=== Checkpoint update complete ===")
            
            # Populate task buffer
            if iteration <= 5:
                logger.info(f"  [Iter {iteration}] Populating task buffer...")
            self._populate_task_buffer()
            if iteration <= 5:
                logger.info(f"  [Iter {iteration}] Task buffer populated: {self.task_buffer.size} tasks ({time.time() - iter_step_start:.1f}s)")
            
            # Sample tasks
            if iteration <= 5:
                logger.info(f"  [Iter {iteration}] Sampling {batch_size} tasks...")
            task_batch = self.task_buffer.sample_by_priority(
                batch_size,
                temperature=self.config.get('curriculum', {}).get('temperature', 1.0)
            )
            
            if not task_batch:
                logger.warning(f"No tasks available at iteration {iteration}")
                continue
            
            if iteration <= 5:
                logger.info(f"  [Iter {iteration}] Sampled {len(task_batch)} tasks ({time.time() - iter_step_start:.1f}s)")
            
            # Generate solutions and compute rewards
            if iteration <= 5:
                logger.info(f"  [Iter {iteration}] Starting GRPO rollout...")
            rollout_start = time.time()
            solutions, rewards, advantages = self._self_play_rollout(task_batch)
            if iteration <= 5:
                logger.info(f"  [Iter {iteration}] Rollout complete: {len(solutions)} solutions ({time.time() - rollout_start:.1f}s)")
            
            # CRITICAL FIX: Align tasks with all generated solutions
            # Each task generates grpo_group_size solutions, so we need to expand tasks
            num_tasks = len(task_batch)
            tasks_for_loss = task_batch
            try:
                if num_tasks > 0:
                    expected = num_tasks * grpo_group_size
                    if len(solutions) == expected:
                        # Expand tasks to match solutions: [t1, t1, t1, t1, t2, t2, t2, t2, ...]
                        tasks_for_loss = [t for t in task_batch for _ in range(grpo_group_size)]
                    elif len(solutions) % num_tasks == 0:
                        repeats = len(solutions) // num_tasks
                        logger.warning(
                            f"Solutions count {len(solutions)} != expected {expected}; assuming repeats={repeats} per task"
                        )
                        tasks_for_loss = [t for t in task_batch for _ in range(repeats)]
                    else:
                        # Fallback: truncate to aligned length
                        min_len = min(len(solutions), num_tasks)
                        logger.error(
                            f"Cannot align tasks ({num_tasks}) and solutions ({len(solutions)}); truncating to {min_len}"
                        )
                        solutions = solutions[:min_len]
                        advantages = advantages[:min_len]
                        tasks_for_loss = task_batch[:min_len]
            except Exception as e:
                logger.warning(f"Task alignment failed: {e}, using original task_batch")
                tasks_for_loss = task_batch
            
            # CRITICAL: Clear CUDA cache before policy loss to free memory from rollout
            # The rollout uses teacher model which consumes significant VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Compute policy loss with properly aligned tasks (use autocast for memory efficiency)
            # MEMORY FIX: We pass grad_accum so we can backward INSIDE the function to free the graph
            policy_loss = self._compute_policy_loss(
                tasks_for_loss, solutions, advantages, monolingual_corpora,
                use_autocast=self.use_mixed_precision,
                grad_accum_steps=grad_accum
            )
            
            # policy_loss is now a float (already backwarded) or None if skipped
            if policy_loss is None or not math.isfinite(policy_loss):
                logger.warning(f"Non-finite or skipped loss at iteration {iteration}")
                # If we already backwarded partial chunks, we should zero grad? 
                # But hard to undo. Assuming if it returns None/Inf, we zero grad.
                self.optimizer.zero_grad()
                continue
            
            # Loss is already scaled by grad_accum during backward, but for logging we want the raw value?
            # Usually accumulated_loss tracks the scaled loss or raw loss? 
            # Original code: accumulated_loss += float(policy_loss.detach().item())
            # where policy_loss was loss/grad_accum.
            # So we should add the returned value (which corresponds to the total loss contribution for this step).
            # However, _compute_policy_loss returns the MEAN loss across the batch (unscaled).
            # Wait, let's check what it returns. It returns the scalar value of the loss.
            # We should divide by grad_accum for consistency with previous tracking?
            # Original: policy_loss = policy_loss / grad_accum; accumulated_loss += item()
            # So accumulated_loss sums (Loss / Grad_Accum).
            
            accumulated_loss += (policy_loss / grad_accum)
            
            # Backward pass is already done inside _compute_policy_loss to save memory
            # policy_loss.backward() -> REMOVED
            
            # Optimizer step (weights updated every grad_accum iterations)
            # GERMAN TRAINER APPROACH: Gradient spike detection, frequent reference updates
            avg_reward = 0.0
            if iteration % grad_accum == 0:
                # Step 1: Compute gradient norm for spike detection
                grad_norm = None
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), 
                        max_norm=1.0,
                        error_if_nonfinite=False
                    )
                    grad_norm = float(grad_norm.item()) if torch.isfinite(grad_norm) else None
                except Exception:
                    pass
                
                # Step 2: Check for gradient spikes (German trainer approach)
                if grad_norm is not None and grad_norm > self.grad_spike_threshold:
                    self.grad_spike_counter += 1
                    logger.warning(
                        f"Large gradient norm: {grad_norm:.2f} > threshold {self.grad_spike_threshold:.2f} "
                        f"(spike #{self.grad_spike_counter}). Reducing LR and skipping step."
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    self._adjust_learning_rate(factor=0.7, min_lr=5e-7)
                    
                    # If too many consecutive spikes, blend with reference
                    if self.grad_spike_counter >= self.grad_spike_up:
                        blend_alpha = float(self.config.get('optimization', {}).get('reference_blend_alpha', 0.3))
                        self._blend_student_with_reference(alpha=blend_alpha)
                        ref_sync_tau = float(self.config.get('optimization', {}).get('reference_sync_tau', 0.98))
                        self._update_reference_model(tau=ref_sync_tau)
                        self.grad_spike_counter = 0
                    
                    accumulated_loss = 0.0
                    continue
                else:
                    # Decay spike counter when gradients are healthy
                    self.grad_spike_counter = max(0, self.grad_spike_counter - self.grad_spike_down)
                
                # Step 3: Check for NaN/Inf gradients (German trainer approach)
                has_bad_grad = False
                try:
                    for name, param in self.student_model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            logger.error(f"[NaN DETECTED] Gradient in {name} is non-finite. Restoring from reference.")
                            has_bad_grad = True
                            break
                except Exception:
                    pass
                
                if has_bad_grad:
                    self.optimizer.zero_grad(set_to_none=True)
                    self._restore_student_from_reference()
                    accumulated_loss = 0.0
                    continue
                
                # Step 4: Apply optimizer step (gradients already clipped above)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Step 5: Update reference model EVERY optimizer step (German trainer approach)
                # This keeps the reference close to the student, maintaining a meaningful anchor
                ref_sync_tau = float(self.config.get('optimization', {}).get('reference_sync_tau', 0.95))
                self._update_reference_model(tau=ref_sync_tau)
                
                # Calculate weighted average reward using active weights
                current_weights = self.weight_scheduler.get_weights()
                active_weights = self._get_active_weights(current_weights)
                avg_reward = float(np.mean([
                    sum(active_weights.get(k, 0.0) * v for k, v in r.items() if isinstance(v, (int, float))) for r in rewards
                ]))

                accumulated_loss = 0.0

            # Log progress every log_interval iterations (OUTSIDE grad_accum block for consistent logging)
            if iteration % log_interval == 0:
                iter_time = time.time() - iteration_start_time
                iteration_start_time = time.time()

                # Calculate current reward if not already computed
                if 'avg_reward' not in locals() or iteration % grad_accum != 0:
                    current_weights = self.weight_scheduler.get_weights()
                    active_weights = self._get_active_weights(current_weights)
                    avg_reward = float(np.mean([
                        sum(active_weights.get(k, 0.0) * v for k, v in r.items() if isinstance(v, (int, float))) for r in rewards
                    ])) if rewards else 0.0
                
                logger.info(f"Iteration {iteration}/{n_iterations}, Avg Reward: {avg_reward:.4f}, Time: {iter_time:.1f}s")

                # Track last average reward for checkpoint best model selection
                self._last_avg_reward = avg_reward

                avg_kl = getattr(self, 'last_avg_kl', 0.0)

                if hasattr(self, 'last_avg_kl'):
                    logger.info(f"  Avg KL: {self.last_avg_kl:.4f} (coef: {self.current_kl_coef:.4f})")

                    # Track KL history for early warning detection
                    self.kl_history.append(self.last_avg_kl)
                    if len(self.kl_history) > 100:
                        self.kl_history = self.kl_history[-100:]  # Keep last 100

                    # EARLY WARNING: Detect negative KL (policy collapse indicator)
                    if self.last_avg_kl < -2.0:
                        self.negative_kl_warnings += 1
                        logger.warning(
                            f"NEGATIVE KL DETECTED ({self.last_avg_kl:.2f}) - "
                            f"Student assigning lower probability to its outputs than reference. "
                            f"Warnings: {self.negative_kl_warnings}"
                        )
                        if self.negative_kl_warnings >= 5:
                            logger.warning(
                                "REPEATED NEGATIVE KL - Consider increasing kl_coef or "
                                "resetting reference model to prevent policy collapse."
                            )
                    else:
                        # Reset warning counter when KL is healthy
                        self.negative_kl_warnings = max(0, self.negative_kl_warnings - 1)

                    # ADAPTIVE KL CONTROL: Adjust kl_coef based on KL divergence
                    if self.adap_kl_ctrl and iteration % grad_accum == 0:
                        abs_kl = abs(self.last_avg_kl)
                        old_coef = self.current_kl_coef

                        if abs_kl > self.kl_target * 1.5:
                            # KL too high (in either direction) - increase penalty
                            self.current_kl_coef = min(self.current_kl_coef * self.kl_growth, self.kl_coef_max)
                            if self.current_kl_coef != old_coef:
                                logger.info(f"  KL coef increased: {old_coef:.4f} -> {self.current_kl_coef:.4f} (|KL|={abs_kl:.2f} > target*1.5={self.kl_target*1.5:.2f})")
                        elif abs_kl < self.kl_target * 0.5:
                            # KL too low - decrease penalty to allow more exploration
                            self.current_kl_coef = max(self.current_kl_coef * self.kl_decay, self.kl_coef_min)
                            if self.current_kl_coef != old_coef:
                                logger.info(f"  KL coef decreased: {old_coef:.4f} -> {self.current_kl_coef:.4f} (|KL|={abs_kl:.2f} < target*0.5={self.kl_target*0.5:.2f})")

                # --- Fine-grained iteration metrics CSV (every log_interval) ---
                self._log_iteration_metrics(
                    iteration, avg_reward, avg_kl,
                    accumulated_loss, iter_time
                )

                # Log per-task EuroEval accuracy
                if iteration % (log_interval * 5) == 0:
                    self._log_euroeval_accuracy()

                # Log memory usage for monitoring
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"  CUDA Memory - Allocated: {mem_alloc:.2f}GB, Reserved: {mem_reserved:.2f}GB")
            
            # Update adaptive weights
            reward_dict = self._organize_rewards_by_component(rewards)
            updated_weights = self.weight_scheduler.update(reward_dict, iteration)
            self.reward_model.update_weights(updated_weights)
            
            # Periodic maintenance
            if iteration % 100 == 0:
                self.task_buffer.recompute_priorities()
            
            # NOTE: Checkpoint saving moved to START of iteration (before any processing)
            # to ensure checkpoints are saved even when gradient spikes cause `continue`
            
            # Memory cleanup and energy maintenance
            if iteration % 10 == 0:
                torch.cuda.empty_cache()
                
                # Energy maintenance: ensure GPU stays active
                energy_interval = self.config.get('optimization', {}).get('energy_maintenance_interval', 50)
                if energy_interval > 0 and iteration % energy_interval == 0:
                    self._energy_maintenance_compute()
            
            # Periodically clean up completed eval checkpoints (every 500 iterations)
            if iteration % 500 == 0:
                self.cleanup_completed_eval_checkpoints(keep_best=True)
        
        # Save final model
        logger.info("Saving final model...")
        final_model_dir = self.output_dir / 'final_model'
        final_model_dir.mkdir(parents=True, exist_ok=True)
        self.student_model.save_pretrained(final_model_dir)
        self.student_tokenizer.save_pretrained(final_model_dir)
        
        # Save final model metadata
        final_metadata = {
            'iteration': n_iterations,
            'training_reward': float(np.mean([sum(r.values()) for r in rewards])) if rewards else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        with open(final_model_dir / 'model_metadata.json', 'w') as f:
            json.dump(final_metadata, f, indent=2)
        
        # Register best and final models for post-training EuroEval
        self.register_checkpoint_for_eval(str(self.best_model_dir), self.best_model_iteration)
        self.register_checkpoint_for_eval(str(final_model_dir), n_iterations)
        
        # Clean up current checkpoint (no longer needed)
        if self.current_checkpoint_dir.exists():
            logger.info(f"Cleaning up current checkpoint: {self.current_checkpoint_dir}")
            shutil.rmtree(self.current_checkpoint_dir)
        
        # Save training summary
        self._save_training_summary()
        
        logger.info("=" * 60)
        logger.info("Swedish GRPO training completed!")
        logger.info("=" * 60)
        logger.info(f"Models saved (storage-efficient):")
        logger.info(f"  - Best model (by training reward): {self.best_model_dir}")
        logger.info(f"    Iteration: {self.best_model_iteration}, Reward: {self.best_training_reward:.4f}")
        logger.info(f"  - Final model: {final_model_dir}")
        logger.info("")
        logger.info("Run post-training EuroEval evaluation:")
        logger.info(f"  sbatch scripts/run_grpo_euroeval.sh")
        logger.info("=" * 60)
    
    def check_evaluation_status(self) -> Dict[str, Any]:
        """
        Check the status of submitted EuroEval jobs and return summary.
        
        Returns dict with:
        - submitted_jobs: List of submitted job info
        - completed_results: List of completed evaluation results
        - pending_count: Number of pending evaluations
        """
        status = {
            'submitted_jobs': [],
            'completed_results': [],
            'pending_count': 0,
            'best_score': None,
            'best_iteration': None
        }
        
        # Read submitted jobs
        jobs_file = self.metrics_dir / 'submitted_eval_jobs.jsonl'
        if jobs_file.exists():
            with open(jobs_file, 'r') as f:
                for line in f:
                    if line.strip():
                        status['submitted_jobs'].append(json.loads(line))
        
        # Read completed results
        results_file = self.metrics_dir / 'euroeval_grpo_progress.jsonl'
        if results_file.exists():
            with open(results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        status['completed_results'].append(result)
                        
                        # Track best score
                        score = result.get('aggregate_score', 0)
                        if status['best_score'] is None or score > status['best_score']:
                            status['best_score'] = score
                            status['best_iteration'] = result.get('iteration')
        
        # Count pending
        completed_iterations = {r['iteration'] for r in status['completed_results']}
        submitted_iterations = {j['iteration'] for j in status['submitted_jobs']}
        status['pending_count'] = len(submitted_iterations - completed_iterations)
        
        return status
    
    def cleanup_completed_eval_checkpoints(self, keep_best: bool = True):
        """
        Clean up eval checkpoints for completed evaluations to save disk space.
        
        Args:
            keep_best: If True, keep the checkpoint with the best EuroEval score
        """
        status = self.check_evaluation_status()
        completed_iterations = {r['iteration'] for r in status['completed_results']}
        
        eval_checkpoints_dir = self.output_dir / 'eval_checkpoints'
        if not eval_checkpoints_dir.exists():
            return
        
        cleaned_count = 0
        best_iteration = status.get('best_iteration')
        
        for checkpoint_dir in eval_checkpoints_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue
            
            # Parse iteration from directory name (e.g., "iter_100")
            try:
                iter_str = checkpoint_dir.name.replace('iter_', '')
                iteration = int(iter_str)
            except ValueError:
                continue
            
            # Only clean up completed evaluations
            if iteration not in completed_iterations:
                continue
            
            # Optionally keep the best checkpoint
            if keep_best and iteration == best_iteration:
                logger.info(f"Keeping best eval checkpoint: {checkpoint_dir}")
                continue
            
            # Remove the checkpoint
            try:
                shutil.rmtree(checkpoint_dir)
                cleaned_count += 1
                logger.debug(f"Cleaned up eval checkpoint: {checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {checkpoint_dir}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} completed eval checkpoints")
    
    def log_evaluation_status(self):
        """Log current evaluation status to the logger."""
        status = self.check_evaluation_status()
        
        logger.info("=" * 40)
        logger.info("EuroEval Status")
        logger.info("=" * 40)
        logger.info(f"Submitted jobs: {len(status['submitted_jobs'])}")
        logger.info(f"Completed: {len(status['completed_results'])}")
        logger.info(f"Pending: {status['pending_count']}")
        
        if status['best_score'] is not None:
            logger.info(f"Best score: {status['best_score']:.4f} (iteration {status['best_iteration']})")
        
        if status['completed_results']:
            logger.info("Recent results:")
            for result in status['completed_results'][-3:]:
                logger.info(f"  Iter {result['iteration']}: {result['aggregate_score']:.4f}")
        
        logger.info("=" * 40)
    
    def _log_eval_summary(self, eval_result: Dict[str, Any]):
        """Log evaluation summary to CSV for easy tracking."""
        csv_file = self.metrics_dir / 'euroeval_progress.csv'
        
        # Write header if file doesn't exist
        write_header = not csv_file.exists()
        
        scores = eval_result.get('scores', {})
        row = {
            'iteration': eval_result.get('iteration', 0),
            'timestamp': eval_result.get('timestamp', ''),
            'aggregate_score': eval_result.get('aggregate_score', 0),
            **scores
        }
        
        with open(csv_file, 'a', newline='') as f:
            fieldnames = ['iteration', 'timestamp', 'aggregate_score'] + list(scores.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    
    def _save_current_checkpoint(self, iteration: int):
        """Save rolling current checkpoint (overwrites previous)."""
        # Clear existing checkpoint
        if self.current_checkpoint_dir.exists():
            shutil.rmtree(self.current_checkpoint_dir)
        
        self.current_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.student_model.save_pretrained(self.current_checkpoint_dir)
        self.student_tokenizer.save_pretrained(self.current_checkpoint_dir)
        
        # Save metadata
        metadata = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.current_checkpoint_dir / 'checkpoint_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Current checkpoint saved at iteration {iteration}")
    
    def _save_persistent_checkpoint(self, iteration: int):
        """Save a persistent checkpoint that is NOT overwritten.

        These are kept for the full duration of training so that every
        eval_interval checkpoint can be evaluated or analysed post-hoc.
        Stored under ``checkpoints/iter_<N>/``.
        """
        persist_dir = self.checkpoint_dir / f'iter_{iteration}'
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.student_model.save_pretrained(persist_dir)
        self.student_tokenizer.save_pretrained(persist_dir)

        metadata = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'n_mastered': len(self.mastered_tasks),
            'mastered_tasks': sorted(self.mastered_tasks),
            'best_training_reward': self.best_training_reward,
        }
        with open(persist_dir / 'checkpoint_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Persistent checkpoint saved: {persist_dir}")

    def _save_best_model(self, iteration: int, reward: float):
        """Save best model checkpoint (overwrites previous best)."""
        # Clear existing best model
        if self.best_model_dir.exists():
            shutil.rmtree(self.best_model_dir)
        
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.student_model.save_pretrained(self.best_model_dir)
        self.student_tokenizer.save_pretrained(self.best_model_dir)
        
        # Save metadata
        metadata = {
            'iteration': iteration,
            'training_reward': float(reward),
            'timestamp': datetime.now().isoformat(),
            'note': 'Best model by training reward (EuroEval evaluation pending)'
        }
        with open(self.best_model_dir / 'best_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Best model saved at iteration {iteration} with reward {reward:.4f}")
    
    def _log_training_metrics(self, iteration: int, loss: float, reward: float):
        """Log training metrics to CSV for tracking progress (every eval_interval)."""
        csv_file = self.metrics_dir / 'training_progress.csv'

        write_header = not csv_file.exists()

        avg_kl = getattr(self, 'last_avg_kl', 0.0)

        # Rolling reward by source
        euro_recent = self.reward_tracker["euroeval"]["recent"]
        sp_recent = self.reward_tracker["selfplay"]["recent"]
        euro_avg = sum(euro_recent) / len(euro_recent) if euro_recent else 0.0
        sp_avg = sum(sp_recent) / len(sp_recent) if sp_recent else 0.0

        row = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'loss': loss,
            'avg_reward': reward,
            'avg_kl': avg_kl,
            'kl_coef': self.current_kl_coef,
            'best_reward': self.best_training_reward,
            'best_iteration': self.best_model_iteration,
            'n_mastered': len(self.mastered_tasks),
            'mastered_tasks': ";".join(sorted(self.mastered_tasks)),
            'selfplay_enabled': self.selfplay_enabled,
            'selfplay_ratio': self.selfplay_ratio,
            'euro_avg_reward': euro_avg,
            'selfplay_avg_reward': sp_avg,
        }

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _log_iteration_metrics(self, iteration: int, avg_reward: float,
                               avg_kl: float, loss: float, iter_time: float):
        """Log fine-grained per-iteration metrics (every log_interval).

        This CSV is the primary data source for paper plots (reward curves,
        KL trajectory, mastery transitions over time).
        """
        csv_file = self.metrics_dir / 'iteration_metrics.csv'
        write_header = not csv_file.exists()

        # Compute overall EuroEval accuracy
        total_correct = sum(v["correct"] for v in self.task_accuracy.values())
        total_count = sum(v["total"] for v in self.task_accuracy.values())
        overall_acc = total_correct / max(total_count, 1)

        mem_alloc = 0.0
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3

        # Self-play aggregate stats
        sp_total_correct = sum(v["correct"] for v in self.selfplay_accuracy.values())
        sp_total_count = sum(v["total"] for v in self.selfplay_accuracy.values())
        sp_overall_acc = sp_total_correct / max(sp_total_count, 1)

        # Rolling average rewards by source
        euro_recent = self.reward_tracker["euroeval"]["recent"]
        sp_recent = self.reward_tracker["selfplay"]["recent"]
        euro_avg_reward = sum(euro_recent) / len(euro_recent) if euro_recent else 0.0
        sp_avg_reward = sum(sp_recent) / len(sp_recent) if sp_recent else 0.0

        row = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'avg_reward': avg_reward,
            'avg_kl': avg_kl,
            'kl_coef': self.current_kl_coef,
            'loss': loss,
            'iter_time_s': iter_time,
            'n_mastered': len(self.mastered_tasks),
            'mastered_tasks': ";".join(sorted(self.mastered_tasks)),
            'overall_euroeval_acc': overall_acc,
            'euroeval_total_samples': total_count,
            'selfplay_enabled': self.selfplay_enabled,
            'selfplay_ratio': self.selfplay_ratio,
            'selfplay_eligible': ";".join(sorted(self.selfplay_eligible_mastered)),
            'selfplay_total_samples': sp_total_count,
            'selfplay_overall_acc': sp_overall_acc,
            'euro_avg_reward': euro_avg_reward,
            'selfplay_avg_reward': sp_avg_reward,
            'gpu_mem_gb': mem_alloc,
        }
        # Per-task EuroEval accuracy columns
        for task_type, stats in self.task_accuracy.items():
            row[f'{task_type}_acc'] = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            )
        # Per-task self-play accuracy columns
        for task_type, stats in self.selfplay_accuracy.items():
            row[f'sp_{task_type}_acc'] = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            )

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_training_summary(self):
        """Save final training summary (evaluation results added post-training)."""
        # Count pending evaluation checkpoints
        eval_queue_file = self.metrics_dir / 'pending_eval_checkpoints.jsonl'
        pending_count = 0
        if eval_queue_file.exists():
            with open(eval_queue_file, 'r') as f:
                pending_count = sum(1 for line in f if line.strip())
        
        summary = {
            'total_iterations': self.iteration,
            'training_time_hours': (time.time() - self.start_time) / 3600 if self.start_time else 0,
            'eval_interval': self.eval_interval,
            'pending_evaluations': pending_count,
            'eval_queue_file': str(eval_queue_file),
            'best_model': {
                'iteration': self.best_model_iteration,
                'training_reward': self.best_training_reward,
                'path': str(self.best_model_dir),
                'note': 'Best by training reward; run EuroEval for validation score'
            },
            'final_model': {
                'path': str(self.output_dir / 'final_model')
            },
            'eval_history': self.eval_history,
            'config': {
                'eval_interval': self.eval_interval,
                'rl_config': self.config.get('rl', {})
            },
            'post_training_eval_required': True,
            'storage_note': 'Only best_model and final_model saved (storage-efficient)'
        }
        
        summary_file = self.output_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_file}")
    
    def _create_reference_model(self):
        """
        Create a reference model for KL penalty.
        
        Uses simple deepcopy approach like the German trainer that worked.
        Memory optimized: disables gradient checkpointing and caching for reference.
        """
        logger.info("Creating reference model via deepcopy...")
        
        # Synchronize CUDA and clear cache before copy
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Get actual device of student model
        student_device = self._get_student_device()
        
        # Simple deepcopy approach - same as German trainer that worked
        ref = copy.deepcopy(self.student_model).to(student_device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        
        # Disable gradient checkpointing on reference model (not needed, saves memory)
        try:
            if hasattr(ref, 'gradient_checkpointing_disable'):
                ref.gradient_checkpointing_disable()
        except Exception:
            pass
        
        # Ensure no caching on reference model
        try:
            ref.config.use_cache = False
        except Exception:
            pass
        
        # Clear cache after copy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Reference model created successfully")
        return ref
    
    def _gpu_warmup(self, duration_seconds: int = 60):
        """
        Perform GPU warmup to ensure high power utilization from the start.
        This prevents the job from being killed for low GPU usage during initialization.
        
        NOTE: Only warms up student model to avoid OOM with large teacher models.
        Teacher model warmup happens naturally during first reward computation.
        """
        logger.info(f"Starting GPU warmup for {duration_seconds} seconds...")
        start_time = time.time()
        warmup_iterations = 0
        
        # Use shorter text to reduce memory pressure
        warmup_text = "GPU warmup: " + "maintaining GPU utilization " * 20
        
        # Get actual device of student model
        student_device = self._get_student_device()
        
        try:
            while time.time() - start_time < duration_seconds:
                # Tokenize with shorter sequence
                inputs = self.student_tokenizer(
                    warmup_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256  # Reduced from 512
                ).to(student_device)
                
                # Forward pass with student model only (skip teacher to avoid OOM)
                with torch.no_grad():
                    self.student_model.eval()
                    _ = self.student_model(**inputs, use_cache=False)
                    
                    # Generate fewer tokens to reduce memory pressure
                    outputs = self.student_model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=32,  # Reduced from 64
                        do_sample=True,
                        temperature=0.8,
                        use_cache=False,
                        pad_token_id=self.student_tokenizer.eos_token_id
                    )
                
                warmup_iterations += 1
                
                # Log progress every 10 iterations
                if warmup_iterations % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"GPU warmup: {warmup_iterations} iterations, {elapsed:.1f}s elapsed")
                
                # Clean up after each iteration
                del inputs, outputs
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"GPU warmup encountered an error: {e}")
            # Critical: Reset CUDA state after error to prevent downstream failures
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # If CUDA is in bad state, try to reset it
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception as reset_error:
                logger.error(f"Failed to reset CUDA state: {reset_error}")
                # Re-raise to prevent training with corrupted CUDA state
                raise RuntimeError(
                    f"CUDA error during warmup cannot be recovered: {e}. "
                    "Try reducing model size or batch size."
                ) from e
        
        logger.info(f"GPU warmup completed after {warmup_iterations} iterations")
    
    def _get_teacher_device(self) -> torch.device:
        """Get the device of the teacher model"""
        try:
            return next(self.teacher_model.parameters()).device
        except Exception:
            return self.device
    
    def _get_student_device(self) -> torch.device:
        """Get the device of the student model - handles device_map models"""
        try:
            return next(self.student_model.parameters()).device
        except Exception:
            return self.device
    
    def _energy_maintenance_compute(self):
        """
        Perform compute-intensive operations to maintain GPU energy consumption.
        This helps prevent the job from being killed for low GPU usage during
        periods where the main training might be doing CPU-bound operations.
        
        NOTE: Only uses student model to avoid OOM with large teacher models.
        """
        logger.debug("Performing energy maintenance compute...")
        
        # Use shorter text to reduce memory pressure
        maintenance_text = "Energy maintenance: " + "maintaining GPU utilization " * 15
        
        # Get actual device of student model
        student_device = self._get_student_device()
        
        try:
            with torch.no_grad():
                # Student model compute only (skip teacher to avoid OOM)
                inputs = self.student_tokenizer(
                    maintenance_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=128  # Reduced
                ).to(student_device)
                
                self.student_model.eval()
                _ = self.student_model(**inputs, use_cache=False)
                
                # Generate fewer tokens
                outputs = self.student_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=16,  # Reduced from 64
                    do_sample=True,
                    temperature=0.8,
                    use_cache=False,
                    pad_token_id=self.student_tokenizer.eos_token_id
                )
                
                # Clean up
                del inputs, outputs
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.debug(f"Energy maintenance compute encountered error: {e}")
            # Don't propagate - this is non-critical
            torch.cuda.empty_cache()
        
        # Restore training mode
        self.student_model.train()
    
    def _populate_task_buffer(self):
        """Populate task buffer with EuroEval + optional self-play tasks.

        EuroEval tasks are weighted toward *unmastered* types (3x vs 1x).
        Self-play tasks are introduced adaptively once the model reaches
        SELFPLAY_UNLOCK_THRESHOLD overall accuracy AND has mastered at
        least one eligible type.

        The split is: at least MIN_EUROEVAL_RATIO EuroEval, at most
        MAX_SELFPLAY_RATIO self-play.  The actual self-play fraction
        scales with how many eligible types are mastered.
        """
        min_tasks = self.config.get('rl', {}).get('batch_size', 8) * 2

        # Recompute mastery + self-play eligibility
        self._update_mastery_state()

        tasks_needed = max(0, min_tasks - self.task_buffer.size)
        if tasks_needed <= 0:
            return

        # Compute adaptive self-play / euroeval split
        if self.selfplay_enabled and self.selfplay_eligible_mastered:
            n_eligible = len(SELFPLAY_ELIGIBLE_TYPES)
            n_mastered_eligible = len(self.selfplay_eligible_mastered)
            self.selfplay_ratio = (n_mastered_eligible / n_eligible) * MAX_SELFPLAY_RATIO
        else:
            self.selfplay_ratio = 0.0

        n_selfplay = int(tasks_needed * self.selfplay_ratio)
        n_euroeval = tasks_needed - n_selfplay

        # Fill EuroEval portion
        added_euro = 0
        attempts = 0
        while added_euro < n_euroeval and attempts < n_euroeval * 3:
            task = self._create_euroeval_task_weighted()
            if task:
                self.task_buffer.insert(task)
                added_euro += 1
            attempts += 1

        # Fill self-play portion
        added_sp = 0
        attempts = 0
        while added_sp < n_selfplay and attempts < n_selfplay * 3:
            task = self._create_selfplay_task()
            if task:
                self.task_buffer.insert(task)
                added_sp += 1
            attempts += 1

        if self.iteration <= 5 or self.iteration % 50 == 0:
            mastered_str = ", ".join(sorted(self.mastered_tasks)) or "none"
            sp_str = ", ".join(sorted(self.selfplay_eligible_mastered)) or "none"
            logger.info(
                f"Task buffer populated: {added_euro} EuroEval + {added_sp} self-play | "
                f"selfplay_ratio={self.selfplay_ratio:.2f} | "
                f"mastered=[{mastered_str}] | selfplay_eligible=[{sp_str}]"
            )

    # ------------------------------------------------------------------
    # Adaptive mastery tracking
    # ------------------------------------------------------------------

    def _update_mastery_state(self):
        """Recompute per-task mastery and self-play eligibility.

        A task type is *mastered* when:
          1. It has been evaluated on >= MIN_SAMPLES_FOR_MASTERY examples, AND
          2. Its accuracy >= TASK_MASTERY_THRESHOLD.

        Mastered types receive 1x sampling weight; unmastered types receive 3x.

        Self-play is unlocked when overall EuroEval accuracy exceeds
        SELFPLAY_UNLOCK_THRESHOLD and corpus texts are available.
        """
        total_count = sum(v["total"] for v in self.task_accuracy.values())

        if total_count < 50:
            self.mastered_tasks = set()
            self.selfplay_enabled = False
            self.selfplay_eligible_mastered = set()
            return

        prev_mastered = set(self.mastered_tasks)
        self.mastered_tasks = set()
        for task_type, stats in self.task_accuracy.items():
            if stats["total"] >= MIN_SAMPLES_FOR_MASTERY:
                acc = stats["correct"] / stats["total"]
                if acc >= TASK_MASTERY_THRESHOLD:
                    self.mastered_tasks.add(task_type)

        # Log mastery transitions
        newly_mastered = self.mastered_tasks - prev_mastered
        for tt in newly_mastered:
            stats = self.task_accuracy[tt]
            acc = stats["correct"] / stats["total"]
            logger.info(
                f"Task type MASTERED: {tt} "
                f"({acc:.1%}, {stats['correct']}/{stats['total']}) — "
                f"sampling weight reduced to 1x"
            )

        # Self-play unlock check
        total_correct = sum(v["correct"] for v in self.task_accuracy.values())
        overall_acc = total_correct / total_count if total_count > 0 else 0.0

        prev_enabled = self.selfplay_enabled
        if overall_acc >= SELFPLAY_UNLOCK_THRESHOLD and self._monolingual_texts:
            self.selfplay_enabled = True
            self.selfplay_eligible_mastered = self.mastered_tasks & SELFPLAY_ELIGIBLE_TYPES
        else:
            self.selfplay_enabled = False
            self.selfplay_eligible_mastered = set()

        if self.selfplay_enabled and not prev_enabled:
            logger.info(
                f"SELF-PLAY UNLOCKED: overall EuroEval accuracy {overall_acc:.1%} "
                f">= {SELFPLAY_UNLOCK_THRESHOLD:.0%} | "
                f"eligible types: {sorted(self.selfplay_eligible_mastered)}"
            )

    # ------------------------------------------------------------------
    # EuroEval task creation (weighted toward unmastered types)
    # ------------------------------------------------------------------

    def _create_euroeval_task_weighted(self) -> Optional[Task]:
        """Create a EuroEval task, weighting toward *unmastered* task types.

        Unmastered core task types are sampled 3x more often than mastered ones.
        This focuses ground-truth practice on the model's weak areas.
        """
        if not self.euroeval_data:
            return None

        # Build weighted candidate list of (dataset_name, task_type)
        candidates = []
        for ds_name, samples in self.euroeval_data.items():
            if not samples:
                continue
            ds_config = EUROEVAL_DATASETS.get(ds_name, {})
            task_type = ds_config.get("task_type", "unknown")
            # Unmastered types get weight 3, mastered get weight 1
            weight = 1 if task_type in self.mastered_tasks else 3
            candidates.extend([(ds_name, task_type)] * weight)

        if not candidates:
            return None

        dataset_name, task_type = random.choice(candidates)
        samples = self.euroeval_data[dataset_name]

        # Round-robin sampling with shuffle on wrap-around
        idx = self._euroeval_indices.get(dataset_name, 0)
        if idx >= len(samples):
            random.shuffle(samples)
            idx = 0
        self._euroeval_indices[dataset_name] = idx + 1
        sample = samples[idx]

        # Create task based on dataset type
        if task_type == "sentiment":
            return self._create_sentiment_task(sample, dataset_name)
        elif task_type == "acceptability":
            return self._create_acceptability_task(sample, dataset_name)
        elif task_type == "ner":
            return self._create_ner_task(sample, dataset_name)
        elif task_type == "reading_comprehension":
            return self._create_reading_comprehension_task(sample, dataset_name)
        elif task_type == "commonsense":
            return self._create_commonsense_task(sample, dataset_name)
        elif task_type == "knowledge":
            return self._create_knowledge_task(sample, dataset_name)

        return None

    def _create_sentiment_task(self, sample: Dict, dataset: str) -> Task:
        """Create a sentiment classification task from SweRec data."""
        text = sample.get("text", "")[:800]
        label = sample.get("label", "")

        prompt = (
            "Klassificera sentimentet i följande recension som "
            "'positiv', 'negativ', 'neutral' eller 'blandad'. "
            "Svara med ett enda ord.\n\n"
            f"Recension: {text}\n\n"
            "Sentiment:"
        )

        return Task(
            task_type="sentiment",
            input_text=prompt,
            metadata={
                "ground_truth": label,
                "source": "euroeval",
                "dataset": dataset,
                "original_text": text,
            },
            expected_success_rate=0.5,
        )

    def _create_acceptability_task(self, sample: Dict, dataset: str) -> Task:
        """Create a linguistic acceptability task from ScaLA-sv data."""
        text = sample.get("text", "")
        label = sample.get("label", "")  # "correct" or "incorrect"

        prompt = (
            "Bedöm om följande mening är grammatiskt korrekt och naturlig "
            "på svenska. Svara med ett enda ord: 'korrekt' eller 'inkorrekt'.\n\n"
            f"Mening: {text}\n\n"
            "Bedömning:"
        )

        return Task(
            task_type="acceptability",
            input_text=prompt,
            metadata={
                "ground_truth": label,
                "source": "euroeval",
                "dataset": dataset,
                "corruption_type": sample.get("corruption_type"),
            },
            expected_success_rate=0.5,
        )

    def _create_ner_task(self, sample: Dict, dataset: str) -> Task:
        """Create a NER task from SUC3 data."""
        text = sample.get("text", "")
        tokens = sample.get("tokens", [])
        labels = sample.get("labels", [])

        # Extract entities from BIO tags for ground truth
        entities = _extract_entities_from_bio(tokens, labels)
        if entities:
            gt_str = "; ".join(f"{ent} ({typ})" for ent, typ in entities)
        else:
            gt_str = "Inga entiteter"

        prompt = (
            "Identifiera alla namngivna entiteter i följande text. "
            "Lista varje entitet med dess typ (PER för person, LOC för plats, "
            "ORG för organisation). Om inga entiteter finns, skriv 'Inga entiteter'.\n\n"
            f"Text: {text}\n\n"
            "Entiteter:"
        )

        return Task(
            task_type="ner",
            input_text=prompt,
            metadata={
                "ground_truth": gt_str,
                "source": "euroeval",
                "dataset": dataset,
                "tokens": tokens,
                "bio_labels": labels,
            },
            expected_success_rate=0.4,
        )

    def _create_reading_comprehension_task(self, sample: Dict, dataset: str) -> Task:
        """Create a reading comprehension task from ScandiQA-sv data."""
        context = sample.get("context", "")[:800]
        question = sample.get("question", "")
        answers = sample.get("answers", {})

        # Extract the first answer text
        answer_texts = answers.get("text", [])
        ground_truth = answer_texts[0] if answer_texts else ""

        prompt = (
            f"Kontext: {context}\n\n"
            f"Fråga: {question}\n\n"
            "Svara kort baserat på kontexten."
        )

        return Task(
            task_type="reading_comprehension",
            input_text=prompt,
            metadata={
                "ground_truth": ground_truth,
                "source": "euroeval",
                "dataset": dataset,
                "context": context,
                "question": question,
            },
            expected_success_rate=0.4,
        )

    def _create_commonsense_task(self, sample: Dict, dataset: str) -> Task:
        """Create a common-sense reasoning task from HellaSwag-sv data."""
        text = sample.get("text", "")
        label = sample.get("label", "")  # a, b, c, or d

        prompt = (
            f"{text}\n\n"
            "Välj rätt svarsalternativ. Svara med enbart bokstaven (a, b, c eller d):"
        )

        return Task(
            task_type="commonsense",
            input_text=prompt,
            metadata={
                "ground_truth": label,
                "source": "euroeval",
                "dataset": dataset,
            },
            expected_success_rate=0.3,
        )

    def _create_knowledge_task(self, sample: Dict, dataset: str) -> Task:
        """Create a knowledge task from MMLU-sv data."""
        text = sample.get("text", "")
        label = sample.get("label", "")  # a, b, c, or d

        prompt = (
            f"{text}\n\n"
            "Välj rätt svarsalternativ. Svara med enbart bokstaven (a, b, c eller d):"
        )

        return Task(
            task_type="knowledge",
            input_text=prompt,
            metadata={
                "ground_truth": label,
                "source": "euroeval",
                "dataset": dataset,
                "category": sample.get("category", ""),
            },
            expected_success_rate=0.3,
        )

    # ------------------------------------------------------------------
    # Self-play task generation (synthetic from FineWeb corpus text)
    # ------------------------------------------------------------------

    def _get_euroeval_example(self, task_type: str) -> str:
        """Sample a random (input, gold_answer) pair from EuroEval and format as text.

        Used as ``correct_example`` in self-play tasks so the LLM judge has a
        reference for what a good answer looks like.
        """
        # Find the dataset name for this task type
        ds_name = None
        for name, cfg in EUROEVAL_DATASETS.items():
            if cfg["task_type"] == task_type:
                ds_name = name
                break
        if ds_name is None or ds_name not in self.euroeval_data:
            return ""

        samples = self.euroeval_data[ds_name]
        if not samples:
            return ""

        sample = random.choice(samples)

        if task_type == "sentiment":
            text = sample.get("text", "")[:400]
            label = sample.get("label", "")
            return f"Text: {text}\nKorrekt sentiment: {label}"

        elif task_type == "acceptability":
            text = sample.get("text", "")
            label = sample.get("label", "")
            return f"Mening: {text}\nKorrekt bedömning: {label}"

        elif task_type == "ner":
            tokens = sample.get("tokens", [])
            labels = sample.get("labels", [])
            entities = _extract_entities_from_bio(tokens, labels)
            text = sample.get("text", "") or " ".join(tokens)
            if entities:
                ent_str = "; ".join(f"{e} ({t})" for e, t in entities)
            else:
                ent_str = "Inga entiteter"
            return f"Text: {text[:400]}\nKorrekta entiteter: {ent_str}"

        elif task_type == "reading_comprehension":
            context = sample.get("context", "")[:400]
            question = sample.get("question", "")
            answers = sample.get("answers", {})
            answer_texts = answers.get("text", [])
            answer = answer_texts[0] if answer_texts else ""
            return f"Kontext: {context}\nFråga: {question}\nKorrekt svar: {answer}"

        return ""

    def _create_selfplay_task(self) -> Optional[Task]:
        """Create a self-play task for a random mastered+eligible type."""
        if not self._monolingual_texts:
            return None
        eligible = list(self.selfplay_eligible_mastered)
        if not eligible:
            return None

        task_type = random.choice(eligible)

        if task_type == "sentiment":
            return self._create_selfplay_sentiment_task()
        elif task_type == "acceptability":
            return self._create_selfplay_acceptability_task()
        elif task_type == "ner":
            return self._create_selfplay_ner_task()
        elif task_type == "reading_comprehension":
            return self._create_selfplay_rc_task()
        return None

    def _create_selfplay_sentiment_task(self) -> Optional[Task]:
        """Synthetic sentiment task from corpus text."""
        for _ in range(10):
            text = random.choice(self._monolingual_texts)
            if len(text) > 50:
                break
        else:
            return None

        text = text[:800]
        prompt = (
            "Klassificera sentimentet i följande text som "
            "'positiv', 'negativ', 'neutral' eller 'blandad'. "
            "Svara med ett enda ord.\n\n"
            f"Text: {text}\n\n"
            "Sentiment:"
        )
        correct_example = self._get_euroeval_example("sentiment")
        return Task(
            task_type="sentiment",
            input_text=prompt,
            metadata={
                "source": "selfplay",
                "correct_example": correct_example,
            },
            expected_success_rate=0.4,
        )

    def _create_selfplay_acceptability_task(self) -> Optional[Task]:
        """Synthetic acceptability task from a single corpus sentence."""
        for _ in range(10):
            text = random.choice(self._monolingual_texts)
            # Pick a random sentence from the passage
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
            if sentences:
                break
        else:
            return None

        sentence = random.choice(sentences)[:300]
        prompt = (
            "Bedöm om följande mening är grammatiskt korrekt och naturlig "
            "på svenska. Svara med ett enda ord: 'korrekt' eller 'inkorrekt'.\n\n"
            f"Mening: {sentence}.\n\n"
            "Bedömning:"
        )
        correct_example = self._get_euroeval_example("acceptability")
        return Task(
            task_type="acceptability",
            input_text=prompt,
            metadata={
                "source": "selfplay",
                "correct_example": correct_example,
            },
            expected_success_rate=0.4,
        )

    def _create_selfplay_ner_task(self) -> Optional[Task]:
        """Synthetic NER task from corpus text."""
        for _ in range(10):
            text = random.choice(self._monolingual_texts)
            if len(text) > 50:
                break
        else:
            return None

        text = text[:600]
        prompt = (
            "Identifiera alla namngivna entiteter i följande text. "
            "Lista varje entitet med dess typ (PER för person, LOC för plats, "
            "ORG för organisation). Om inga entiteter finns, skriv 'Inga entiteter'.\n\n"
            f"Text: {text}\n\n"
            "Entiteter:"
        )
        correct_example = self._get_euroeval_example("ner")
        return Task(
            task_type="ner",
            input_text=prompt,
            metadata={
                "source": "selfplay",
                "correct_example": correct_example,
            },
            expected_success_rate=0.3,
        )

    def _create_selfplay_rc_task(self) -> Optional[Task]:
        """Synthetic reading comprehension task from corpus text."""
        for _ in range(10):
            text = random.choice(self._monolingual_texts)
            if len(text) > 100:
                break
        else:
            return None

        text = text[:800]
        prompt = (
            f"Kontext: {text}\n\n"
            "Fråga: Vad handlar texten om? Svara kort baserat på kontexten."
        )
        correct_example = self._get_euroeval_example("reading_comprehension")
        return Task(
            task_type="reading_comprehension",
            input_text=prompt,
            metadata={
                "source": "selfplay",
                "correct_example": correct_example,
            },
            expected_success_rate=0.3,
        )

    def _self_play_rollout(self, task_batch: List[Task]) -> Tuple[List[str], List[Dict], List[float]]:
        """Perform GRPO rollout with weighted reward advantages."""
        all_solutions = []
        all_rewards = []
        all_advantages = []
        
        grpo_group_size = self.config.get('rl', {}).get('grpo_group_size', 4)
        advantage_clip = float(self.config.get('rl', {}).get('advantage_clip', 3.0))
        advantage_epsilon = float(self.config.get('rl', {}).get('advantage_epsilon', 1e-6))
        
        # Detailed logging for first few iterations
        verbose_logging = self.iteration <= 3
        total_tasks = len(task_batch)
        
        for task_idx, task in enumerate(task_batch):
            solutions = []
            rewards = []
            task_start = time.time()
            
            if verbose_logging:
                logger.info(f"    [Rollout] Task {task_idx+1}/{total_tasks} ({task.task_type})")
            
            for sol_idx in range(grpo_group_size):
                gen_start = time.time()
                solution = self._generate_student_solution(task)
                gen_time = time.time() - gen_start
                solutions.append(solution)
                
                if verbose_logging:
                    logger.info(f"      [Rollout] Solution {sol_idx+1}/{grpo_group_size} generated ({gen_time:.1f}s)")
                
                reward_start = time.time()
                reward_dict = self.reward_model.compute_rewards(task, solution)
                reward_time = time.time() - reward_start
                rewards.append(reward_dict)
                
                if verbose_logging:
                    logger.info(f"      [Rollout] Reward computed ({reward_time:.1f}s): {reward_dict}")
            
            if verbose_logging:
                task_time = time.time() - task_start
                logger.info(f"    [Rollout] Task {task_idx+1} complete ({task_time:.1f}s total)")
            
            # Compute weighted total rewards using active weights from scheduler
            current_weights = self.weight_scheduler.get_weights()
            active_weights = self._get_active_weights(current_weights)
            total_rewards = [
                sum(active_weights.get(k, 0.0) * v for k, v in r.items() if isinstance(v, (int, float))) for r in rewards
            ]
            
            # Robust advantage computation with numerical stability
            if len(total_rewards) == 0:
                advantages = []
            else:
                tr = np.array(total_rewards, dtype=np.float64)
                if not np.isfinite(tr).all():
                    logger.warning("Non-finite rewards detected in GRPO group, using zero advantages")
                    advantages = [0.0] * len(tr)
                else:
                    mean_reward = float(np.mean(tr))
                    std_reward = float(np.std(tr))
                    
                    if std_reward < 1e-4 or np.isclose(std_reward, 0.0):
                        # Rank-based advantages when variance is too low
                        ranks = np.argsort(np.argsort(tr))
                        if len(ranks) > 1:
                            advantages = 2.0 * (ranks / (len(ranks) - 1)) - 1.0
                            advantages = advantages.tolist()
                        else:
                            advantages = [0.0] * len(tr)
                        logger.debug(f"Using rank-based advantages due to low variance (std={std_reward:.6f})")
                    else:
                        advantages = (tr - mean_reward) / (std_reward + advantage_epsilon)
                        advantages = np.clip(advantages, -advantage_clip, advantage_clip).tolist()
            
            # Update task buffer with results and track accuracy by source
            for sol, reward_dict in zip(solutions, rewards):
                active_weights = self._get_active_weights(self.weight_scheduler.get_weights())
                total_reward = sum(
                    active_weights.get(k, 0.0) * v
                    for k, v in reward_dict.items()
                    if k != "reward_method"
                )
                success = total_reward > 0.0
                self.task_buffer.update_task_result(task, success, sol)

                source = task.metadata.get("source", "euroeval")
                tt = task.task_type

                if source == "euroeval":
                    # EuroEval accuracy (used for mastery computation)
                    if tt in self.task_accuracy:
                        self.task_accuracy[tt]["total"] += 1
                        if success:
                            self.task_accuracy[tt]["correct"] += 1
                    # Reward tracker
                    tracker = self.reward_tracker["euroeval"]
                    tracker["count"] += 1
                    tracker["recent"].append(total_reward)
                    if len(tracker["recent"]) > 200:
                        tracker["recent"] = tracker["recent"][-200:]

                elif source == "selfplay":
                    # Self-play accuracy (separate from mastery)
                    if tt in self.selfplay_accuracy:
                        self.selfplay_accuracy[tt]["total"] += 1
                        if success:
                            self.selfplay_accuracy[tt]["correct"] += 1
                    # Reward tracker
                    tracker = self.reward_tracker["selfplay"]
                    tracker["count"] += 1
                    tracker["recent"].append(total_reward)
                    if len(tracker["recent"]) > 200:
                        tracker["recent"] = tracker["recent"][-200:]
            
            all_solutions.extend(solutions)
            all_rewards.extend(rewards)
            all_advantages.extend(advantages)
            
            # Clear CUDA cache after each task to prevent memory buildup during rollout
            # This is critical for preventing OOM when policy loss starts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_solutions, all_rewards, all_advantages
    
    def _generate_student_solution(self, task: Task) -> str:
        """Generate solution from student model.

        Uses task-specific generation parameters.  All tasks are EuroEval:
        classification/MC tasks get very short output, NER and RC get moderate.
        """
        gen_config = self.config.get('rl', {}).get('generation', {})

        # Classification tasks (one word: positive/negative, korrekt/inkorrekt)
        if task.task_type in {"sentiment", "acceptability"}:
            max_new_tokens = 20
            temperature = 0.5
            repetition_penalty = 1.3
        # Multiple-choice tasks (single letter: a/b/c/d)
        elif task.task_type in {"commonsense", "knowledge"}:
            max_new_tokens = 20
            temperature = 0.5
            repetition_penalty = 1.3
        # NER (entity list)
        elif task.task_type == "ner":
            max_new_tokens = 128
            temperature = 0.6
            repetition_penalty = 1.2
        # Reading comprehension (short answer)
        elif task.task_type == "reading_comprehension":
            max_new_tokens = 128
            temperature = 0.6
            repetition_penalty = 1.1
        else:
            # Fallback
            max_new_tokens = gen_config.get('max_new_tokens', 64)
            temperature = gen_config.get('temperature', 0.6)
            repetition_penalty = gen_config.get('repetition_penalty', 1.2)

        try:
            student_device = self._get_student_device()

            inputs = self.student_tokenizer(
                task.input_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(student_device)

            self.student_model.eval()
            with torch.no_grad():
                outputs = self.student_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=gen_config.get('top_p', 0.9),
                    do_sample=gen_config.get('do_sample', True),
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.student_tokenizer.eos_token_id,
                    use_cache=False,
                )

            self.student_model.train()

            gen_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            return self.student_tokenizer.decode(gen_tokens, skip_special_tokens=True)

        except Exception as e:
            logger.warning(f"Student generation failed: {e}")
            self.student_model.train()
            return ""
    
    def _update_reference_model(self, tau: float = 0.95):
        """
        Soft update reference model towards current student.
        
        German trainer approach:
        - KL-aware skip: Don't update if KL is too high (unstable)
        - Dynamic tau schedule: Start with faster tracking (lower tau) during warmup
        - ref = tau * ref + (1 - tau) * student
        
        Args:
            tau: Base interpolation factor (higher = slower tracking)
        """
        if self.reference_model is None:
            return
        
        # KL-aware skip: Don't update reference if KL is too high (indicates instability)
        # This prevents the reference from tracking a potentially corrupted student
        try:
            if hasattr(self, 'last_avg_kl') and abs(float(self.last_avg_kl)) > 5.0:
                logger.warning(f"Skipping reference update due to high |KL|: {abs(float(self.last_avg_kl)):.2f}")
                return
        except Exception:
            pass
        
        with torch.no_grad():
            # Get base tau from config, falling back to argument
            base_tau = float(self.config.get('optimization', {}).get('reference_update_tau', tau))
            
            # Dynamic tau schedule during warmup (German trainer approach)
            # Start with faster tracking (tau=0.80) and gradually slow down to base_tau
            warmup_steps = int(self.config.get('rl', {}).get('kl_penalty_warmup_steps', 500))
            if warmup_steps <= 0:
                warmup_steps = 500  # Default warmup
            
            if hasattr(self, 'iteration') and int(self.iteration) < warmup_steps:
                initial_tau = 0.80  # Faster tracking early (20% student influence)
                progress = float(self.iteration) / float(warmup_steps)
                tau_val = initial_tau + (base_tau - initial_tau) * progress
            else:
                tau_val = base_tau
            
            # Soft update: ref = tau * ref + (1 - tau) * student
            for ref_param, student_param in zip(
                self.reference_model.parameters(),
                self.student_model.parameters()
            ):
                ref_param.data.mul_(tau_val).add_(student_param.data, alpha=1 - tau_val)
    
    def _blend_student_with_reference(self, alpha: float = 0.3):
        """
        Softly blend student parameters with the reference to stabilize without full reset.
        
        This is used when gradient spikes are detected to pull the student back towards
        a more stable state without completely resetting it.
        
        German trainer approach:
        - student = (1 - alpha) * student + alpha * reference
        - alpha=0.3 means 30% reference influence
        
        Args:
            alpha: Blend factor (higher = more reference influence)
        """
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return
        
        if self.reference_model is None:
            try:
                self.reference_model = self._create_reference_model()
            except Exception as e:
                logger.warning(f"Unable to create reference model for blending: {e}")
                return
        
        try:
            with torch.no_grad():
                for p_student, p_ref in zip(
                    self.student_model.parameters(),
                    self.reference_model.parameters()
                ):
                    if p_student is None or p_ref is None:
                        continue
                    # lerp_: linear interpolation in-place
                    # p_student = (1 - alpha) * p_student + alpha * p_ref
                    p_student.data.lerp_(p_ref.data, alpha)
            logger.info(f"Blended student parameters with reference (alpha={alpha:.2f}).")
        except Exception as e:
            logger.warning(f"Failed to blend student with reference: {e}")
    
    def _restore_student_from_reference(self):
        """
        Restore student model weights from the reference model.
        
        This is used as a last resort when NaN/Inf gradients are detected,
        effectively rolling back the student to the last known good state.
        
        German trainer approach: Full state dict copy from reference to student.
        """
        if self.reference_model is None:
            logger.warning("Reference model is not available; cannot restore student.")
            return
        
        try:
            self.student_model.load_state_dict(
                self.reference_model.state_dict(), 
                strict=False
            )
            self.student_model.train()
            logger.info("Student model restored from reference.")
        except Exception as e:
            logger.warning(f"Failed to restore student from reference: {e}")
    
    def _adjust_learning_rate(self, factor: float = 0.5, min_lr: float = 1e-7):
        """
        Reduce optimizer learning rate multiplicatively, bounded by min_lr.
        
        Used when gradient instability is detected to allow training to continue
        with more conservative updates.
        
        Args:
            factor: Multiplicative factor (e.g., 0.7 reduces LR by 30%)
            min_lr: Minimum learning rate to prevent LR from becoming too small
        """
        try:
            for group in self.optimizer.param_groups:
                old_lr = float(group.get('lr', 0.0))
                new_lr = max(min_lr, old_lr * factor)
                if new_lr < old_lr:
                    group['lr'] = new_lr
            logger.info(f"Reduced learning rate after instability (factor={factor:.3f}).")
        except Exception as e:
            logger.warning(f"Failed to adjust learning rate: {e}")
    
    def _compute_policy_loss(
        self,
        tasks: List[Task],
        solutions: List[str],
        advantages: List[float],
        monolingual_corpora: Dict[str, Dataset],
        use_autocast: bool = True,
        grad_accum_steps: int = 1
    ) -> float:
        """
        Compute GRPO policy loss with KL regularization.
        MEMORY OPTIMIZED: Performs backward pass incrementally to avoid storing full graph.
        Returns the total loss value (float).
        """
        
        # Early validation
        if not tasks or not solutions or not advantages:
            return 0.0

        # Parameters
        kl_coef = self.current_kl_coef
        pt_weight = float(self.config.get('rl', {}).get('pretraining_weight', 0.1))
        max_kl = float(self.config.get('rl', {}).get('max_kl_divergence', 5.0))
        entropy_weight = float(self.config.get('rl', {}).get('entropy_penalty_weight', 0.01))
        min_response_tokens = int(self.config.get('rl', {}).get('min_response_tokens', 5))
        max_length = self.config.get('rl', {}).get('max_length', 512)
        student_device = self.student_model.device

        # Track total loss for reporting
        total_loss_val = 0.0
        
        # Stats tracking
        valid_sequences = 0
        total_kl = 0.0
        total_response_tokens = 0
        
        # Use policy_chunk_size from config for memory management (SAME AS GERMAN TRAINER)
        # This two-level loop structure is critical for memory efficiency
        batch_size = len(tasks)
        chunk_size = self.config.get('optimization', {}).get('policy_chunk_size', 4)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            chunk_size = min(4, batch_size)  # Default to 4 for memory safety
        
        # Setup mixed precision context (CRITICAL for memory efficiency - same as German trainer)
        mp_enabled = torch.cuda.is_available() and use_autocast
        if mp_enabled:
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            autocast_context = torch.autocast(device_type='cuda', dtype=autocast_dtype, enabled=True)
        else:
            autocast_context = nullcontext()
        
        # TWO-LEVEL LOOP STRUCTURE (same as German trainer - critical for memory)
        # Outer loop: process chunks of tasks
        # Inner loop: process individual tasks within chunk
        # After each chunk, we BACKWARD immediately to free graph
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)
            chunk_tasks = tasks[chunk_start:chunk_end]
            chunk_solutions = solutions[chunk_start:chunk_end]
            chunk_advantages = advantages[chunk_start:chunk_end]
            
            # Initialize chunk loss tensor (key for memory management)
            chunk_loss = torch.tensor(0.0, device=student_device, requires_grad=True)
            chunk_valid_count = 0
            
            # Clear cache at start of each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            for task, solution, advantage in zip(chunk_tasks, chunk_solutions, chunk_advantages):
                full_text = f"{task.input_text}\n{solution}"
                
                # Tokenize full sequence
                inputs = self.student_tokenizer(
                    full_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length
                ).to(student_device)
                
                # Tokenize prompt only (for masking)
                prompt_tokens = self.student_tokenizer(
                    task.input_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=max_length
                ).to(student_device)
                
                seq_len = inputs['input_ids'].shape[1]
                prompt_len = prompt_tokens['input_ids'].shape[1]
                response_len = max(0, seq_len - prompt_len)
                
                # Skip sequences with insufficient response tokens
                if response_len < min_response_tokens:
                    logger.debug(f"Skipping sequence: response too short ({response_len} < {min_response_tokens})")
                    del inputs, prompt_tokens
                    continue
                
                total_response_tokens += response_len
                
                # Create labels with prompt masking
                labels = inputs['input_ids'].clone()
                if prompt_len > 0:
                    labels[:, :prompt_len] = -100  # Mask prompt tokens
                
                # Student forward pass with mixed precision (autocast)
                self.student_model.train()
                with autocast_context:
                    student_outputs = self.student_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=labels,
                        use_cache=False
                    )
                student_logits = student_outputs.logits.float()  # Cast to float32 for stable log_softmax
                
                # Compute student log probabilities for response tokens
                log_probs = self._safe_log_softmax(student_logits)
                response_mask = (labels != -100)
                safe_labels = torch.where(response_mask, labels, torch.zeros_like(labels))
                token_logprob = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                token_logprob = token_logprob.masked_fill(~response_mask, 0.0)
                student_logprob = token_logprob.sum(dim=-1).squeeze(0)
                
                if not torch.isfinite(student_logprob):
                    logger.warning("Student logprob non-finite; skipping sequence")
                    del inputs, prompt_tokens, labels, student_outputs, student_logits
                    continue
                
                # Reference model forward pass (no gradients) - same as German trainer
                ref_logprob = student_logprob.detach()  # Default: use student as reference (zero KL)
                
                if self.reference_model is not None:
                    with torch.no_grad(), autocast_context:
                        ref_outputs = self.reference_model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=labels,
                            use_cache=False
                        )
                        ref_logits = ref_outputs.logits.float()  # Cast to float32 for stable log_softmax
                        ref_log_probs = self._safe_log_softmax(ref_logits)
                        ref_token_logprob = ref_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                        ref_token_logprob = ref_token_logprob.masked_fill(~response_mask, 0.0)
                        ref_logprob = ref_token_logprob.sum(dim=-1).squeeze(0).detach()
                        del ref_outputs, ref_logits, ref_log_probs, ref_token_logprob
                
                # Compute KL divergence with normalization by response length
                # This makes KL comparable across different response lengths
                response_token_count = response_mask.sum().float().clamp(min=1.0)
                student_logprob_norm = student_logprob / response_token_count
                ref_logprob_norm = ref_logprob / response_token_count
                
                # Raw KL (per-token average)
                kl_div_raw = student_logprob_norm - ref_logprob_norm
                
                # Adaptive clipping based on training progress
                adaptive_max_kl = float(min(max_kl, 5.0 + 5.0 * np.exp(-self.iteration / 1000)))
                kl_div_clipped = torch.clamp(kl_div_raw, min=-adaptive_max_kl, max=adaptive_max_kl)
                
                # Use SQUARED log-ratio for penalty (standard approach)
                # Squared penalty penalizes both positive and negative divergence symmetrically
                # and has smoother gradients near zero compared to abs()
                # This is more aligned with standard RLHF practice (e.g., Anthropic's approach)
                kl_penalty = kl_div_clipped ** 2
                
                if not torch.isfinite(kl_div_raw):
                    logger.warning("KL divergence non-finite; using student loss only")
                    pg_loss = 0.1 * student_outputs.loss
                    kl_div = torch.tensor(0.0, device=student_device)  # For tracking
                else:
                    # Policy gradient loss: -advantage * log_prob + kl_coef * (log_ratio)^2
                    # Using normalized log_prob for more stable gradients
                    advantage_scaled = float(np.clip(advantage, -2.0, 2.0))
                    pg_loss = -advantage_scaled * student_logprob_norm + kl_coef * kl_penalty
                    kl_div = kl_div_raw  # Store raw log-ratio for logging (can be negative)
                
                # Entropy regularization (encourage exploration)
                try:
                    if response_mask.any():
                        response_logits = student_logits.squeeze(0)[response_mask.squeeze(0)]
                        logits_max = response_logits.max(dim=-1, keepdim=True)[0]
                        logits_normalized = response_logits - logits_max
                        entropy_log_probs = self._safe_log_softmax(logits_normalized)
                        entropy_probs = torch.exp(entropy_log_probs)
                        entropy = -(entropy_probs * entropy_log_probs).sum(dim=-1)
                        
                        if torch.isfinite(entropy).all():
                            entropy_mean = entropy.mean()
                            vocab_size = student_logits.size(-1)
                            target_entropy = float(np.log(min(vocab_size, 100)) * 0.5)
                            entropy_bonus = torch.clamp(entropy_mean - target_entropy, min=-2.0, max=2.0)
                            pg_loss = pg_loss - entropy_weight * entropy_bonus
                except Exception as e:
                    logger.debug(f"Entropy calculation failed: {e}")
                
                # Clamp final loss and add to CHUNK loss (not total_loss directly)
                pg_loss = torch.clamp(pg_loss, min=-10.0, max=10.0)
                chunk_loss = chunk_loss + pg_loss
                
                # Update stats
                total_kl += float(kl_div.detach().item()) if torch.isfinite(kl_div) else 0.0
                valid_sequences += 1
                chunk_valid_count += 1
                
                # Aggressive cleanup to prevent memory accumulation (same as German trainer)
                del inputs, prompt_tokens, labels, student_outputs, student_logits
                del log_probs, safe_labels, token_logprob, student_logprob
                del ref_logprob, kl_div, kl_penalty, pg_loss
                del response_mask, response_token_count, student_logprob_norm, ref_logprob_norm
                del kl_div_raw, kl_div_clipped
            
            # After processing chunk: BACKWARD immediately
            # Scale loss by (batch_size * grad_accum) to average over the full batch
            # We use batch_size as normalization factor to stabilize gradients regardless of skipped tasks
            if chunk_valid_count > 0:
                scaled_chunk_loss = chunk_loss / (batch_size * grad_accum_steps)
                scaled_chunk_loss.backward()
                total_loss_val += chunk_loss.item() / batch_size # Log the mean loss
                del scaled_chunk_loss
                
            del chunk_loss
            
            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Store KL stats for logging
        if valid_sequences > 0:
            self.last_avg_kl = total_kl / valid_sequences
        self.last_response_token_count = total_response_tokens
        
        # Add pretraining loss to maintain language modeling
        if pt_weight > 0:
            pt_loss = self._compute_pretraining_loss(monolingual_corpora, use_autocast=use_autocast)
            if torch.isfinite(pt_loss):
                # Backward pretraining loss separately
                scaled_pt_loss = (pt_weight * pt_loss) / grad_accum_steps
                scaled_pt_loss.backward()
                total_loss_val += (pt_weight * pt_loss.item())
                del pt_loss, scaled_pt_loss
        
        # If no valid sequences and no PT loss, return None to signal skip
        if valid_sequences == 0 and pt_weight <= 0:
             return None

        return total_loss_val
    
    def _log_euroeval_accuracy(self):
        """Log per-task EuroEval + self-play accuracy and mastery state."""
        total_correct = 0
        total_count = 0

        logger.info("  EuroEval per-task accuracy:")
        for task_type, stats in sorted(self.task_accuracy.items()):
            c, t = stats["correct"], stats["total"]
            total_correct += c
            total_count += t
            mastered_flag = " [MASTERED]" if task_type in self.mastered_tasks else ""
            if t > 0:
                acc = c / t
                logger.info(f"    {task_type}: {acc:.1%} ({c}/{t}){mastered_flag}")
            else:
                logger.info(f"    {task_type}: N/A (0 samples)")

        if total_count > 0:
            overall = total_correct / total_count
            mastered_str = ", ".join(sorted(self.mastered_tasks)) or "none"
            logger.info(
                f"  Overall EuroEval accuracy: {overall:.1%} "
                f"({total_correct}/{total_count}) | "
                f"mastered=[{mastered_str}]"
            )

        # Self-play accuracy
        sp_total_correct = sum(v["correct"] for v in self.selfplay_accuracy.values())
        sp_total_count = sum(v["total"] for v in self.selfplay_accuracy.values())
        if sp_total_count > 0:
            logger.info("  Self-play per-task accuracy:")
            for task_type, stats in sorted(self.selfplay_accuracy.items()):
                c, t = stats["correct"], stats["total"]
                if t > 0:
                    logger.info(f"    sp_{task_type}: {c/t:.1%} ({c}/{t})")
            sp_overall = sp_total_correct / sp_total_count
            logger.info(
                f"  Overall self-play accuracy: {sp_overall:.1%} "
                f"({sp_total_correct}/{sp_total_count})"
            )

        # Rolling reward comparison
        euro_recent = self.reward_tracker["euroeval"]["recent"]
        sp_recent = self.reward_tracker["selfplay"]["recent"]
        if euro_recent:
            logger.info(f"  EuroEval avg reward (recent {len(euro_recent)}): {sum(euro_recent)/len(euro_recent):.4f}")
        if sp_recent:
            logger.info(f"  Self-play avg reward (recent {len(sp_recent)}): {sum(sp_recent)/len(sp_recent):.4f}")

        if self.selfplay_enabled:
            logger.info(
                f"  Self-play: enabled | ratio={self.selfplay_ratio:.2f} | "
                f"eligible={sorted(self.selfplay_eligible_mastered)}"
            )

        # Save accuracy to CSV for tracking
        csv_file = self.metrics_dir / 'euroeval_task_accuracy.csv'
        write_header = not csv_file.exists()
        row = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'overall_accuracy': total_correct / max(total_count, 1),
            'total_samples': total_count,
            'n_mastered': len(self.mastered_tasks),
            'mastered_tasks': ";".join(sorted(self.mastered_tasks)),
            'selfplay_enabled': self.selfplay_enabled,
            'selfplay_ratio': self.selfplay_ratio,
            'selfplay_overall_acc': sp_total_correct / max(sp_total_count, 1),
            'selfplay_total_samples': sp_total_count,
        }
        for task_type, stats in self.task_accuracy.items():
            row[f'{task_type}_accuracy'] = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            row[f'{task_type}_count'] = stats["total"]
        for task_type, stats in self.selfplay_accuracy.items():
            row[f'sp_{task_type}_accuracy'] = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            row[f'sp_{task_type}_count'] = stats["total"]

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'checkpoint-{iteration}'
        checkpoint_path.mkdir(exist_ok=True)
        
        self.student_model.save_pretrained(checkpoint_path)
        self.student_tokenizer.save_pretrained(checkpoint_path)
        
        logger.info(f"Saved checkpoint at iteration {iteration}")
