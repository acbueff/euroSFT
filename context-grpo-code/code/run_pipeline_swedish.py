"""
FRÓÐI Training Pipeline - SWEDISH ADAPTATION

This script orchestrates the complete FRÓÐI training pipeline for Swedish,
implementing a two-stage knowledge distillation followed by GRPO self-play RL.

Phase 1: Knowledge Distillation (Two Stages)
  - Stage 1 (CPT): Continued Pre-Training for Swedish fluency
  - Stage 2 (SFT): Supervised Fine-Tuning for reasoning in Swedish

Phase 2: GRPO Self-Play RL
  - Uses Mistral-Small-3.1-24B as judge (best Swedish model on EuroEval)
"""

import os
import argparse

# Set PyTorch CUDA memory allocation to help with fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import yaml
import logging
import torch
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from transformers.utils import logging as hf_logging

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set seed for reproducibility across all random sources.
    
    This ensures consistent behavior across:
    - Python's random module (used for sampling, shuffling)
    - NumPy's random (used for advantage computation, etc.)
    - PyTorch (model initialization, dropout, etc.)
    - CUDA operations
    
    Args:
        seed: Integer seed value (default in config is 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # Note: Setting these to True can impact performance but ensures full determinism
        # Uncomment if you need fully deterministic behavior:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed} for reproducibility")

# Suppress repetitive generation warnings
warnings.filterwarnings(
    "ignore",
    message=".*generation flags are not valid and may be ignored.*",
)
hf_logging.set_verbosity_error()


def setup_logging(log_file: str):
    """Setup logging configuration"""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_configurations():
    """Load Swedish-specific configuration files"""
    configs = {}
    
    config_files = {
        'default': 'configs/default_config_swedish.yaml',
        'model': 'configs/model_config_swedish.yaml',
        'training': 'configs/training_config_swedish.yaml'
    }
    
    for config_name, config_path in config_files.items():
        try:
            with open(config_path, 'r') as f:
                configs[config_name] = yaml.safe_load(f)
            logger.info(f"Loaded {config_name} configuration from {config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise
    
    return configs


def load_model(model_id: str, quantization: bool = False, device_index: int = None, 
               quantization_config: dict = None, token: str = None, use_lora: bool = False,
               lora_config_dict: dict = None):
    """Load a model with optional quantization. Supports both HF model IDs and local paths."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # Check if it's a local path
    is_local = model_id.startswith('/') or model_id.startswith('./')
    logger.info(f"Loading model: {model_id} ({'local path' if is_local else 'HuggingFace'})")
    
    # Tokenizer - don't pass token for local paths
    tokenizer_kwargs = {"trust_remote_code": True}
    if not is_local and token:
        tokenizer_kwargs["token"] = token
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
    }
    # Only pass token for HuggingFace models, not local paths
    if not is_local and token:
        model_kwargs["token"] = token
    
    # Quantization config
    if quantization and quantization_config:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.get('load_in_4bit', True),
            load_in_8bit=quantization_config.get('load_in_8bit', False),
            bnb_4bit_compute_dtype=torch.bfloat16 if quantization_config.get('bnb_4bit_compute_dtype') == 'bfloat16' else torch.float16,
            bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', True),
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    # Device mapping
    if device_index is not None:
        model_kwargs["device_map"] = {"": device_index}
    else:
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    logger.info(f"Model loaded: {model_id}")
    return model, tokenizer


def load_swedish_corpora(config: dict) -> dict:
    """Load Swedish and English monolingual corpora"""
    from datasets import Dataset
    
    data_dir = Path(config.get('data', {}).get('local_data_dir', 'frodi_data'))
    corpora = {}
    
    # Load Swedish data
    swedish_file = data_dir / 'swedish_monolingual.txt'
    if swedish_file.exists():
        with open(swedish_file, 'r', encoding='utf-8') as f:
            swedish_texts = [line.strip() for line in f if line.strip()]
        corpora['sv'] = Dataset.from_dict({
            'text': swedish_texts,
            'language': ['sv'] * len(swedish_texts)
        })
        logger.info(f"Loaded {len(swedish_texts)} Swedish texts")
    else:
        logger.warning(f"Swedish data file not found: {swedish_file}")
        corpora['sv'] = Dataset.from_dict({'text': [], 'language': []})
    
    # Load English anchor data
    english_file = data_dir / 'english_monolingual.txt'
    if english_file.exists():
        with open(english_file, 'r', encoding='utf-8') as f:
            english_texts = [line.strip() for line in f if line.strip()]
        corpora['en'] = Dataset.from_dict({
            'text': english_texts,
            'language': ['en'] * len(english_texts)
        })
        logger.info(f"Loaded {len(english_texts)} English anchor texts")
    else:
        corpora['en'] = Dataset.from_dict({'text': [], 'language': []})
    
    return corpora


def load_seed_tasks(file_path: str):
    """Load seed tasks from JSONL file"""
    from datasets import Dataset
    
    if not os.path.exists(file_path):
        logger.warning(f"Seed tasks file not found: {file_path}")
        return Dataset.from_dict({
            'task_type': [], 'source_lang': [], 'target_lang': [],
            'source_text': [], 'target_text': [], 'difficulty': []
        })
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    logger.info(f"Loaded {len(data)} seed tasks from {file_path}")
    return Dataset.from_list(data)


def run_knowledge_distillation(configs: dict, models: dict):
    """
    Run Phase 1: Knowledge Distillation (Two-Stage Process)
    
    Stage 1 (CPT): Continued Pre-Training for Swedish fluency
    Stage 2 (SFT): Supervised Fine-Tuning for reasoning
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: KNOWLEDGE DISTILLATION FOR SWEDISH")
    logger.info("=" * 60)
    
    from src.core.frodi_trainer_swedish import SwedishKnowledgeDistillation
    
    distill_config = configs['training'].get('distillation', {})
    model_config = configs['model']
    
    # Initialize the Swedish KD trainer
    kd_trainer = SwedishKnowledgeDistillation(
        student_model=models['student_model'],
        student_tokenizer=models['student_tokenizer'],
        teacher_model=models['teacher_model'],
        teacher_tokenizer=models['teacher_tokenizer'],
        config=distill_config,
        model_config=model_config,
        device=models['student_model'].device
    )
    
    # Load corpora
    corpora = load_swedish_corpora(configs['default'])
    seed_tasks = load_seed_tasks(configs['default']['data']['seed_tasks_path'])
    
    # Stage 1: CPT for Swedish fluency
    logger.info("-" * 40)
    logger.info("Stage 1: Continued Pre-Training (CPT)")
    logger.info("-" * 40)
    
    if distill_config.get('stage1_cpt', {}).get('enabled', True):
        kd_trainer.run_cpt_stage(corpora)
    else:
        logger.info("Stage 1 (CPT) disabled in config, skipping...")
    
    # Stage 2: SFT for reasoning
    logger.info("-" * 40)
    logger.info("Stage 2: Supervised Fine-Tuning (SFT)")
    logger.info("-" * 40)
    
    if distill_config.get('stage2_sft', {}).get('enabled', True):
        kd_trainer.run_sft_stage(seed_tasks, corpora)
    else:
        logger.info("Stage 2 (SFT) disabled in config, skipping...")
    
    # Save the distilled model
    output_dir = Path(distill_config.get('output_dir', 'outputs/distillation_swedish'))
    model_name = distill_config.get('save_model_name', 'frodi-swedish-student-kd')
    save_path = output_dir / model_name
    
    logger.info(f"Saving distilled model to: {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)
    models['student_model'].save_pretrained(save_path)
    models['student_tokenizer'].save_pretrained(save_path)
    
    # Save training metadata
    metadata = {
        'model_name': model_name,
        'teacher_model': model_config.get('teacher'),
        'student_model': model_config.get('student'),
        'training_date': datetime.now().isoformat(),
        'config': distill_config
    }
    with open(save_path / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Knowledge Distillation completed!")
    return save_path


def run_grpo_self_play(configs: dict, models: dict, kd_model_path: str = None, resume_checkpoint: str = None):
    """
    Run Phase 2: GRPO Self-Play RL with Teacher Feedback
    
    Uses Mistral-Small-3.1-24B as judge for Swedish tasks
    
    Args:
        configs: Configuration dictionaries
        models: Dictionary of loaded models
        kd_model_path: Path to KD model (used if not resuming)
        resume_checkpoint: Path to checkpoint to resume from (e.g., iter_300 checkpoint)
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: GRPO SELF-PLAY RL FOR SWEDISH")
    logger.info("=" * 60)
    
    from src.core.frodi_trainer_swedish import FrodiTrainerSwedish
    
    # Determine which model path to use
    # Priority: resume_checkpoint > kd_model_path
    model_load_path = None
    start_iteration = 0
    
    if resume_checkpoint and Path(resume_checkpoint).exists():
        model_load_path = resume_checkpoint
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        
        # Try to read the iteration from checkpoint metadata
        metadata_file = Path(resume_checkpoint) / 'checkpoint_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                start_iteration = metadata.get('iteration', 0)
                logger.info(f"Checkpoint iteration: {start_iteration}")
        else:
            # Try to extract iteration from path name (e.g., iter_300)
            checkpoint_name = Path(resume_checkpoint).name
            if checkpoint_name.startswith('iter_'):
                try:
                    start_iteration = int(checkpoint_name.split('_')[1])
                    logger.info(f"Extracted iteration {start_iteration} from checkpoint path")
                except (ValueError, IndexError):
                    logger.warning(f"Could not extract iteration from checkpoint name: {checkpoint_name}")
    elif kd_model_path and Path(kd_model_path).exists():
        model_load_path = kd_model_path
        logger.info(f"Loading KD model from: {kd_model_path}")
    
    # Load student model from the determined path
    if model_load_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Force student model to single GPU (cuda:0) - it's small enough (1.7B)
        # Using device_map="auto" distributes across GPUs causing device mismatch errors
        models['student_model'] = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},  # Force to cuda:0
            trust_remote_code=True
        )
        models['student_tokenizer'] = AutoTokenizer.from_pretrained(
            model_load_path,
            trust_remote_code=True
        )
    
    # Determine device for student model
    # For models loaded with device_map, we need to get the actual device
    try:
        student_device = next(models['student_model'].parameters()).device
    except StopIteration:
        student_device = torch.device("cuda:0")
    logger.info(f"Student model device: {student_device}")
    
    # Initialize the Swedish GRPO trainer
    trainer = FrodiTrainerSwedish(
        student_model=models['student_model'],
        student_tokenizer=models['student_tokenizer'],
        teacher_model=models['judge_model'],  # Use judge model for GRPO
        teacher_tokenizer=models['judge_tokenizer'],
        config=configs['training'],
        device=student_device
    )
    
    # Load corpora
    corpora = load_swedish_corpora(configs['default'])
    
    # Run GRPO training
    rl_config = configs['training'].get('rl', {})
    n_iterations = rl_config.get('iterations', 1000)
    
    # Pass start_iteration for resumption
    trainer.train_self_play_rl(
        monolingual_corpora=corpora,
        n_iterations=n_iterations,
        start_iteration=start_iteration
    )
    
    # Save final model
    output_dir = Path(rl_config.get('output_dir', 'outputs/self_play_rl_swedish'))
    model_name = configs['model'].get('output', {}).get('phase2_model_name', 'frodi-swedish-qwen3-1.7b-grpo')
    save_path = output_dir / model_name
    
    logger.info(f"Saving GRPO model to: {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)
    models['student_model'].save_pretrained(save_path)
    models['student_tokenizer'].save_pretrained(save_path)
    
    logger.info("GRPO Self-Play completed!")
    return save_path


def main():
    """Main entry point for Swedish FRÓÐI training"""
    parser = argparse.ArgumentParser(description='FRÓÐI Swedish Training Pipeline')
    parser.add_argument('--phase', type=str, choices=['kd', 'grpo', 'all'], default='all',
                        help='Which phase to run: kd (knowledge distillation), grpo (self-play RL), or all')
    parser.add_argument('--kd-model-path', type=str, default=None,
                        help='Path to pre-trained KD model (for GRPO phase)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint to resume GRPO training from (e.g., eval_checkpoints/iter_300)')
    args = parser.parse_args()
    
    try:
        # Load configurations
        configs = load_configurations()
        
        # Set random seed for reproducibility (MUST be done early, before any random operations)
        seed = configs['default'].get('seed', 42)
        set_seed(seed)
        
        # Setup logging
        log_file = configs['default']['logging']['log_file']
        setup_logging(log_file)
        
        logger.info("=" * 60)
        logger.info("FRÓÐI SWEDISH TRAINING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Phase: {args.phase}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Date: {datetime.now().isoformat()}")
        
        # Setup output directories
        for key in ['distillation', 'rl']:
            if key in configs['training']:
                output_dir = configs['training'][key].get('output_dir')
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get HuggingFace token
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            logger.info("Using HuggingFace token from environment")
        else:
            logger.warning("No HuggingFace token found")
        
        models = {}
        model_config = configs['model']
        
        # Load models based on phase
        if args.phase in ['kd', 'all']:
            logger.info("Loading models for Knowledge Distillation...")
            
            # Load student model (Qwen3-1.7B)
            logger.info(f"Loading student model: {model_config['student']}")
            models['student_model'], models['student_tokenizer'] = load_model(
                model_config['student'],
                quantization=False,
                device_index=0 if torch.cuda.device_count() >= 1 else None,
                token=hf_token
            )
            
            # Load teacher model (Qwen3-32B with quantization)
            logger.info(f"Loading teacher model: {model_config['teacher']}")
            models['teacher_model'], models['teacher_tokenizer'] = load_model(
                model_config['teacher'],
                quantization=True,
                quantization_config=model_config.get('teacher_quantization', {}),
                device_index=1 if torch.cuda.device_count() >= 2 else None,
                token=hf_token
            )
        
        if args.phase in ['grpo', 'all']:
            # Load judge model for GRPO (Mistral-Small-3.1-24B)
            logger.info(f"Loading judge model: {model_config['judge']}")
            models['judge_model'], models['judge_tokenizer'] = load_model(
                model_config['judge'],
                quantization=True,
                quantization_config=model_config.get('judge_quantization', {}),
                device_index=1 if torch.cuda.device_count() >= 2 else None,
                token=hf_token
            )
        
        # Enable TF32 for better throughput
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled TF32 and cuDNN benchmark")
        
        # Run phases
        kd_model_path = args.kd_model_path
        
        if args.phase in ['kd', 'all']:
            kd_model_path = run_knowledge_distillation(configs, models)
        
        if args.phase in ['grpo', 'all']:
            run_grpo_self_play(configs, models, kd_model_path, resume_checkpoint=args.resume_from)
        
        logger.info("=" * 60)
        logger.info("FRÓÐI SWEDISH TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()

