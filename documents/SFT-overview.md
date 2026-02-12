## EuroEval Swedish Training Data Summary

| Dataset                | Task Type                | Samples         | Labels/Notes                                      |
| ---------------------- | ------------------------ | --------------- | ------------------------------------------------- |
| **swerec**       | Sentiment Classification | 1,024           | positive: 481, negative: 415, neutral: 128        |
| **scala-sv**     | Linguistic Acceptability | 1,024           | correct: 512, incorrect: 512 (perfectly balanced) |
| **suc3**         | Named Entity Recognition | 1,024           | 340 samples have entities (PER/ORG/LOC/MISC)      |
| **scandiqa-sv**  | Reading Comprehension    | 1,024           | Extractive QA, avg answer ~11 chars               |
| **hellaswag-sv** | Commonsense Reasoning    | 1,024           | 4-way MC (a/b/c/d), roughly balanced              |
| **mmlu-sv**      | Knowledge (MMLU)         | 1,024           | 4-way MC, 57 categories                           |
| **swedn**        | Summarization            | 1,024           | News summarization (domestic/economy/sports)      |
| **TOTAL**        |                          | **7,168** | **~7.4 MB on disk**                         |

## SFT Baseline Recommendations for Qwen3-1.7B

### Is this a good approach?

Yes - this is a solid baseline comparison. SFT on the same EuroEval training data as your GRPO experiment gives a clean apples-to-apples comparison: same data, same base model, different training method. This directly isolates the effect of GRPO vs supervised learning.

### Data Formatting

Each task needs to be converted to instruction-response pairs. The format depends on the task:

* **Classification tasks** (swerec, scala-sv, hellaswag-sv, mmlu-sv): Input is the text/question, target is the label. These are straightforward.
* **NER** (suc3): Input is the text, target is structured entity output.
* **Reading comprehension** (scandiqa-sv): Input is context + question, target is the answer span.
* **Summarization** (swedn): Input is the article, target is the headline/summary.

You should use the **same system prompts** that GRPO uses (from `_build_swedish_system_prompts` in the trainer) so the comparison is fair.

### Training Specifics

**Epochs: 3-5** (recommended:  **3 with early stopping** )

Rationale: 7,168 samples is small for a 1.7B model. With 3 epochs:

* Total training examples seen: ~21.5k
* Risk of overfitting is real but manageable with proper LR and weight decay
* More than 5 epochs will almost certainly overfit

**Hyperparameters:**

| Parameter             | Recommended Value        | Rationale                                                           |
| --------------------- | ------------------------ | ------------------------------------------------------------------- |
| Learning rate         | **2e-5**           | Standard SFT LR for small models; your SFT config already uses this |
| Batch size            | 4-8                      | Depends on GPU memory                                               |
| Gradient accumulation | 8-16                     | Effective batch 32-64                                               |
| Max sequence length   | 512                      | Sufficient for all tasks (answers are short)                        |
| Warmup ratio          | 0.1                      | ~60-100 warmup steps                                                |
| Weight decay          | 0.01                     | Standard                                                            |
| LR scheduler          | Linear decay with warmup | Standard                                                            |
| Precision             | bf16                     | Standard for Qwen3                                                  |

**What NOT to do:**

* Don't use a teacher model - this is pure SFT, no distillation
* Don't use LoRA - full fine-tuning for fair comparison to your GRPO which also updates full weights
* Don't upsample any task - keep uniform sampling across all 7 datasets to match GRPO's access to the same data

### Data Transfer

The entire dataset is only **7.4 MB** - trivially small to copy:

```bash
# From this HPC to your laptop
scp -r x_anbue@berzelius1.nsc.liu.se:/proj/berzelius-aiics-real/users/x_anbue/euroeval/train_sets/sv/ ~/euroeval_sv_train/

# From laptop to other HPC
scp -r ~/euroeval_sv_train/ user@other-hpc:/path/to/data/
```

### Key Context for the LLM on the Other HPC

When you pass this conversation, the key facts are:

1. **Goal** : SFT baseline to compare against GRPO on EuroEval Swedish
2. **Base model** : `Qwen/Qwen3-1.7B`
3. **Data** : 7 JSON files, 1,024 samples each, 7,168 total, format is `{"samples": [...]}`
4. **Each task has a different structure** - the data loading + prompt formatting code in [frodi_trainer_swedish_euro.py](vscode-webview://09uahqsnq2utd95glfhtkfadmqnkd72n2co3bjbndl6oda9md8na/src/core/frodi_trainer_swedish_euro.py) (lines 93-174) shows exactly how each task type maps to prompts and expected outputs
5. **Evaluation** : The trained model should be evaluated on EuroEval Swedish **validation/test split** (not train) using the EuroEval benchmark tool
6. **Fair comparison** : Use the same system prompts from the GRPO trainer so the model sees identical instruction formatting
