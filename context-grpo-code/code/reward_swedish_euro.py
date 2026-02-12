"""
FRÓÐI Reward Model - SWEDISH EUROEVAL ADAPTATION

Binary Flexible Feedback Reward model for Swedish EuroEval + self-play tasks.

Supports EuroEval benchmark tasks (sentiment, NER, acceptability, reading
comprehension, common-sense reasoning, knowledge) and self-play tasks generated
from FineWeb corpus text for the same task types.

Smart reward routing:
  - EuroEval + deterministic type (sentiment, acceptability, commonsense, knowledge):
    String matching against gold answer (no LLM judge needed).
  - EuroEval + generative type (NER, reading comprehension):
    LLM judge with gold answer in context.
  - Self-play (any type):
    LLM judge with a correct example from EuroEval in context.
"""

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ..behavior_logger import BehaviorLogger
except Exception:
    BehaviorLogger = None


logger = logging.getLogger(__name__)

# EuroEval task types where gold answer is a single word/letter and can be
# checked via deterministic string matching (no LLM judge needed).
DETERMINISTIC_TASK_TYPES = {"sentiment", "acceptability", "commonsense", "knowledge"}

# Deterministic reward magnitude (slightly dampened vs ±1.0 to stay in scale
# with normalized LLM judge rewards).
DETERMINISTIC_REWARD_MAGNITUDE = 0.8


@dataclass
class BinaryPrinciple:
    """Definition of a binary principle used for flexible feedback."""

    name: str
    description: str
    positive_guidance: str
    negative_guidance: str

    def locale_instructions(self, locale: str = "en") -> str:
        loc = locale.lower()
        if loc.startswith("sv"):
            return (
                "Besvara följande fråga strikt med YES eller NO. "
                "Bedöm endast om principen är uppfylld."
            )
        return (
            "Answer the question strictly with YES or NO. "
            "Judge only whether the principle is satisfied."
        )


def logit_diff(log_probs: torch.Tensor, yes_token_id: int, no_token_id: int) -> torch.Tensor:
    yes_lp = log_probs[..., yes_token_id]
    no_lp = log_probs[..., no_token_id]
    return yes_lp - no_lp


def safe_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=-10.0, neginf=-10.0, posinf=10.0)
    logits = torch.clamp(logits, min=-50.0, max=50.0)
    return F.log_softmax(logits, dim=-1)


class BinaryFlexibleFeedbackRewardSwedish:
    """Binary flexible feedback reward model for Swedish tasks."""

    def __init__(
        self,
        teacher_model,
        teacher_tokenizer,
        device,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        norm_momentum: float = 0.1,
    ):
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.device = device
        self.behavior_logger: Optional[BehaviorLogger] = None
        self.nan_debug = False
        self.clip_min = min(clip_min, clip_max)
        self.clip_max = max(clip_min, clip_max)
        self.norm_momentum = max(norm_momentum, 1e-3)
        self.running_mean = 0.0
        self.running_var = 1.0

        # Swedish grammar principle (optional reward dimension)
        self.grammar_principle = BinaryPrinciple(
            name="grammar",
            description=(
                "Studentens utdata måste vara grammatiskt korrekt, välformad "
                "och naturlig på det begärda målspråket."
            ),
            positive_guidance=(
                "Belöna svar som följer grammatik och syntax för målspråket, "
                "med korrekt morfologi, kongruens, interpunktion och "
                "idiomatisk formulering."
            ),
            negative_guidance=(
                "Straffa grammatiska fel, felaktig ordföljd, klumpig eller "
                "bokstavlig formulering, kongruensfel eller trasiga meningar."
            ),
        )

        # === EuroEval task-specific principles ===

        self.sentiment_principle = BinaryPrinciple(
            name="sentiment_accuracy",
            description=(
                "Studentens sentimentklassificering måste matcha det korrekta "
                "sentimentet för den givna texten."
            ),
            positive_guidance=(
                "Belöna om studentens svar korrekt identifierar sentimentet "
                "(positiv, negativ, neutral eller blandad) som matchar facit."
            ),
            negative_guidance=(
                "Straffa om studentens klassificering inte matchar det korrekta "
                "sentimentet, eller om svaret är otydligt."
            ),
        )

        self.acceptability_principle = BinaryPrinciple(
            name="acceptability_accuracy",
            description=(
                "Studentens bedömning av grammatisk acceptabilitet måste matcha "
                "det korrekta svaret för den givna meningen."
            ),
            positive_guidance=(
                "Belöna om studenten korrekt identifierar meningen som "
                "grammatiskt korrekt eller inkorrekt, i enlighet med facit."
            ),
            negative_guidance=(
                "Straffa om studentens bedömning inte matchar det korrekta svaret, "
                "eller om svaret är otydligt eller tvetydigt."
            ),
        )

        self.ner_principle = BinaryPrinciple(
            name="ner_accuracy",
            description=(
                "Studenten måste korrekt identifiera de namngivna entiteterna "
                "(personer, platser, organisationer) i texten."
            ),
            positive_guidance=(
                "Belöna om studenten identifierar de viktigaste entiteterna "
                "med rätt typ (PER, LOC, ORG), även om formateringen varierar."
            ),
            negative_guidance=(
                "Straffa om studenten missar viktiga entiteter, identifierar "
                "felaktiga entiteter, eller anger fel entitetstyp."
            ),
        )

        self.reading_comprehension_principle = BinaryPrinciple(
            name="reading_comprehension_accuracy",
            description=(
                "Studentens svar på frågan måste vara korrekt och stödjas av "
                "den givna kontexten."
            ),
            positive_guidance=(
                "Belöna om studentens svar innehåller korrekt information som "
                "matchar eller är ekvivalent med facit, baserat på kontexten."
            ),
            negative_guidance=(
                "Straffa om svaret är felaktigt, inte matchar facit, eller "
                "innehåller information som inte finns i kontexten."
            ),
        )

        self.commonsense_principle = BinaryPrinciple(
            name="commonsense_accuracy",
            description=(
                "Studenten måste välja det korrekta svarsalternativet för "
                "frågan om sunt förnuft."
            ),
            positive_guidance=(
                "Belöna om studentens val av svarsalternativ (a, b, c eller d) "
                "matchar det korrekta svaret."
            ),
            negative_guidance=(
                "Straffa om studenten väljer fel svarsalternativ eller ger "
                "ett otydligt svar."
            ),
        )

        self.knowledge_principle = BinaryPrinciple(
            name="knowledge_accuracy",
            description=(
                "Studenten måste välja det korrekta svarsalternativet för "
                "kunskapsfrågan."
            ),
            positive_guidance=(
                "Belöna om studentens val av svarsalternativ (a, b, c eller d) "
                "matchar det korrekta svaret."
            ),
            negative_guidance=(
                "Straffa om studenten väljer fel svarsalternativ eller ger "
                "ett otydligt svar."
            ),
        )

        # Map task types to principles for EuroEval tasks
        self.euroeval_principles = {
            "sentiment": self.sentiment_principle,
            "acceptability": self.acceptability_principle,
            "ner": self.ner_principle,
            "reading_comprehension": self.reading_comprehension_principle,
            "commonsense": self.commonsense_principle,
            "knowledge": self.knowledge_principle,
        }

        # Token IDs for YES/NO
        special_tokens = {
            "YES": self.teacher_tokenizer.encode(" YES", add_special_tokens=False),
            "NO": self.teacher_tokenizer.encode(" NO", add_special_tokens=False),
        }
        try:
            self.yes_token_id = special_tokens["YES"][0]
            self.no_token_id = special_tokens["NO"][0]
        except Exception as exc:
            raise ValueError("Tokenizer must encode ' YES' and ' NO'") from exc

    def set_behavior_logger(self, logger_instance: BehaviorLogger):
        self.behavior_logger = logger_instance

    def enable_nan_debug(self, enabled: bool = True):
        self.nan_debug = bool(enabled)

    def _infer_locale(self, task_input: str, student_output: str) -> str:
        """Infer locale from text - Swedish detection"""
        text = f"{task_input} {student_output}".lower()
        
        # Swedish-specific characters and words
        if re.search(r"[åäö]", text) or " svenska" in text or " swedish" in text:
            return "sv"
        if " translate from sv" in text or "to sv" in text:
            return "sv"
        if " översätt" in text or " svensk" in text:
            return "sv"
        return "en"

    def _teacher_device(self) -> torch.device:
        try:
            emb = self.teacher_model.get_input_embeddings()
            if emb is not None:
                return emb.weight.device
        except Exception:
            pass
        try:
            return next(self.teacher_model.parameters()).device
        except Exception:
            return self.device

    def _build_rating_messages(
        self,
        task_type: str,
        task_input: str,
        student_output: str,
        principle: BinaryPrinciple,
        ground_truth: str = "",
        correct_example: str = "",
    ) -> List[Dict[str, str]]:
        """Build rating messages for teacher judge"""
        locale = self._infer_locale(task_input, student_output)
        
        if locale.startswith("sv"):
            system = (
                "Du är en noggrann bedömare. Svara strikt med YES eller NO."
            )
        else:
            system = "You are a meticulous judge. Answer strictly with YES or NO."

        principle_block = (
            f"Principle: {principle.name}\n"
            f"What to reward: {principle.positive_guidance}\n"
            f"What to penalize: {principle.negative_guidance}\n"
            f"Principle explanation: {principle.description}"
        )

        if principle.name == "grammar":
            user = (
                "Assess whether the student's response is grammatically correct and natural in the target language.\n"
                f"Prompt:\n{task_input}\n\n"
                f"Student response:\n{student_output}\n\n"
                f"{principle_block}\n\n"
                "Respond with YES if the response is grammatically correct and fluent, otherwise NO."
            )
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        # === EuroEval task types (ground-truth-aware) ===
        if task_type == "sentiment":
            if ground_truth:
                user = (
                    "Assess whether the student correctly classified the sentiment of the text.\n"
                    f"Text:\n{task_input}\n\n"
                    f"Correct sentiment: {ground_truth}\n\n"
                    f"Student response:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the student's classification matches the correct sentiment, otherwise NO."
                )
            else:
                example_block = f"\nExempel på korrekt bedömning:\n{correct_example}\n\n" if correct_example else ""
                user = (
                    "Assess whether the student's sentiment classification is plausible for the text.\n"
                    f"Text:\n{task_input}\n\n"
                    f"{example_block}"
                    f"Student response:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the classification is reasonable and well-justified, otherwise NO."
                )
        elif task_type == "acceptability":
            if ground_truth:
                user = (
                    "Assess whether the student correctly judged the grammatical acceptability of the sentence.\n"
                    f"Sentence:\n{task_input}\n\n"
                    f"Correct judgment: {ground_truth}\n\n"
                    f"Student response:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the student's judgment matches the correct answer, otherwise NO."
                )
            else:
                example_block = f"\nExempel på korrekt bedömning:\n{correct_example}\n\n" if correct_example else ""
                user = (
                    "Assess whether the student correctly judged the grammatical acceptability of the sentence.\n"
                    f"Sentence:\n{task_input}\n\n"
                    f"{example_block}"
                    f"Student response:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the student's judgment is linguistically sound, otherwise NO."
                )
        elif task_type == "ner":
            if ground_truth:
                user = (
                    "Assess whether the student correctly identified the named entities in the text.\n"
                    f"Text:\n{task_input}\n\n"
                    f"Correct entities: {ground_truth}\n\n"
                    f"Student response:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the student identified the key entities correctly, otherwise NO."
                )
            else:
                example_block = f"\nExempel på korrekt bedömning:\n{correct_example}\n\n" if correct_example else ""
                user = (
                    "Assess whether the student plausibly identified named entities in the text.\n"
                    f"Text:\n{task_input}\n\n"
                    f"{example_block}"
                    f"Student response:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the entities listed are real and correctly typed (PER/LOC/ORG), otherwise NO."
                )
        elif task_type == "reading_comprehension":
            if ground_truth:
                user = (
                    "Assess whether the student correctly answered the question based on the context.\n"
                    f"Question and context:\n{task_input}\n\n"
                    f"Correct answer: {ground_truth}\n\n"
                    f"Student answer:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the student's answer matches or is equivalent to the correct answer, otherwise NO."
                )
            else:
                example_block = f"\nExempel på korrekt bedömning:\n{correct_example}\n\n" if correct_example else ""
                user = (
                    "Assess whether the student's answer is accurate and supported by the context.\n"
                    f"Question and context:\n{task_input}\n\n"
                    f"{example_block}"
                    f"Student answer:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the answer is factually correct and grounded in the context, otherwise NO."
                )
        elif task_type in ("commonsense", "knowledge"):
            if ground_truth:
                user = (
                    "Assess whether the student selected the correct answer option.\n"
                    f"Question:\n{task_input}\n\n"
                    f"Correct answer: {ground_truth}\n\n"
                    f"Student answer:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the student selected the correct option, otherwise NO."
                )
            else:
                example_block = f"\nExempel på korrekt bedömning:\n{correct_example}\n\n" if correct_example else ""
                user = (
                    "Assess whether the student's response is factually accurate and demonstrates sound reasoning.\n"
                    f"Question:\n{task_input}\n\n"
                    f"{example_block}"
                    f"Student answer:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the response is coherent and well-reasoned, otherwise NO."
                )
        else:
            # Fallback for any unrecognised task type
            user = (
                "Assess whether the student answer satisfies the principle.\n"
                f"Prompt:\n{task_input}\n\n"
                f"Student answer:\n{student_output}\n\n"
                f"{principle_block}\n\n"
                "Respond with YES if satisfied, otherwise NO."
            )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _score_yes_no(
        self,
        prompt_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.teacher_model(**prompt_inputs)
        logits = outputs.logits
        last_token_logits = logits[:, -1, :]
        log_probs = safe_log_softmax(last_token_logits.float())
        yes_lp = log_probs[:, self.yes_token_id]
        no_lp = log_probs[:, self.no_token_id]
        return yes_lp, no_lp

    def _prepare_prompt_inputs(
        self,
        task_type: str,
        task_input: str,
        student_output: str,
        principle: BinaryPrinciple,
        ground_truth: str = "",
        correct_example: str = "",
    ) -> Dict[str, torch.Tensor]:
        device = self._teacher_device()
        messages = self._build_rating_messages(
            task_type, task_input, student_output, principle,
            ground_truth=ground_truth,
            correct_example=correct_example,
        )
        try:
            prompt_ids = self.teacher_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            pad_id = (
                self.teacher_tokenizer.pad_token_id
                if self.teacher_tokenizer.pad_token_id is not None
                else self.teacher_tokenizer.eos_token_id
            )
            attention_mask = (prompt_ids != pad_id).long()
            return {"input_ids": prompt_ids, "attention_mask": attention_mask}
        except AttributeError:
            # Fallback for tokenizers without chat template
            prompt_text = "\n\n".join(m.get("content", "") for m in messages)
            encoding = self.teacher_tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024,
            ).to(device)
            input_ids = encoding["input_ids"]
            attention_mask = encoding.get("attention_mask")
            if attention_mask is None:
                pad_id = (
                    self.teacher_tokenizer.pad_token_id
                    if self.teacher_tokenizer.pad_token_id is not None
                    else self.teacher_tokenizer.eos_token_id
                )
                attention_mask = (input_ids != pad_id).long()
            else:
                attention_mask = attention_mask.long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _binary_reward(
        self,
        task_type: str,
        task_input: str,
        student_output: str,
        principle: BinaryPrinciple,
        ground_truth: str = "",
        correct_example: str = "",
    ) -> float:
        prompt_inputs = self._prepare_prompt_inputs(
            task_type, task_input, student_output, principle,
            ground_truth=ground_truth,
            correct_example=correct_example,
        )
        with torch.no_grad():
            yes_lp, no_lp = self._score_yes_no(prompt_inputs)

        raw_reward = float((yes_lp - no_lp).cpu().item())
        if not math.isfinite(raw_reward):
            if self.nan_debug:
                logger.error("Binary reward is non-finite; forcing to 0.0")
            raw_reward = 0.0

        # Running normalization
        momentum = self.norm_momentum
        raw_mean = self.running_mean
        new_mean = raw_mean + momentum * (raw_reward - raw_mean)
        centered = raw_reward - new_mean
        self.running_var = (1 - momentum) * self.running_var + momentum * (centered ** 2)
        self.running_mean = new_mean
        running_std = math.sqrt(max(self.running_var, 1e-6))
        reward = centered / running_std
        reward = float(np.clip(reward, self.clip_min, self.clip_max))

        if self.behavior_logger is not None:
            self.behavior_logger.log(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "component": principle.name,
                    "prompt": task_input,
                    "student_output": student_output,
                    "reward": reward,
                    "details": {
                        "principle": principle.description,
                        "positive": principle.positive_guidance,
                        "negative": principle.negative_guidance,
                        "log_prob_yes": float(yes_lp.cpu().item()),
                        "log_prob_no": float(no_lp.cpu().item()),
                        "raw_reward": raw_reward,
                    },
                }
            )

        return reward

    def compute_accuracy(
        self,
        task,
        student_output: str,
        principle: Optional[BinaryPrinciple] = None,
        ground_truth: str = "",
        correct_example: str = "",
    ) -> float:
        if principle is None:
            # Fallback: look up principle from task type
            tt = getattr(task, "task_type", "unknown")
            principle = self.euroeval_principles.get(tt, self.grammar_principle)
        return self._binary_reward(
            getattr(task, "task_type", "unknown"),
            getattr(task, "input_text", ""),
            student_output,
            principle,
            ground_truth=ground_truth,
            correct_example=correct_example,
        )

    def compute_grammar(
        self,
        task_type: str,
        task_input: str,
        student_output: str,
        principle: Optional[BinaryPrinciple] = None,
    ) -> float:
        if principle is None:
            principle = self.grammar_principle
        return self._binary_reward(
            task_type,
            task_input,
            student_output,
            principle,
        )



class TeacherStudentSimpleRewardSwedish:
    """GRPO reward wrapper for Swedish tasks."""

    def __init__(
        self,
        student_tokenizer,
        teacher_model,
        teacher_tokenizer,
        reward_weights,
        device,
        back_translation_model=None,
        back_translation_tokenizer=None,
    ):
        reward_weights = reward_weights or {}
        clip_min = float(reward_weights.pop("reward_clip_min", -1.0))
        clip_max = float(reward_weights.pop("reward_clip_max", 1.0))
        norm_momentum = float(reward_weights.pop("reward_norm_momentum", 0.1))

        self.student_tokenizer = student_tokenizer
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.weights = reward_weights
        self.device = device

        self.behavior_logger: Optional[BehaviorLogger] = None
        self.nan_debug = False
        self.binary_reward = BinaryFlexibleFeedbackRewardSwedish(
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            device=device,
            clip_min=clip_min,
            clip_max=clip_max,
            norm_momentum=norm_momentum,
        )

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.reward_stats = {
            "accuracy": {"mean": 0.0, "var": 1.0, "std": 1.0},
            #"grammar": {"mean": 0.0, "var": 1.0, "std": 1.0},
        }

        self.solution_cache: Dict[str, Any] = {}

        logger.info("Swedish EuroEval Binary Flexible Feedback reward initialized")

    # ------------------------------------------------------------------
    # Deterministic string-matching reward (no LLM judge)
    # ------------------------------------------------------------------

    def _deterministic_reward(
        self, task_type: str, student_output: str, ground_truth: str
    ) -> float:
        """Return ±DETERMINISTIC_REWARD_MAGNITUDE via string matching.

        Works for task types where the gold answer is a single label or letter:
          - sentiment: positiv / negativ / neutral / blandad
          - acceptability: korrekt / inkorrekt
          - commonsense / knowledge: a / b / c / d
        """
        output_lower = student_output.strip().lower()
        gold_lower = ground_truth.strip().lower()

        if not gold_lower:
            # No gold answer available — cannot do deterministic check
            return 0.0

        correct = False

        if task_type == "sentiment":
            # Gold is one of: positiv, negativ, neutral, blandad
            correct = gold_lower in output_lower

        elif task_type == "acceptability":
            # Gold is "korrekt" or "inkorrekt"
            # Careful: "inkorrekt" contains "korrekt", so check longer first
            if gold_lower == "inkorrekt":
                correct = "inkorrekt" in output_lower
            else:
                # gold is "korrekt" — match only if "inkorrekt" is NOT present
                correct = "korrekt" in output_lower and "inkorrekt" not in output_lower

        elif task_type in ("commonsense", "knowledge"):
            # Gold is a letter: a, b, c, or d
            # Extract first a-d letter from student output
            match = re.search(r"\b([a-d])\b", output_lower)
            if match:
                correct = match.group(1) == gold_lower
            else:
                # Fallback: check if the gold letter appears at the start
                correct = output_lower.startswith(gold_lower)

        return DETERMINISTIC_REWARD_MAGNITUDE if correct else -DETERMINISTIC_REWARD_MAGNITUDE

    def set_behavior_logger(self, behavior_logger: BehaviorLogger):
        self.behavior_logger = behavior_logger
        self.binary_reward.set_behavior_logger(behavior_logger)

    def enable_nan_debug(self, enabled: bool = True):
        self.nan_debug = bool(enabled)
        self.binary_reward.enable_nan_debug(enabled)

    def compute_rewards(self, task, solution: str) -> Dict[str, float]:
        """Compute rewards with smart three-way routing.

        Routing logic (by source and task type):
          1. EuroEval + deterministic type + gold answer → string matching (no judge)
          2. EuroEval + generative type (NER, RC) + gold answer → LLM judge WITH gold
          3. Self-play (any type) → LLM judge with correct_example from EuroEval

        Returns dict with "accuracy" key (and optionally "grammar" if enabled).
        Also returns "reward_method" metadata key for logging.
        """
        rewards: Dict[str, float] = {}
        metadata = getattr(task, "metadata", {}) or {}
        task_type = getattr(task, "task_type", "unknown")
        source = metadata.get("source", "euroeval")
        ground_truth = metadata.get("ground_truth", "")
        correct_example = metadata.get("correct_example", "")

        with torch.no_grad():
            if task_type not in self.binary_reward.euroeval_principles:
                logger.warning("Unknown task type '%s' — skipping reward", task_type)
                rewards["accuracy"] = 0.0
                rewards["reward_method"] = "none"

            elif source == "euroeval" and task_type in DETERMINISTIC_TASK_TYPES and ground_truth:
                # --- Route 1: Deterministic string matching (cheap, no judge) ---
                rewards["accuracy"] = self._deterministic_reward(
                    task_type, solution, ground_truth,
                )
                rewards["reward_method"] = "deterministic"

            elif source == "euroeval":
                # --- Route 2: LLM judge for generative EuroEval (NER, RC) ---
                # Also handles the rare case where a deterministic type lacks gold
                principle = self.binary_reward.euroeval_principles[task_type]
                rewards["accuracy"] = self.binary_reward.compute_accuracy(
                    task, solution, principle,
                    ground_truth=ground_truth,
                    correct_example=correct_example,
                )
                rewards["reward_method"] = "llm_judge_gold"

            else:
                # --- Route 3: LLM judge for self-play tasks ---
                # No gold answer; correct_example provides reference for judge
                principle = self.binary_reward.euroeval_principles[task_type]
                rewards["accuracy"] = self.binary_reward.compute_accuracy(
                    task, solution, principle,
                    ground_truth="",
                    correct_example=correct_example,
                )
                rewards["reward_method"] = "llm_judge_example"

        # Clip to valid range (binary_reward already normalizes internally)
        for key, value in rewards.items():
            if key == "reward_method":
                continue
            if not math.isfinite(value):
                if self.nan_debug:
                    logger.error("Reward for %s is non-finite", key)
                rewards[key] = 0.0
                continue
            rewards[key] = float(np.clip(value, self.clip_min, self.clip_max))
        return rewards

    def update_weights(self, new_weights: Dict[str, float]):
        self.weights.update(new_weights)

    def _normalize_rewards(self, rewards: Dict[str, float]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for component, value in rewards.items():
            stats = self.reward_stats.setdefault(component, {"mean": 0.0, "var": 1.0, "std": 1.0})
            alpha = 0.1
            delta = value - stats["mean"]
            mean_new = stats["mean"] + alpha * delta
            var_new = (1 - alpha) * stats["var"] + alpha * delta * delta
            var_new = max(var_new, 1e-6)
            stats["mean"] = mean_new
            stats["var"] = var_new
            stats["std"] = math.sqrt(var_new)

            z = (value - mean_new) / stats["std"]
            normalized[component] = float(np.clip(z, -3.0, 3.0))
        return normalized



