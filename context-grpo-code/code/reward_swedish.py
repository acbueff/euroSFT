"""
FRÓÐI Reward Model - SWEDISH ADAPTATION

Binary Flexible Feedback Reward model for Swedish language tasks.
Supports Swedish↔English translation, Swedish QA, and Swedish summarization.
"""

import logging
import math
import re
from collections import deque
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
        accuracy_principle: Optional[BinaryPrinciple] = None,
        qa_principle: Optional[BinaryPrinciple] = None,
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

        # Swedish translation accuracy principle
        self.accuracy_principle = accuracy_principle or BinaryPrinciple(
            name="accuracy",
            description=(
                "Översättningen måste bevara den faktiska betydelsen av källtexten "
                "utan utelämnanden eller hallucinationer, samtidigt som den håller sig "
                "till det begärda målspråket (svenska eller engelska)."
            ),
            positive_guidance=(
                "Belöna utdata som troget förmedlar varje betydelseenhet från "
                "inmatningen, behåller viktig terminologi och använder det begärda "
                "målspråket flytande."
            ),
            negative_guidance=(
                "Straffa betydelseförändringar, utelämnanden, hallucinationer eller "
                "blandning av språk (t.ex. att lämna svaret på källspråket)."
            ),
        )
        
        # Swedish QA accuracy principle
        self.qa_principle = qa_principle or BinaryPrinciple(
            name="qa_accuracy",
            description=(
                "Svaret måste vara faktiskt korrekt, direkt stött av kontexten "
                "och förbli på det begärda språket."
            ),
            positive_guidance=(
                "Belöna svar som tydligt anger korrekt information från "
                "kontexten utan saknade nödvändiga detaljer."
            ),
            negative_guidance=(
                "Straffa hallucinationer, motsägande uttalanden, saknade "
                "nyckelfakta eller svar på fel språk."
            ),
        )
        
        # Swedish question quality principle
        self.question_principle = BinaryPrinciple(
            name="question_quality",
            description=(
                "Studentgenererade frågor måste vara välformade, besvarabara "
                "från den givna kontexten och formulerade på samma språk."
            ),
            positive_guidance=(
                "Belöna frågor som är tydliga, grammatiskt korrekta, slutar med "
                "frågetecken och kan besvaras direkt med hjälp av kontexten."
            ),
            negative_guidance=(
                "Straffa frågor som är orelaterade till kontexten, tvetydiga, "
                "inte formulerade som frågor eller skrivna på fel språk."
            ),
        )
        
        # Swedish summary quality principle
        self.summary_principle = BinaryPrinciple(
            name="summary_quality",
            description=(
                "Sammanfattningen måste vara faktiskt korrekt, direkt stödd av "
                "kontexten och förbli på det begärda språket."
            ),
            positive_guidance=(
                "Belöna sammanfattningar som tydligt anger korrekt information "
                "från kontexten utan saknade nödvändiga detaljer."
            ),
            negative_guidance=(
                "Straffa hallucinationer, motsägande uttalanden, saknade "
                "nyckelfakta eller sammanfattningar på fel språk."
            ),
        )
        
        # Swedish summary source quality principle
        self.summary_source_principle = BinaryPrinciple(
            name="summary_source_quality",
            description=(
                "Det utökade stycket måste förbli troget mot ursprungskontexten, "
                "lägga till sammanhängande stödjande detaljer och hålla sig till "
                "det begärda språket."
            ),
            positive_guidance=(
                "Belöna stycken som naturligt utvecklar kontexten, lägger till "
                "relaterade detaljer och bevarar ton och språk."
            ),
            negative_guidance=(
                "Straffa hallucinationer, motsägelser, off-topic-utökningar "
                "eller språkbyten."
            ),
        )
        
        # Swedish grammar principle
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
            if task_type == "translation":
                user = (
                    "Assess whether the student translation is grammatically correct and natural in the target language.\n"
                    f"Source text:\n{task_input}\n\n"
                    f"Student translation:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the translation is grammatically correct and fluent, otherwise NO."
                )
            elif task_type == "summary":
                user = (
                    "Assess whether the student's summary is grammatically correct and natural in the target language.\n"
                    f"Reference passage:\n{task_input}\n\n"
                    f"Student summary:\n{student_output}\n\n"
                    f"{principle_block}\n\n"
                    "Respond with YES if the summary is grammatically correct and fluent, otherwise NO."
                )
            else:
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

        if task_type == "translation":
            user = (
                "Assess whether the student translation satisfies the principle.\n"
                f"Source text:\n{task_input}\n\n"
                f"Student translation:\n{student_output}\n\n"
                f"{principle_block}\n\n"
                "Respond with YES if satisfied, otherwise NO."
            )
        elif task_type == "question_eval":
            user = (
                "Assess whether the student's question is acceptable for the context.\n"
                f"Context:\n{task_input}\n\n"
                f"Student question:\n{student_output}\n\n"
                f"{principle_block}\n\n"
                "Respond with YES if the question meets the criteria, otherwise NO."
            )
        elif task_type == "summary":
            user = (
                "Assess whether the student's summary captures the key ideas without hallucinations.\n"
                f"Passage:\n{task_input}\n\n"
                f"Student summary:\n{student_output}\n\n"
                f"{principle_block}\n\n"
                "Respond with YES if the summary is faithful, otherwise NO."
            )
        elif task_type == "summary_source":
            user = (
                "Assess whether the expanded passage elaborates on the seed context coherently and faithfully.\n"
                f"Seed context:\n{task_input}\n\n"
                f"Expanded passage:\n{student_output}\n\n"
                f"{principle_block}\n\n"
                "Respond with YES if the expansion is suitable for summarisation, otherwise NO."
            )
        else:
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
    ) -> Dict[str, torch.Tensor]:
        device = self._teacher_device()
        messages = self._build_rating_messages(
            task_type, task_input, student_output, principle
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
    ) -> float:
        prompt_inputs = self._prepare_prompt_inputs(
            task_type, task_input, student_output, principle
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
    ) -> float:
        if principle is None:
            principle = self.accuracy_principle
        return self._binary_reward(
            getattr(task, "task_type", "unknown"),
            getattr(task, "input_text", ""),
            student_output,
            principle,
        )

    def compute_summary(
        self,
        passage: str,
        student_summary: str,
        principle: Optional[BinaryPrinciple] = None,
    ) -> float:
        if principle is None:
            principle = self.summary_principle
        return self._binary_reward(
            "summary",
            passage,
            student_summary,
            principle,
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

    def judge_question(self, context: str, question: str) -> Tuple[bool, Dict[str, float]]:
        prompt_inputs = self._prepare_prompt_inputs(
            "question_eval",
            context,
            question,
            self.question_principle,
        )
        with torch.no_grad():
            yes_lp, no_lp = self._score_yes_no(prompt_inputs)

        yes_value = float(yes_lp.cpu().item())
        no_value = float(no_lp.cpu().item())
        decision = bool(yes_value >= no_value)
        details = {
            "yes_logprob": yes_value,
            "no_logprob": no_value,
            "logprob_diff": yes_value - no_value,
        }

        if self.behavior_logger is not None:
            self.behavior_logger.log(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "component": "question_validation",
                    "prompt": context,
                    "student_output": question,
                    "decision": "YES" if decision else "NO",
                    "details": details.copy(),
                }
            )

        return decision, details

    def judge_summary_source(self, context: str, passage: str) -> Tuple[bool, Dict[str, float]]:
        prompt_inputs = self._prepare_prompt_inputs(
            "summary_source",
            context,
            passage,
            self.summary_source_principle,
        )
        with torch.no_grad():
            yes_lp, no_lp = self._score_yes_no(prompt_inputs)

        yes_value = float(yes_lp.cpu().item())
        no_value = float(no_lp.cpu().item())
        decision = bool(yes_value >= no_value)
        details = {
            "yes_logprob": yes_value,
            "no_logprob": no_value,
            "logprob_diff": yes_value - no_value,
        }

        if self.behavior_logger is not None:
            self.behavior_logger.log(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "component": "summary_source_validation",
                    "prompt": context,
                    "student_output": passage,
                    "decision": "YES" if decision else "NO",
                    "details": details.copy(),
                }
            )

        return decision, details


class TeacherStudentSimpleRewardSwedish:
    """GRPO reward wrapper for Swedish tasks."""

    def __init__(
        self,
        student_tokenizer,
        teacher_model,
        teacher_tokenizer,
        back_translation_model,
        back_translation_tokenizer,
        reward_weights,
        device,
    ):
        reward_weights = reward_weights or {}
        clip_min = float(reward_weights.pop("reward_clip_min", -1.0))
        clip_max = float(reward_weights.pop("reward_clip_max", 1.0))
        norm_momentum = float(reward_weights.pop("reward_norm_momentum", 0.1))

        self.student_tokenizer = student_tokenizer
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.bt_model = back_translation_model
        self.bt_tokenizer = back_translation_tokenizer
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
        self._teacher_cache: Dict[str, str] = {}
        self._teacher_cache_order: deque = deque(maxlen=128)
        self._teacher_cache_max = 128

        logger.info("Swedish Binary Flexible Feedback reward initialized")

    def set_behavior_logger(self, behavior_logger: BehaviorLogger):
        self.behavior_logger = behavior_logger
        self.binary_reward.set_behavior_logger(behavior_logger)

    def enable_nan_debug(self, enabled: bool = True):
        self.nan_debug = bool(enabled)
        self.binary_reward.enable_nan_debug(enabled)

    def evaluate_student_question(self, context: str, question: str) -> Tuple[bool, Dict[str, float]]:
        return self.binary_reward.judge_question(context, question)

    def generate_teacher_question(self, context: str) -> str:
        return self._get_teacher_question(context)

    def compute_rewards(self, task, solution: str) -> Dict[str, float]:
        """
        Compute rewards for a task solution.
        
        Supports ablation studies via config:
        - grammar_enabled: Whether to include grammar reward (default: True)
        - Only computes grammar if weight > 0 AND grammar_enabled is True
        """
        rewards: Dict[str, float] = {}
        
        # Check if grammar evaluation is enabled (for ablation studies)
        grammar_enabled = self.weights.get("grammar_enabled", True)
        grammar_weight = self.weights.get("grammar", 0.0)
        include_grammar = grammar_enabled and grammar_weight > 0.0
        
        with torch.no_grad():
            task_type = getattr(task, "task_type", "translation")
            
            if task_type == "qa":
                principle = self.binary_reward.qa_principle
                rewards["accuracy"] = self.binary_reward.compute_accuracy(
                    task, solution, principle
                )
                # QA tasks can optionally include grammar
                if include_grammar:
                    rewards["grammar"] = self.binary_reward.compute_grammar(
                        "qa",
                        getattr(task, "input_text", ""),
                        solution,
                        self.binary_reward.grammar_principle,
                    )
                    
            elif task_type == "summary":
                passage = task.metadata.get("passage", task.input_text)
                rewards["accuracy"] = self.binary_reward.compute_summary(
                    passage, solution, self.binary_reward.summary_principle
                )
                if include_grammar:
                    rewards["grammar"] = self.binary_reward.compute_grammar(
                        "summary",
                        passage,
                        solution,
                        self.binary_reward.grammar_principle,
                    )
            else:
                # Translation and other task types
                principle = self.binary_reward.accuracy_principle
                rewards["accuracy"] = self.binary_reward.compute_accuracy(
                    task, solution, principle
                )
                if task_type == "translation" and include_grammar:
                    rewards["grammar"] = self.binary_reward.compute_grammar(
                        "translation",
                        getattr(task, "metadata", {}).get("source_text", task.input_text),
                        solution,
                        self.binary_reward.grammar_principle,
                    )

        # NOTE: Rewards from _binary_reward() are already normalized with running mean/variance
        # We only clip here to avoid a second normalization layer which can cause signal degradation
        # over long training runs (double normalization compresses the reward signal excessively)
        for key, value in rewards.items():
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

    def _teacher_cache_key(self, kind: str, prompt: str) -> str:
        return f"{kind}::{prompt}"

    def _cache_teacher_result(self, cache_key: str, value: str) -> None:
        self._teacher_cache[cache_key] = value
        self._teacher_cache_order.append(cache_key)
        while len(self._teacher_cache_order) > self._teacher_cache_max:
            old_key = self._teacher_cache_order.popleft()
            self._teacher_cache.pop(old_key, None)

    def _run_teacher_chat(self, messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
        device = self.binary_reward._teacher_device()
        input_ids: torch.Tensor
        attention_mask: torch.Tensor

        try:
            prompt_inputs = self.teacher_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except AttributeError:
            fallback_text = "\n\n".join(m.get("content", "") for m in messages)
            prompt_inputs = self.teacher_tokenizer(
                fallback_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )

        if isinstance(prompt_inputs, torch.Tensor):
            input_ids = prompt_inputs.to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            if hasattr(prompt_inputs, "to"):
                prompt_inputs = prompt_inputs.to(device)
            if isinstance(prompt_inputs, dict):
                input_ids = prompt_inputs["input_ids"]
                attention_mask = prompt_inputs.get("attention_mask")
            else:
                input_ids = prompt_inputs.input_ids
                attention_mask = getattr(prompt_inputs, "attention_mask", None)

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=device)
            else:
                attention_mask = attention_mask.to(device)
            input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = self.teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=False,
                pad_token_id=self.teacher_tokenizer.eos_token_id,
            )

        try:
            gen_tokens = outputs[0][input_ids.shape[1]:]
            completion = self.teacher_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        except Exception:
            completion = self.teacher_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return completion

    def _get_teacher_translation(self, prompt: str) -> str:
        cache_key = self._teacher_cache_key("translation", prompt)
        cached = self._teacher_cache.get(cache_key)
        if cached is not None:
            return cached

        system = (
            "Du är en precis svensk↔engelsk översättare. Översätt användarens text "
            "korrekt och ge endast översättningen utan förklaringar."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        translation = self._run_teacher_chat(messages)
        if not translation:
            translation = "Översättning ej tillgänglig."
        self._cache_teacher_result(cache_key, translation)
        return translation

    def _get_teacher_answer(self, prompt: str) -> str:
        cache_key = self._teacher_cache_key("qa", prompt)
        cached = self._teacher_cache.get(cache_key)
        if cached is not None:
            return cached

        locale = self.binary_reward._infer_locale(prompt, "")
        if locale.startswith("sv"):
            system = (
                "Du är en precis fråga-svar-assistent. Besvara frågan "
                "endast med information från kontexten. Svara på svenska "
                "om kontexten är på svenska, annars på önskat språk."
            )
        else:
            system = (
                "You are a precise QA assistant. Answer the question using only "
                "information grounded in the provided context. Respond in the "
                "requested language (Swedish or English)."
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        answer = self._run_teacher_chat(messages, max_new_tokens=160)
        if not answer:
            answer = "Svar ej tillgängligt." if locale.startswith("sv") else "Answer unavailable."
        self._cache_teacher_result(cache_key, answer)
        return answer

    def _get_teacher_reasoning(self, prompt: str) -> str:
        cache_key = self._teacher_cache_key("reasoning", prompt)
        cached = self._teacher_cache.get(cache_key)
        if cached is not None:
            return cached

        locale = self.binary_reward._infer_locale(prompt, "")
        if locale.startswith("sv"):
            system = (
                "Du är en logisk assistent. Analysera premisserna noggrant "
                "och formulera en kort slutsats på svenska."
            )
            user_suffix = "\nVänligen ge slutsatsen som en mening."
        else:
            system = (
                "You are a logical reasoning assistant. Carefully analyze the "
                "premises and provide a concise final conclusion in English."
            )
            user_suffix = "\nProvide the final conclusion as a single sentence."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt + user_suffix},
        ]
        reasoning = self._run_teacher_chat(messages, max_new_tokens=120)
        if not reasoning:
            reasoning = "Ingen slutsats tillgänglig." if locale.startswith("sv") else "Conclusion unavailable."
        self._cache_teacher_result(cache_key, reasoning)
        return reasoning

    def _get_teacher_summary(self, context: str, passage: str) -> str:
        cache_key = self._teacher_cache_key("summary", context + "||" + passage)
        cached = self._teacher_cache.get(cache_key)
        if cached is not None:
            return cached

        locale = self.binary_reward._infer_locale(passage, "")
        if locale.startswith("sv"):
            system = (
                "Du är en precis sammanfattare. Sammanfatta följande stycke i två till tre meningar."
            )
        else:
            system = (
                "You are a precise summarizer. Summarize the passage in two to three sentences without adding new facts."
            )
        user = (
            "Passage:\n"
            f"{passage}\n\n"
            "Provide a concise summary that stays factual."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        summary = self._run_teacher_chat(messages, max_new_tokens=160).strip()
        if summary:
            self._cache_teacher_result(cache_key, summary)
        return summary

    def _get_teacher_question(self, context: str) -> str:
        cache_key = self._teacher_cache_key("question", context)
        cached = self._teacher_cache.get(cache_key)
        if cached is not None:
            return cached

        locale = self.binary_reward._infer_locale(context, "")
        if locale.startswith("sv"):
            system = (
                "Du är en noggrann uppgiftsskapare. Formulera exakt en "
                "tydlig fråga som kan besvaras direkt med hjälp av den givna "
                "kontexten. Håll frågan på svenska."
            )
        else:
            system = (
                "You are a meticulous task designer. Craft exactly one clear question "
                "that can be answered directly using the provided context. Use the "
                "same language as the context."
            )
        user = (
            "Context:\n"
            f"{context}\n\n"
            "Produce exactly one concise question that the student should answer."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        question = self._run_teacher_chat(messages, max_new_tokens=64).strip()
        if question and not question.endswith("?"):
            question = f"{question}?"
        if question:
            self._cache_teacher_result(cache_key, question)
        return question

    def get_reward(self, original_texts, student_solutions):
        from ..core.task_buffer import Task

        rewards = []
        for original_text, solution in zip(original_texts, student_solutions):
            task = Task(
                task_type="translation",
                input_text=original_text,
                metadata={},
                expected_success_rate=0.7,
            )
            reward_dict = self.compute_rewards(task, solution)
            total_reward = sum(
                self.weights.get(k, 0.0) * v for k, v in reward_dict.items()
            )
            rewards.append(total_reward)
        return torch.tensor(rewards, dtype=torch.float32)

