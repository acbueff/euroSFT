"""
Task Buffer and Curriculum Learning System for FRÓÐI

This module implements the priority queue-based task buffer with curriculum learning
as described in the FRÓÐI research methodology. It manages task generation, storage,
and sampling for the self-play RL pipeline.
"""

import time
import heapq
import random
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Data structure for a single task in the FRÓÐI system"""
    task_type: str  # 'translation', 'qa', 'reasoning'
    input_text: str
    metadata: Dict[str, Any]
    expected_success_rate: float
    timestamp: float = field(default_factory=time.time)
    solution: Optional[str] = None
    solved: bool = False
    
    def __post_init__(self):
        """Ensure timestamp and metadata are well-formed"""
        if self.timestamp is None:
            self.timestamp = time.time()
        # Guarantee metadata is always a dict
        if not isinstance(self.metadata, dict):
            self.metadata = {}
    
    def get_difficulty(self) -> float:
        """Get the difficulty of this task (0.0 = easy, 1.0 = hard)"""
        return self.metadata.get('difficulty', 0.5)
    
    def summary(self) -> str:
        """Get a brief summary of the task for logging"""
        task_desc = f"{self.task_type}: {self.input_text[:50]}..."
        if len(self.input_text) <= 50:
            task_desc = f"{self.task_type}: {self.input_text}"
        return task_desc


class TaskBuffer:
    """
    Priority queue-based task buffer with curriculum learning
    
    This buffer stores tasks and samples them based on priority, which encourages
    the model to work on tasks that are neither too easy nor too hard.
    """
    
    def __init__(self, max_size: int = 10000, target_success_rate: float = 0.7):
        self.max_size = max_size
        self.target_success_rate = target_success_rate
        self.tasks = []  # Min heap: (priority, counter, task)
        self.task_counter = 0  # For tie-breaking in heap
        
        # Track statistics for curriculum learning
        self.success_rates = defaultdict(list)  # task_type -> [success_rates]
        self.solution_cache = {}  # task_type -> deque of solutions
        self.task_history = []  # Recent tasks for analysis
        
        logger.info(f"TaskBuffer initialized with max_size={max_size}, target_success_rate={target_success_rate}")
    
    def insert(self, task: Task, priority: Optional[float] = None):
        """
        Insert a task into the buffer with given priority
        
        Args:
            task: The task to insert
            priority: Priority value (lower = higher priority). If None, computed automatically.
        """
        if priority is None:
            priority = self._compute_priority(task)
        
        # If buffer is full, remove lowest priority task if new task has higher priority
        if len(self.tasks) >= self.max_size:
            if priority > self.tasks[0][0]:  # Lower numbers = higher priority
                heapq.heappop(self.tasks)
            else:
                logger.debug(f"Task rejected due to low priority: {task.summary()}")
                return
        
        self.task_counter += 1
        heapq.heappush(self.tasks, (priority, self.task_counter, task))
        logger.debug(f"Task inserted with priority {priority:.3f}: {task.summary()}")
    
    def sample_by_priority(self, n: int, temperature: float = 1.0) -> List[Task]:
        """
        Sample n tasks with probability proportional to priority
        
        Args:
            n: Number of tasks to sample
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            List of sampled tasks
        """
        if len(self.tasks) == 0:
            logger.warning("No tasks available for sampling")
            return []
        
        # Get top candidates (sample from top portion of heap)
        candidates = []
        temp_heap = []
        
        # Extract more candidates than needed to ensure diversity
        n_candidates = min(len(self.tasks), max(n * 3, 20))
        
        for _ in range(n_candidates):
            if self.tasks:
                item = heapq.heappop(self.tasks)
                candidates.append(item)
                temp_heap.append(item)
        
        # Restore heap
        for item in temp_heap:
            heapq.heappush(self.tasks, item)
        
        if not candidates:
            return []
        
        # Convert priorities to probabilities (lower priority = higher probability)
        priorities = np.array([1.0 / (p + 1e-6) for p, _, _ in candidates])
        
        # Apply temperature
        if temperature > 0:
            probs = np.exp(priorities / temperature)
            probs = probs / probs.sum()
        else:
            # Greedy selection
            best_idx = np.argmax(priorities)
            probs = np.zeros(len(priorities))
            probs[best_idx] = 1.0
        
        # Sample without replacement
        n_sample = min(n, len(candidates))
        try:
            indices = np.random.choice(len(candidates), size=n_sample, p=probs, replace=False)
            sampled_tasks = [candidates[i][2] for i in indices]
            
            logger.debug(f"Sampled {len(sampled_tasks)} tasks from buffer")
            return sampled_tasks
            
        except ValueError as e:
            logger.warning(f"Error in sampling: {e}. Falling back to random sampling.")
            indices = np.random.choice(len(candidates), size=n_sample, replace=False)
            return [candidates[i][2] for i in indices]
    
    def update_task_result(self, task: Task, success: bool, solution: str = ""):
        """
        Update the buffer with the result of attempting a task
        
        Args:
            task: The task that was attempted
            success: Whether the task was solved successfully
            solution: The solution that was generated
        """
        # Update success rates for curriculum learning
        self.success_rates[task.task_type].append(1.0 if success else 0.0)
        
        # Keep only recent success rates for adaptive curriculum
        max_history = 1000
        if len(self.success_rates[task.task_type]) > max_history:
            self.success_rates[task.task_type] = self.success_rates[task.task_type][-max_history:]
        
        # Cache solution for novelty computation
        if solution and task.task_type not in self.solution_cache:
            self.solution_cache[task.task_type] = deque(maxlen=5000)
        
        if solution:
            self.solution_cache[task.task_type].append(solution)
        
        # Update task
        task.solved = success
        task.solution = solution
        
        # Add to history for analysis
        self.task_history.append({
            'task_type': task.task_type,
            'difficulty': task.get_difficulty(),
            'success': success,
            'timestamp': time.time()
        })
        
        # Keep history manageable
        if len(self.task_history) > 10000:
            self.task_history = self.task_history[-5000:]
    
    def recompute_priorities(self, success_estimator_fn=None):
        """
        Recompute priorities for all tasks based on current model capability
        
        Args:
            success_estimator_fn: Function that estimates success probability for a task
        """
        logger.info("Recomputing task priorities...")
        
        new_heap = []
        
        for priority, counter, task in self.tasks:
            if success_estimator_fn:
                expected_success = success_estimator_fn(task)
            else:
                expected_success = self._estimate_success_rate(task)
            
            new_priority = self._compute_priority_from_success_rate(expected_success, task)
            heapq.heappush(new_heap, (new_priority, counter, task))
        
        self.tasks = new_heap
        logger.info(f"Recomputed priorities for {len(self.tasks)} tasks")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring"""
        stats = {
            'total_tasks': len(self.tasks),
            'task_types': {},
            'avg_difficulty': 0.0,
            'success_rates': {}
        }
        
        # Task type distribution
        type_counts = defaultdict(int)
        difficulties = []
        
        for _, _, task in self.tasks:
            type_counts[task.task_type] += 1
            difficulties.append(task.get_difficulty())
        
        stats['task_types'] = dict(type_counts)
        if difficulties:
            stats['avg_difficulty'] = np.mean(difficulties)
        
        # Recent success rates
        for task_type, successes in self.success_rates.items():
            if successes:
                recent_successes = successes[-100:]  # Last 100 attempts
                stats['success_rates'][task_type] = np.mean(recent_successes)
        
        return stats
    
    def get_past_solutions(self, task_type: str, n: int = 1000) -> List[str]:
        """Get past solutions for a task type (for novelty computation)"""
        if task_type not in self.solution_cache:
            return []
        
        solutions = list(self.solution_cache[task_type])
        return solutions[-n:] if len(solutions) > n else solutions
    
    def _compute_priority(self, task: Task) -> float:
        """
        Compute priority for a task based on curriculum learning principles
        
        Lower priority number = higher actual priority in the heap
        """
        # Estimate current success rate for this type of task
        expected_success = self._estimate_success_rate(task)
        return self._compute_priority_from_success_rate(expected_success, task)
    
    def _compute_priority_from_success_rate(self, expected_success: float, task: Task) -> float:
        """Compute priority from estimated success rate"""
        # Optimal difficulty: tasks where success rate is close to target
        difficulty_match = abs(expected_success - self.target_success_rate)
        
        # Novelty/recency bonus
        age = time.time() - task.timestamp
        novelty_bonus = 1.0 / (age + 1.0)  # Newer tasks get slight preference
        
        # Priority is lower (higher actual priority) when difficulty is optimal
        priority = difficulty_match + 0.1 * (1.0 - novelty_bonus)
        
        return priority
    
    def _estimate_success_rate(self, task: Task) -> float:
        """Estimate success rate for a task based on recent performance"""
        task_type = task.task_type
        difficulty = task.get_difficulty()
        
        if task_type not in self.success_rates or not self.success_rates[task_type]:
            # No history, return based on difficulty
            return max(0.1, 1.0 - difficulty)
        
        # Get recent success rate for this task type
        recent_successes = self.success_rates[task_type][-100:]
        base_success_rate = np.mean(recent_successes)
        
        # Adjust based on difficulty relative to average
        # Higher difficulty should lower expected success
        avg_difficulty = 0.5  # Assume average difficulty
        difficulty_adjustment = (avg_difficulty - difficulty) * 0.5
        
        estimated_rate = base_success_rate + difficulty_adjustment
        return np.clip(estimated_rate, 0.05, 0.95)
    
    @property
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.tasks)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.tasks) == 0


class AdaptiveWeightSchedulerSimple:
    """Pareto-style adaptive scheduler for accuracy/grammar reward weights."""

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        ema: float = 0.7,
        accuracy_bias: float = 0.15,
        grammar_activation_patience: int = 200,
    ):
        defaults = {'accuracy': 1.0, 'grammar': 0.0}
        self.weights = defaults.copy()
        if isinstance(initial_weights, dict):
            for k in defaults:
                if k in initial_weights:
                    try:
                        self.weights[k] = float(initial_weights[k])
                    except Exception:
                        pass

        self.weights = self._normalize_weights(self.weights)
        self.ema = float(np.clip(ema, 0.0, 0.99))
        self.accuracy_bias = float(np.clip(accuracy_bias, 0.0, 0.5))
        self.grammar_activation_patience = max(int(grammar_activation_patience), 1)

        self.history: List[Dict[str, Dict[str, float]]] = []
        self.iteration = 0
        self.grammar_enabled = False
        self._grammar_missing_steps = 0

        logger.info(
            "Pareto weight scheduler initialized | weights=%s ema=%.2f bias=%.2f",
            self.weights,
            self.ema,
            self.accuracy_bias,
        )

    def update(self, rewards: Dict[str, List[float]], iteration: int) -> Dict[str, float]:
        self.iteration = iteration

        stats: Dict[str, Dict[str, float]] = {}
        mean_scores: Dict[str, Optional[float]] = {}
        counts: Dict[str, int] = {}

        for component in ('accuracy', 'grammar'):
            values = [float(v) for v in rewards.get(component, []) if np.isfinite(v)]
            counts[component] = len(values)
            if values:
                stats[component] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'count': len(values),
                }
                mean_scores[component] = stats[component]['mean']
            else:
                mean_scores[component] = None

        self.history.append(stats)

        if counts.get('grammar', 0) > 0:
            self.grammar_enabled = True
            self._grammar_missing_steps = 0
        else:
            self._grammar_missing_steps += 1
            if self._grammar_missing_steps >= self.grammar_activation_patience:
                self.grammar_enabled = False

        pareto_target = self._compute_pareto_weights(mean_scores)

        if not self.grammar_enabled:
            pareto_target['grammar'] = 0.0
            pareto_target['accuracy'] = 1.0

        prior = {'accuracy': 1.0, 'grammar': 0.0}
        blended_target: Dict[str, float] = {}
        for comp in ('accuracy', 'grammar'):
            pareto_value = pareto_target.get(comp, 0.0)
            blended_target[comp] = (
                (1.0 - self.accuracy_bias) * pareto_value
                + self.accuracy_bias * prior.get(comp, 0.0)
            )

        new_weights: Dict[str, float] = {}
        for comp in ('accuracy', 'grammar'):
            prev = self.weights.get(comp, 0.0)
            target_val = blended_target.get(comp, 0.0)
            new_weights[comp] = self.ema * prev + (1.0 - self.ema) * target_val

        self.weights = self._normalize_weights(new_weights, grammar_active=self.grammar_enabled)

        logger.debug(
            "Pareto weights updated at iter %d | stats=%s target=%s weights=%s",
            iteration,
            stats,
            pareto_target,
            self.weights,
        )

        return self.weights.copy()

    def _compute_pareto_weights(
        self,
        mean_scores: Dict[str, Optional[float]],
    ) -> Dict[str, float]:
        active = {k: v for k, v in mean_scores.items() if v is not None}
        default_weights = {'accuracy': 1.0, 'grammar': 0.0}
        if not active:
            return default_weights

        if len(active) == 1:
            comp = next(iter(active))
            if comp == 'grammar':
                return {'accuracy': 0.0, 'grammar': 1.0}
            return default_weights

        min_comp = min(active, key=active.get)
        max_comp = max(active, key=active.get)
        min_score = float(active[min_comp])
        max_score = float(active[max_comp])

        weights = {'accuracy': 0.0, 'grammar': 0.0}

        if min_score <= 0.0 <= max_score and max_score != min_score:
            denom = max_score - min_score
            weights[max_comp] = -min_score / denom
            weights[min_comp] = max_score / denom
        elif min_score > 0.0:
            weights[min_comp] = 1.0
        elif max_score < 0.0:
            weights[max_comp] = 1.0
        else:
            return default_weights

        return weights

    @staticmethod
    def _normalize_weights(
        weights: Dict[str, float],
        grammar_active: bool = True,
    ) -> Dict[str, float]:
        accuracy_weight = max(0.0, weights.get('accuracy', 0.0))
        grammar_weight = max(0.0, weights.get('grammar', 0.0)) if grammar_active else 0.0
        total = accuracy_weight + grammar_weight
        if total <= 0.0:
            return {'accuracy': 1.0, 'grammar': 0.0}
        return {
            'accuracy': accuracy_weight / total,
            'grammar': grammar_weight / total,
        }
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights (only 'accuracy' and 'grammar')."""
        return self.weights.copy()
    
    def get_history(self) -> List[Dict]:
        """Get weight adaptation history."""
        return self.history.copy()

class AdaptiveWeightScheduler:
    """
    Adaptive weight scheduler for multi-objective rewards in FRÓÐI
    
    This scheduler adjusts the weights of different reward components
    based on training progress and curriculum learning principles.
    """
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the weight scheduler
        
        Args:
            initial_weights: Initial weights for reward components
        """
        self.weights = initial_weights or {
            'accuracy': 0.4,
            'fluency': 0.3,
            'reconstruction': 0.2,
            'novelty': 0.1,
            'grammar': 0.0
        }
        
        self.history = []
        self.iteration = 0
        
        logger.info(f"AdaptiveWeightScheduler initialized with weights: {self.weights}")
    
    def update(self, rewards: Dict[str, List[float]], iteration: int) -> Dict[str, float]:
        """
        Update weights based on reward statistics and training progress
        
        Args:
            rewards: Dictionary mapping reward component names to lists of values
            iteration: Current training iteration
            
        Returns:
            Updated weights dictionary
        """
        self.iteration = iteration
        
        # Track reward statistics
        reward_stats = {}
        for component, values in rewards.items():
            if values:
                reward_stats[component] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        self.history.append(reward_stats)
        
        # Curriculum-based weight adaptation
        if iteration < 1000:
            # Early training: focus on fluency and basic accuracy
            self.weights.update({
                'fluency': 0.4,
                'accuracy': 0.3,
                'grammar': 0.2,
                'reconstruction': 0.1,
                'novelty': 0.0
            })
        elif iteration < 5000:
            # Mid training: balance all components
            self.weights.update({
                'accuracy': 0.35,
                'fluency': 0.25,
                'reconstruction': 0.25,
                'novelty': 0.1,
                'grammar': 0.05
            })
        else:
            # Late training: focus on accuracy and novelty
            self.weights.update({
                'accuracy': 0.4,
                'novelty': 0.25,
                'fluency': 0.2,
                'reconstruction': 0.1,
                'grammar': 0.05
            })
        
        # Dynamic adjustment based on performance trends
        if len(self.history) > 100:
            for component in self.weights.keys():
                trend = self._compute_improvement_trend(component)
                if trend < 0.01:  # Component has plateaued
                    self.weights[component] *= 0.95  # Slightly reduce weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.debug(f"Updated weights at iteration {iteration}: {self.weights}")
        return self.weights.copy()
        
    
    def _compute_improvement_trend(self, component: str) -> float:
        """Compute improvement trend for a reward component"""
        if len(self.history) < 20:
            return 0.0
        
        recent_window = self.history[-100:]
        values = []
        
        for stats in recent_window:
            if component in stats and 'mean' in stats[component]:
                values.append(stats[component]['mean'])
        
        if len(values) < 10:
            return 0.0
        
        # Compute trend as improvement from first half to second half
        mid = len(values) // 2
        first_half = np.mean(values[:mid])
        second_half = np.mean(values[mid:])
        
        if first_half == 0:
            return 0.0
        
        improvement = (second_half - first_half) / abs(first_half)
        return improvement
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return self.weights.copy()
    
    def get_history(self) -> List[Dict]:
        """Get weight adaptation history"""
        return self.history.copy() 