"""Unified optimization pipeline supporting multiple optimizers.

This module provides a single interface for running DSPy optimization
experiments with different optimizers (GEPA, MIPROv2).
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

import dspy
from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction

from .config import (
    MODEL_CONFIG,
    OPTIMIZER_CONFIG,
    PATH_CONFIG,
    PROCESSING_CONFIG,
)
from .core import BasicQA, DataLoader, OptimizationTracker, gepa_metric, validate_answer


class OptimizerType(Enum):
    """Supported optimizer types."""

    GEPA = "gepa"
    MIPRO = "mipro"
    MIPROV2 = "miprov2"


@dataclass
class OptimizationResult:
    """Results from an optimization run."""

    optimizer: str
    baseline_accuracy: float
    optimized_accuracy: float
    improvement: float
    instruction: Optional[str]
    demos: List[Dict[str, Any]]
    trials: List[Dict[str, Any]]
    instruction_candidates: List[Dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    evolution_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "optimizer": self.optimizer,
            "baseline_accuracy": self.baseline_accuracy,
            "optimized_accuracy": self.optimized_accuracy,
            "improvement": self.improvement,
            "instruction": self.instruction,
            "demos": self.demos,
            "optimization_trials": self.trials,
            "instruction_candidates": self.instruction_candidates,
            "timestamp": self.timestamp,
            "evolution_summary": self.evolution_summary,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Save results to JSON file.

        Args:
            path: Output path. Defaults to optimization_results.json.

        Returns:
            Path to saved file.
        """
        if path is None:
            path = PATH_CONFIG.results_file
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


class OptimizationPipeline:
    """Unified pipeline for running DSPy optimization with any supported optimizer.

    This pipeline handles:
    - DSPy and logging configuration (explicit, no side effects on import)
    - Dataset loading (lazy, only when needed)
    - Baseline evaluation
    - Optimization with configurable optimizer
    - Result extraction and summarization

    Example:
        pipeline = OptimizationPipeline()
        result = await pipeline.run(OptimizerType.GEPA)
        result.save()
    """

    def __init__(
        self,
        model: Optional[str] = None,
        async_max: int = MODEL_CONFIG.async_max,
        data_loader: Optional[DataLoader] = None,
    ) -> None:
        """Initialize the optimization pipeline.

        Args:
            model: Model name for DSPy LM. Defaults to MODEL_CONFIG.default_model.
            async_max: Maximum async concurrency. Defaults to MODEL_CONFIG.async_max.
            data_loader: DataLoader instance. Defaults to new DataLoader().
        """
        self.model_name = model or MODEL_CONFIG.default_model
        self.async_max = async_max
        self.data_loader = data_loader or DataLoader()
        self.tracker = OptimizationTracker()
        self._configured = False

    def configure(self) -> "OptimizationPipeline":
        """Configure DSPy and logging. Call this explicitly before running.

        Returns:
            Self for method chaining.
        """
        if self._configured:
            return self

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")

        # Attach tracker to relevant loggers
        for logger_name in [
            "dspy.teleprompt.mipro_optimizer_v2",
            "dspy.teleprompt.signature_opt_typed",
            "dspy.teleprompt.gepa",
            "dspy",
        ]:
            logging.getLogger(logger_name).addHandler(self.tracker)

        # Configure DSPy
        lm = dspy.LM(self.model_name)
        dspy.configure(lm=lm, async_max=self.async_max)

        self._configured = True
        return self

    def _create_optimizer(
        self,
        optimizer_type: OptimizerType,
    ) -> dspy.teleprompt.Teleprompter:
        """Factory method to create the appropriate optimizer.

        Args:
            optimizer_type: Type of optimizer to create.

        Returns:
            Configured optimizer instance.

        Raises:
            ValueError: If optimizer type is not supported.
        """
        if optimizer_type == OptimizerType.GEPA:
            reflection_lm = dspy.LM(
                self.model_name,
                temperature=MODEL_CONFIG.reflection_temperature,
            )
            return dspy.GEPA(
                metric=gepa_metric,
                auto=OPTIMIZER_CONFIG.gepa_auto,
                reflection_lm=reflection_lm,
                num_threads=OPTIMIZER_CONFIG.gepa_num_threads,
            )
        elif optimizer_type in (OptimizerType.MIPRO, OptimizerType.MIPROV2):
            return dspy.MIPROv2(
                metric=validate_answer,
                auto=OPTIMIZER_CONFIG.mipro_auto,
                num_threads=OPTIMIZER_CONFIG.mipro_num_threads,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def _compile_module(
        self,
        optimizer: dspy.teleprompt.Teleprompter,
        optimizer_type: OptimizerType,
        module_class: Type[dspy.Module],
    ) -> dspy.Module:
        """Compile module with the appropriate optimizer settings.

        Args:
            optimizer: Optimizer instance.
            optimizer_type: Type of optimizer being used.
            module_class: DSPy module class to compile.

        Returns:
            Compiled (optimized) module.
        """
        if optimizer_type in (OptimizerType.MIPRO, OptimizerType.MIPROV2):
            return optimizer.compile(
                module_class(),
                trainset=self.data_loader.trainset,
                max_bootstrapped_demos=OPTIMIZER_CONFIG.mipro_max_bootstrapped_demos,
                max_labeled_demos=OPTIMIZER_CONFIG.mipro_max_labeled_demos,
            )
        else:
            return optimizer.compile(
                module_class(),
                trainset=self.data_loader.trainset,
            )

    async def evaluate(
        self,
        module: dspy.Module,
        devset: Sequence[Example],
    ) -> float:
        """Evaluate module on a dataset asynchronously.

        Args:
            module: DSPy module to evaluate.
            devset: Evaluation dataset.

        Returns:
            Accuracy percentage (0-100).
        """

        async def evaluate_one(example: Example) -> bool:
            try:
                pred = await module.acall(
                    context=example.context,
                    question=example.question,
                )
                return validate_answer(example, pred)
            except Exception:
                return False

        results = await asyncio.gather(*[evaluate_one(ex) for ex in devset])
        return sum(results) / len(devset) * 100

    async def run(
        self,
        optimizer_type: OptimizerType,
        module_class: Type[dspy.Module] = BasicQA,
        generate_summary: bool = True,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run the full optimization pipeline.

        Args:
            optimizer_type: Which optimizer to use.
            module_class: DSPy module class to optimize.
            generate_summary: Whether to generate LLM evolution summary.
            verbose: Whether to print progress messages.

        Returns:
            OptimizationResult with all metrics and data.
        """
        if not self._configured:
            self.configure()

        # Reset tracker for new run
        self.tracker.reset()
        self.tracker.open_jsonl()

        try:
            # Baseline evaluation
            if verbose:
                print("=" * 60)
                print("BASELINE EVALUATION")
                print("=" * 60)

            baseline_module = module_class()
            baseline_accuracy = await self.evaluate(
                baseline_module, self.data_loader.testset
            )

            if verbose:
                print(f"Baseline Accuracy: {baseline_accuracy:.1f}%\n")

            # Run optimization
            if verbose:
                print("=" * 60)
                print(f"RUNNING {optimizer_type.value.upper()} OPTIMIZATION")
                print("=" * 60)

            optimizer = self._create_optimizer(optimizer_type)
            optimized_module = self._compile_module(
                optimizer, optimizer_type, module_class
            )

            # Extract prompt info
            prompt_info = self._extract_prompt_info(optimized_module)

            # Evaluate optimized
            if verbose:
                print("\n" + "=" * 60)
                print("OPTIMIZED EVALUATION")
                print("=" * 60)

            optimized_accuracy = await self.evaluate(
                optimized_module, self.data_loader.testset
            )

            if verbose:
                print(f"Optimized Accuracy: {optimized_accuracy:.1f}%")

            # Get tracker summary
            tracker_summary = self.tracker.get_summary()

            # Build result
            result = OptimizationResult(
                optimizer=optimizer_type.value,
                baseline_accuracy=baseline_accuracy,
                optimized_accuracy=optimized_accuracy,
                improvement=optimized_accuracy - baseline_accuracy,
                instruction=prompt_info.get("instruction"),
                demos=prompt_info.get("demos", []),
                trials=tracker_summary["trials"],
                instruction_candidates=tracker_summary["instructions_proposed"],
            )

            # Generate summary if requested
            if generate_summary:
                if verbose:
                    print("\n" + "=" * 60)
                    print("GENERATING EVOLUTION SUMMARY...")
                    print("=" * 60)
                result.evolution_summary = self._generate_summary(result)
                if verbose:
                    print("\nEvolution Summary:")
                    print(result.evolution_summary)

            # Print final comparison
            if verbose:
                print("\n" + "=" * 60)
                print("RESULTS COMPARISON")
                print("=" * 60)
                print(f"Baseline Accuracy:  {baseline_accuracy:.1f}%")
                print(f"Optimized Accuracy: {optimized_accuracy:.1f}%")
                print(f"Improvement:        {result.improvement:+.1f}%")
                print("=" * 60)

            return result

        finally:
            self.tracker.close_jsonl()

    def _extract_prompt_info(self, module: dspy.Module) -> Dict[str, Any]:
        """Extract optimized instruction and demos from compiled module.

        Args:
            module: Compiled DSPy module.

        Returns:
            Dictionary with 'instruction' and 'demos' keys.
        """
        predictor = module.generate_answer

        info: Dict[str, Any] = {
            "instruction": None,
            "demos": [],
        }

        # Try extended_signature.instructions (MIPROv2, COPRO)
        instruction = None
        if hasattr(predictor, "extended_signature"):
            sig = predictor.extended_signature
            if hasattr(sig, "instructions"):
                instruction = sig.instructions

        # Try signature.instructions (GEPA, some optimizers)
        if not instruction and hasattr(predictor, "signature"):
            sig = predictor.signature
            if hasattr(sig, "instructions"):
                instruction = sig.instructions

        # Try predictor.predict.signature for GEPA
        if not instruction and hasattr(predictor, "predict"):
            if hasattr(predictor.predict, "signature"):
                sig = predictor.predict.signature
                if hasattr(sig, "instructions"):
                    instruction = sig.instructions
            if hasattr(predictor.predict, "extended_signature"):
                sig = predictor.predict.extended_signature
                if hasattr(sig, "instructions"):
                    instruction = sig.instructions

        info["instruction"] = instruction

        # Extract few-shot demos
        if hasattr(predictor, "demos") and predictor.demos:
            for demo in predictor.demos:
                demo_info: Dict[str, str] = {
                    "question": (
                        demo.question[: PROCESSING_CONFIG.demo_question_limit] + "..."
                        if len(demo.question) > PROCESSING_CONFIG.demo_question_limit
                        else demo.question
                    ),
                    "answer": demo.answer,
                }
                if hasattr(demo, "reasoning"):
                    demo_info["reasoning"] = (
                        demo.reasoning[: PROCESSING_CONFIG.demo_reasoning_limit] + "..."
                        if len(demo.reasoning) > PROCESSING_CONFIG.demo_reasoning_limit
                        else demo.reasoning
                    )
                info["demos"].append(demo_info)

        return info

    def _generate_summary(self, result: OptimizationResult) -> str:
        """Generate LLM summary of optimization evolution.

        Args:
            result: OptimizationResult with trials and instruction candidates.

        Returns:
            Generated summary text.
        """
        # Deduplicate instruction candidates
        seen: set = set()
        unique_candidates: List[Dict] = []
        for cand in result.instruction_candidates:
            key = (cand.get("iteration", cand["index"]), cand["instruction"][:100])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(cand)

        # Sort by iteration
        unique_candidates.sort(key=lambda x: x.get("iteration", x["index"]))

        # Truncate individual instructions
        for cand in unique_candidates:
            if (
                len(cand["instruction"])
                > PROCESSING_CONFIG.instruction_truncation_limit
            ):
                cand["instruction"] = (
                    cand["instruction"][
                        : PROCESSING_CONFIG.instruction_truncation_limit
                    ]
                    + "... [truncated]"
                )

        # Sample evenly if total chars exceed budget
        total_chars = sum(len(c["instruction"]) for c in unique_candidates)
        if (
            total_chars > PROCESSING_CONFIG.max_instruction_chars
            and len(unique_candidates) > 1
        ):
            avg_chars = total_chars / len(unique_candidates)
            max_candidates = int(PROCESSING_CONFIG.max_instruction_chars / avg_chars)
            max_candidates = max(max_candidates, 2)

            step = len(unique_candidates) / max_candidates
            sampled_indices = [int(i * step) for i in range(max_candidates)]
            unique_candidates = [unique_candidates[i] for i in sampled_indices]

        # Build instruction history
        instruction_history = []
        for cand in unique_candidates:
            iteration = cand.get("iteration", cand["index"])
            instruction_history.append(
                f"## Iteration {iteration}\n{cand['instruction']}"
            )

        # Deduplicate and sort trials
        seen_trials: set = set()
        unique_trials: List[Dict] = []
        for trial in result.trials:
            key = (trial["trial"], trial["score"])
            if key not in seen_trials:
                seen_trials.add(key)
                unique_trials.append(trial)
        unique_trials.sort(key=lambda x: x["trial"])

        # Format score progression
        score_progression = "\n".join(
            [f"Iteration {t['trial']}: {t['score']:.1f}%" for t in unique_trials]
        )

        # Create the prompt
        prompt = f"""Analyze the following DSPy GEPA optimization run and provide a concise summary of the instruction evolution.

## Optimization Results
- Baseline Accuracy (before optimization): {result.baseline_accuracy:.1f}%
- Optimized Accuracy (after optimization): {result.optimized_accuracy:.1f}%
- Improvement: {result.improvement:+.1f}%

## Score Progression Per Iteration
{score_progression}

## Instruction Proposals (in chronological order)
{chr(10).join(instruction_history)}

## Final Optimized Instruction
{result.instruction if result.instruction else "(Not extracted)"}

---

Please provide a summary that covers:
1. **Overall Trajectory**: How did the optimization progress? Did scores improve consistently or plateau?
2. **What Worked**: What instruction patterns or strategies led to score improvements?
3. **What Didn't Work**: What approaches were tried but didn't help (or hurt performance)?
4. **Key Patterns**: Any notable trends in how instructions evolved (e.g., increasing specificity, adding constraints, etc.)
5. **Recommendations**: Based on this evolution, what might improve results further?

Keep the summary concise (3-5 paragraphs) and actionable."""

        # Use DSPy LM to generate the summary
        summary_lm = dspy.LM(
            self.model_name, temperature=MODEL_CONFIG.summary_temperature
        )

        try:
            response = summary_lm(prompt)
            return response if isinstance(response, str) else response[0]
        except Exception as e:
            return f"Error generating summary: {e}"
