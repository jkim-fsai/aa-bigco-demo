"""
Run DSPy optimization with selectable optimizer (GEPA or MIPROv2).
This allows comparing instruction evolution between different optimizers.
"""
import asyncio
import sys

from demo import (
    BasicQA,
    OptimizationTracker,
    evaluate_async,
    extract_optimized_prompt_info,
    gepa_metric,
    optimization_tracker,
    print_optimization_progress,
    print_prompt_evolution,
    testset,
    trainset,
    validate_answer,
)
import json
import logging
from pathlib import Path

import dspy


async def run_optimization(optimizer_name: str = "gepa"):
    """Run optimization with specified optimizer."""

    # Reset tracker for new run
    optimization_tracker.trials = []
    optimization_tracker.instructions = []

    # Open JSONL file for real-time logging
    optimization_tracker.open_jsonl()

    try:
        # Capture baseline instruction
        baseline_qa = BasicQA()
        baseline_instruction = "Given the fields `context`, `question`, produce the fields `answer`."

        # Step 1: Evaluate baseline on TEST set (unoptimized)
        print("=" * 60)
        print("BASELINE EVALUATION (no optimization)")
        print("=" * 60)
        baseline_accuracy = await evaluate_async(baseline_qa, testset)
        print(f"Baseline Accuracy: {baseline_accuracy:.1f}%\n")

        # Step 2: Run selected optimizer
        print("=" * 60)
        print(f"RUNNING {optimizer_name.upper()} OPTIMIZATION")
        print("=" * 60)

        if optimizer_name.lower() == "gepa":
            print("Optimizing instructions via evolutionary search...")
            print("(Watch the logs below to see prompt evolution)\n")

            # GEPA needs a separate LM for reflection/instruction proposal
            reflection_lm = dspy.LM("openai/gpt-4.1-nano", temperature=1.0)

            optimizer = dspy.GEPA(
                metric=gepa_metric,
                auto="light",
                reflection_lm=reflection_lm,
                num_threads=10,
            )
            optimized_qa = optimizer.compile(BasicQA(), trainset=trainset)

        elif optimizer_name.lower() == "mipro" or optimizer_name.lower() == "miprov2":
            print("Optimizing with MIPROv2 (instructions + few-shot)...")
            print("(Watch the logs below to see prompt evolution)\n")

            optimizer = dspy.MIPROv2(
                metric=validate_answer,
                auto="light",
                num_threads=10,
            )
            optimized_qa = optimizer.compile(
                BasicQA(),
                trainset=trainset,
                num_trials=30,
                max_bootstrapped_demos=3,
                max_labeled_demos=3,
            )

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Use 'gepa' or 'mipro'.")

        # Step 3: Display per-iteration metrics
        print_optimization_progress(optimization_tracker)

        # Step 4: Extract and display prompt evolution
        optimized_info = extract_optimized_prompt_info(optimized_qa)
        print_prompt_evolution(baseline_instruction, optimized_info)

        # Include iteration metrics in the saved results
        tracker_summary = optimization_tracker.get_summary()
        optimized_info["optimization_trials"] = tracker_summary["trials"]
        optimized_info["instruction_candidates"] = tracker_summary["instructions_proposed"]
        optimized_info["optimizer"] = optimizer_name.lower()

        # Save optimization results to JSON for documentation
        results_file = Path(f"optimization_results_{optimizer_name.lower()}.json")
        results_file.write_text(json.dumps(optimized_info, indent=2))
        print(f"\nOptimization details saved to: {results_file}")

        # Step 5: Evaluate optimized module on TEST set (held-out)
        print("\n" + "=" * 60)
        print("OPTIMIZED EVALUATION (on held-out test set)")
        print("=" * 60)
        optimized_accuracy = await evaluate_async(optimized_qa, testset)
        print(f"Optimized Accuracy: {optimized_accuracy:.1f}%\n")

        # Step 6: Show comparison
        print("=" * 60)
        print("RESULTS COMPARISON")
        print("=" * 60)
        print(f"Optimizer:          {optimizer_name.upper()}")
        print(f"Baseline Accuracy:  {baseline_accuracy:.1f}%")
        print(f"Optimized Accuracy: {optimized_accuracy:.1f}%")
        improvement = optimized_accuracy - baseline_accuracy
        print(f"Improvement:        {improvement:+.1f}%")
        print("=" * 60)

        # Update results file with final scores
        optimized_info["baseline_accuracy"] = baseline_accuracy
        optimized_info["optimized_accuracy"] = optimized_accuracy
        optimized_info["improvement"] = improvement
        results_file.write_text(json.dumps(optimized_info, indent=2))

    finally:
        # Close JSONL file
        optimization_tracker.close_jsonl()


if __name__ == "__main__":
    # Get optimizer from command line argument
    optimizer = sys.argv[1] if len(sys.argv) > 1 else "gepa"

    if optimizer.lower() not in ["gepa", "mipro", "miprov2"]:
        print("Usage: python demo_compare.py [gepa|mipro]")
        print("  gepa   - Use GEPA optimizer (evolutionary search)")
        print("  mipro  - Use MIPROv2 optimizer (instruction + few-shot)")
        sys.exit(1)

    asyncio.run(run_optimization(optimizer))
