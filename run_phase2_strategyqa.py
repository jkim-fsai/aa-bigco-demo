#!/usr/bin/env python3
"""Run Phase 2 hyperparameter sweep for StrategyQA (issue #9).

Runs 6 GEPA configs exploring budget x coach x minibatch dimensions.
Patches pipeline-level configs before each run to override hyperparameters.

Usage:
    python run_phase2_strategyqa.py           # Run all 6 configs
    python run_phase2_strategyqa.py 3 5 7     # Run specific configs by number
"""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import dspy_demo.pipeline as pipeline_module
from dspy_demo import (
    STRATEGYQA_DATASET_CONFIG,
    BooleanQA,
    OptimizationPipeline,
    OptimizerType,
    gepa_boolean_metric,
    validate_boolean_answer,
)
from dspy_demo.config import ModelConfig, OptimizerConfig
from dspy_demo.core import DataLoader

# Phase 2: 6 remaining cells of the factorial design
# Phase 1 already ran: #1 GEPA heavy/nano/3, #2 MIPROv2 heavy/nano
PHASE2_CONFIGS = {
    "3": {"auto": "medium", "reflection_model": "openai/gpt-4.1-nano", "minibatch": 3},
    "4": {"auto": "medium", "reflection_model": "openai/gpt-4.1-nano", "minibatch": 16},
    "5": {"auto": "medium", "reflection_model": "openai/gpt-4.1", "minibatch": 3},
    "6": {"auto": "heavy", "reflection_model": "openai/gpt-4.1-nano", "minibatch": 16},
    "7": {"auto": "heavy", "reflection_model": "openai/gpt-4.1", "minibatch": 3},
    "8": {"auto": "heavy", "reflection_model": "openai/gpt-4.1", "minibatch": 16},
}


async def run_config(label: str, config: dict) -> dict:
    """Run a single GEPA config and return summary."""
    print(f"\n{'=' * 60}")
    print(
        f"PHASE 2 - Config #{label}: auto={config['auto']}, "
        f"reflection={config['reflection_model']}, minibatch={config['minibatch']}"
    )
    print(f"{'=' * 60}\n")

    # Patch global configs used by the pipeline
    pipeline_module.OPTIMIZER_CONFIG = replace(
        pipeline_module.OPTIMIZER_CONFIG,
        gepa_auto=config["auto"],
        gepa_reflection_minibatch_size=config["minibatch"],
    )
    pipeline_module.MODEL_CONFIG = replace(
        pipeline_module.MODEL_CONFIG,
        reflection_model=config["reflection_model"],
    )

    data_loader = DataLoader(config=STRATEGYQA_DATASET_CONFIG)
    pipeline = OptimizationPipeline(
        data_loader=data_loader,
        metric=validate_boolean_answer,
        gepa_metric_fn=gepa_boolean_metric,
    )
    result = await pipeline.run(
        OptimizerType.GEPA,
        module_class=BooleanQA,
        verbose=True,
    )

    path = Path(f"optimization_results_strategyqa_gepa_{label}.json")
    result.save(path)
    print(f"\nConfig #{label} results saved to: {path}")

    return {
        "label": label,
        "baseline": result.baseline_accuracy,
        "optimized": result.optimized_accuracy,
        "improvement": result.improvement,
    }


async def main(config_labels: list[str]) -> None:
    """Run specified configs sequentially."""
    results = []
    for label in config_labels:
        config = PHASE2_CONFIGS.get(label)
        if not config:
            print(f"Unknown config: {label}. Available: {list(PHASE2_CONFIGS.keys())}")
            continue
        summary = await run_config(label, config)
        results.append(summary)

    # Print summary table
    print(f"\n{'=' * 60}")
    print("PHASE 2 SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(
            f"  #{r['label']}: {r['baseline']:.1f}% -> {r['optimized']:.1f}% "
            f"({r['improvement']:+.1f}%)"
        )


if __name__ == "__main__":
    labels = sys.argv[1:] if len(sys.argv) > 1 else list(PHASE2_CONFIGS.keys())
    asyncio.run(main(labels))
