#!/usr/bin/env python3
"""Run DSPy optimization with selectable optimizer and dataset.

This script allows comparing optimization results between different
DSPy optimizers (GEPA, MIPROv2) on different datasets (HotPotQA, StrategyQA).

Usage:
    python demo_compare.py [optimizer] [dataset]

    optimizer: gepa (default), mipro, or miprov2
    dataset:   hotpotqa (default) or strategyqa
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

from dspy_demo import (
    DATASET_CONFIG,
    STRATEGYQA_DATASET_CONFIG,
    BasicQA,
    BooleanQA,
    OptimizationPipeline,
    OptimizerType,
    gepa_boolean_metric,
    gepa_metric,
    validate_answer,
    validate_boolean_answer,
)
from dspy_demo.core import DataLoader

OPTIMIZER_MAP = {
    "gepa": OptimizerType.GEPA,
    "mipro": OptimizerType.MIPRO,
    "miprov2": OptimizerType.MIPROV2,
}

DATASET_MAP: Dict[str, Dict[str, Any]] = {
    "hotpotqa": {
        "config": DATASET_CONFIG,
        "module": BasicQA,
        "metric": validate_answer,
        "gepa_metric": gepa_metric,
    },
    "strategyqa": {
        "config": STRATEGYQA_DATASET_CONFIG,
        "module": BooleanQA,
        "metric": validate_boolean_answer,
        "gepa_metric": gepa_boolean_metric,
    },
}


async def main(optimizer_name: str, dataset_name: str) -> None:
    """Run optimization with specified optimizer and dataset.

    Args:
        optimizer_name: Name of optimizer to use (gepa, mipro, miprov2).
        dataset_name: Name of dataset to use (hotpotqa, strategyqa).
    """
    optimizer_type = OPTIMIZER_MAP.get(optimizer_name.lower())
    if not optimizer_type:
        print(f"Unknown optimizer: {optimizer_name}")
        print(f"Available: {', '.join(OPTIMIZER_MAP.keys())}")
        sys.exit(1)

    dataset_info = DATASET_MAP.get(dataset_name.lower())
    if not dataset_info:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {', '.join(DATASET_MAP.keys())}")
        sys.exit(1)

    data_loader = DataLoader(config=dataset_info["config"])
    pipeline = OptimizationPipeline(
        data_loader=data_loader,
        metric=dataset_info["metric"],
        gepa_metric_fn=dataset_info["gepa_metric"],
    )
    result = await pipeline.run(
        optimizer_type,
        module_class=dataset_info["module"],
        verbose=True,
    )

    path = Path(
        f"optimization_results_{dataset_name.lower()}_{optimizer_name.lower()}.json"
    )
    result.save(path)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    optimizer = sys.argv[1] if len(sys.argv) > 1 else "gepa"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "hotpotqa"

    if optimizer.lower() not in OPTIMIZER_MAP:
        print(
            f"Usage: python demo_compare.py [{' | '.join(OPTIMIZER_MAP.keys())}]"
            f" [{' | '.join(DATASET_MAP.keys())}]"
        )
        print("  gepa        - Use GEPA optimizer (evolutionary search)")
        print("  mipro       - Use MIPROv2 optimizer (instruction + few-shot)")
        print("  hotpotqa    - Multi-hop reading comprehension (default)")
        print("  strategyqa  - Implicit boolean reasoning")
        sys.exit(1)

    asyncio.run(main(optimizer, dataset))
