#!/usr/bin/env python3
"""Run DSPy optimization with selectable optimizer.

This script allows comparing optimization results between different
DSPy optimizers (GEPA, MIPROv2).

Usage:
    python demo_compare.py [optimizer]

    optimizer: gepa (default), mipro, or miprov2
"""

import asyncio
import sys

from dspy_demo import OptimizationPipeline, OptimizerType


async def main(optimizer_name: str) -> None:
    """Run optimization with specified optimizer.

    Args:
        optimizer_name: Name of optimizer to use (gepa, mipro, miprov2).
    """
    optimizer_map = {
        "gepa": OptimizerType.GEPA,
        "mipro": OptimizerType.MIPRO,
        "miprov2": OptimizerType.MIPROV2,
    }

    optimizer_type = optimizer_map.get(optimizer_name.lower())
    if not optimizer_type:
        print(f"Unknown optimizer: {optimizer_name}")
        print("Available: gepa, mipro, miprov2")
        sys.exit(1)

    pipeline = OptimizationPipeline()
    result = await pipeline.run(optimizer_type, verbose=True)

    # Save results with optimizer-specific filename
    from pathlib import Path
    path = Path(f"optimization_results_{optimizer_name.lower()}.json")
    result.save(path)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    optimizer = sys.argv[1] if len(sys.argv) > 1 else "gepa"

    if optimizer.lower() not in ["gepa", "mipro", "miprov2"]:
        print("Usage: python demo_compare.py [gepa|mipro]")
        print("  gepa   - Use GEPA optimizer (evolutionary search)")
        print("  mipro  - Use MIPROv2 optimizer (instruction + few-shot)")
        sys.exit(1)

    asyncio.run(main(optimizer))
