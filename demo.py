#!/usr/bin/env python3
"""DSPy optimization demo with GEPA.

This script demonstrates DSPy's GEPA optimizer on the HotPotQA dataset.
Run directly to execute the full optimization pipeline.

Usage:
    python demo.py
"""

import asyncio

from dspy_demo import OptimizationPipeline, OptimizerType


async def main() -> None:
    """Run GEPA optimization pipeline."""
    pipeline = OptimizationPipeline()
    result = await pipeline.run(OptimizerType.GEPA, verbose=True)

    # Save results
    path = result.save()
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    asyncio.run(main())
