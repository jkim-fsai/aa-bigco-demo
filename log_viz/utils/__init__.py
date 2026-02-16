"""Utilities for log_viz visualization."""

from .config import (
    BASE_DIR,
    CACHE_TTL_HISTORICAL,
    CACHE_TTL_JSONL,
    CACHE_TTL_RUNS,
    CARD_HEIGHT,
    COLORS,
    HISTORICAL_RESULTS,
    MAX_DISPLAY_TRIALS,
    MAX_INSTRUCTION_DISPLAY,
    PLOT_HEIGHT,
    REFRESH_INTERVAL_MS,
    RUNS_DIR,
)
from .instruction_grouping import (
    deduplicate_instructions,
    group_by_optimizer_type,
    sort_by_iteration,
)

__all__ = [
    # Config
    "BASE_DIR",
    "RUNS_DIR",
    "HISTORICAL_RESULTS",
    "REFRESH_INTERVAL_MS",
    "MAX_DISPLAY_TRIALS",
    "MAX_INSTRUCTION_DISPLAY",
    "PLOT_HEIGHT",
    "CARD_HEIGHT",
    "CACHE_TTL_RUNS",
    "CACHE_TTL_HISTORICAL",
    "CACHE_TTL_JSONL",
    "COLORS",
    # Instruction grouping
    "deduplicate_instructions",
    "group_by_optimizer_type",
    "sort_by_iteration",
]
