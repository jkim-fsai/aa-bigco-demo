"""Shared instruction grouping utilities.

This module provides functions for deduplicating and grouping instruction
candidates, used by both instruction_viewer.py and compare_optimizers.py.
"""

from typing import Any, Dict, List, Set, Tuple


def deduplicate_instructions(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deduplicate instruction candidates by (iteration, instruction prefix).

    Some optimizers log the same instruction multiple times. This function
    removes duplicates while preserving order.

    Args:
        candidates: List of instruction candidate dictionaries.

    Returns:
        Deduplicated list of candidates.
    """
    seen: Set[Tuple[int, str]] = set()
    unique: List[Dict[str, Any]] = []

    for cand in candidates:
        iteration = cand.get("iteration", cand.get("index", 0))
        instruction_prefix = cand.get("instruction", "")[:100]
        key = (iteration, instruction_prefix)

        if key not in seen:
            seen.add(key)
            unique.append(cand)

    return unique


def group_by_optimizer_type(
    candidates: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group instruction candidates by optimizer type.

    Args:
        candidates: List of instruction candidate dictionaries.

    Returns:
        Dictionary with 'gepa', 'mipro', and 'other' keys containing
        lists of candidates for each optimizer type.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {
        "gepa": [],
        "mipro": [],
        "other": [],
    }

    for cand in candidates:
        opt_type = cand.get("type", "other").lower()
        if opt_type in groups:
            groups[opt_type].append(cand)
        else:
            groups["other"].append(cand)

    return groups


def sort_by_iteration(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sort candidates by iteration number.

    Args:
        candidates: List of instruction candidate dictionaries.

    Returns:
        Sorted list of candidates.
    """
    return sorted(
        candidates,
        key=lambda x: x.get("iteration", x.get("index", 0)),
    )
