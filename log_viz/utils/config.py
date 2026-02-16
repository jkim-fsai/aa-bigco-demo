"""Configuration constants for the visualization dashboard."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
RUNS_DIR = BASE_DIR / "runs"
HISTORICAL_RESULTS = BASE_DIR.parent / "optimization_results.json"

# UI Configuration
REFRESH_INTERVAL_MS = 2000  # Auto-refresh interval in milliseconds
MAX_DISPLAY_TRIALS = 1000  # Max rows in trials table
MAX_INSTRUCTION_DISPLAY = 10  # Max instruction proposals to show
PLOT_HEIGHT = 400
CARD_HEIGHT = 120

# Cache TTLs (seconds)
CACHE_TTL_RUNS = 2  # Available runs list cache
CACHE_TTL_HISTORICAL = 10  # Historical results cache
CACHE_TTL_JSONL = 10  # JSONL full load cache

# Color scheme (TensorBoard-inspired)
COLORS = {
    "primary": "#FF6F00",
    "secondary": "#0091EA",
    "success": "#00C853",
    "warning": "#FFD600",
    "error": "#D50000",
    "live_run": "#FF6F00",
    "historical": "#757575",
}
