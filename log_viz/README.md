# DSPy Optimization Visualization Dashboard

A TensorBoard-like real-time visualization tool for DSPy optimization runs.

## Features

- **Real-time Updates**: Auto-refreshes every 2 seconds to show live optimization progress
- **TensorBoard-Style Plots**: Score over time, running maximum, distribution, and eval type comparison
- **Interactive Tables**: Sortable, filterable trials table with CSV export
- **Historical Comparison**: Overlay past optimization runs for comparison
- **Metrics Cards**: Summary statistics including best score, mean, std dev, and improvement
- **Instruction Evolution**: View proposed instruction candidates and few-shot demonstrations

## Installation

```bash
cd log_viz
pip install -r requirements.txt
```

## Usage

### 1. Start the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### 2. Run Optimization (in separate terminal)

```bash
cd ..
python demo.py
```

The dashboard will automatically detect the new JSONL file and start displaying trials in real-time.

## Dashboard Controls

### Sidebar
- **Auto-refresh**: Toggle automatic updates (default: ON)
- **Refresh interval**: Control update frequency (1-10 seconds)
- **Select Run**: Choose which optimization run to visualize
- **Historical comparison**: Overlay data from optimization_results.json
- **Force Refresh**: Manually clear cache and reload data

### Filters (Trials Table)
- **Evaluation Type**: Filter by minibatch vs full evaluation
- **Optimizer**: Filter by optimizer type (GEPA, MIPROv2, COPRO)
- **Score Range**: Slider to filter trials by score range

## File Structure

```
log_viz/
├── app.py                          # Main Streamlit dashboard
├── data_loader.py                  # JSONL + JSON reading logic
├── plots.py                        # Plotly visualizations
├── components/
│   ├── metrics_cards.py           # Summary metrics display
│   ├── trials_table.py            # Interactive table
│   └── instruction_viewer.py      # Instruction evolution
├── utils/
│   └── config.py                  # Configuration constants
├── runs/                          # Auto-generated trial data
│   └── trials_YYYYMMDD_HHMMSS.jsonl
└── requirements.txt               # Dependencies
```

## Data Flow

1. **demo.py** runs optimization and writes trials to `runs/trials_YYYYMMDD_HHMMSS.jsonl` in real-time
2. **app.py** reads JSONL incrementally (only new lines since last read)
3. Dashboard updates every 2 seconds with new trials
4. **optimization_results.json** can be loaded for historical comparison

## Visualizations

### Score Over Time
Line plot showing trial scores as they're evaluated. Supports overlay of historical runs.

### Running Maximum
Filled area chart showing the best score achieved so far at each iteration.

### Score Distribution
Histogram showing the distribution of trial scores across the optimization run.

### Eval Type Comparison
Box plot comparing scores for minibatch vs full evaluations (MIPROv2 only).

## Troubleshooting

### No runs found
- Ensure `log_viz/runs/` directory exists
- Run `python demo.py` to generate trial data

### Dashboard not updating
- Check that auto-refresh is enabled in sidebar
- Click "Force Refresh" button to manually reload
- Verify `demo.py` is writing to JSONL (check file size growing)

### Import errors
- Ensure you're running from `log_viz/` directory: `streamlit run app.py`
- Verify all dependencies installed: `pip install -r requirements.txt`

## Performance

- **Incremental Reading**: Only reads new JSONL lines, not entire file
- **Caching**: Uses Streamlit's `@st.cache_data` for efficient reloads
- **Table Pagination**: Limits displayed trials to 1000 rows for performance
- **No API Polling**: Zero additional LLM calls beyond optimization itself

## Customization

Edit `utils/config.py` to customize:
- Refresh interval
- Plot colors (TensorBoard-inspired by default)
- Plot height
- Maximum displayed trials
