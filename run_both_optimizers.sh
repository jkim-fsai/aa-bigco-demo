#!/bin/bash
# Run both optimizers and compare results

echo "=========================================="
echo "Running GEPA Optimization"
echo "=========================================="
.venv/bin/python demo_compare.py gepa

echo ""
echo "=========================================="
echo "Running MIPROv2 Optimization"
echo "=========================================="
.venv/bin/python demo_compare.py mipro

echo ""
echo "=========================================="
echo "Both optimizations complete!"
echo "Check the dashboard at http://localhost:8501"
echo "=========================================="
