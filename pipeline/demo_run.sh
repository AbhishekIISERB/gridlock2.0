#!/bin/bash
# demo_run.sh — One-command demo launcher
# Run from gridlock/ directory: bash pipeline/demo_run.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

echo "═══════════════════════════════════════════════"
echo "  GRIDLOCK AI — LIVE PIPELINE DEMO"
echo "═══════════════════════════════════════════════"

cd "$ROOT"
source venv/bin/activate 2>/dev/null || true

# Step 1: Init DB (idempotent)
echo ""
echo "▶ Step 1/4 — Initialising database..."
python pipeline/01_init_db.py

# Step 2: Seed with first 7 days only (live sim adds the rest)
echo ""
echo "▶ Step 2/4 — Seeding DB with first 7 days of data..."
python pipeline/02_seed_db.py --days 7

# Step 3: Run initial score calculation
echo ""
echo "▶ Step 3/4 — Running initial score calculation..."
python pipeline/04_score_engine.py

# Step 4: Start pipeline in background
echo ""
echo "▶ Step 4/4 — Starting live pipeline (demo speed: 5000 violations/tick, 10s interval)..."
python pipeline/06_run_pipeline.py --demo_speed &
PIPELINE_PID=$!
echo "  Pipeline PID: $PIPELINE_PID"

# Give pipeline one tick to generate some data
sleep 12

# Step 5: Launch live dashboard
echo ""
echo "▶ Launching Streamlit dashboard..."
echo "  Open http://localhost:8501 in your browser"
echo "  Press Ctrl+C to stop"
echo ""
streamlit run streamlit_live.py

# Cleanup
echo ""
echo "Stopping pipeline (PID $PIPELINE_PID)..."
kill $PIPELINE_PID 2>/dev/null || true
echo "Done."
