#!/bin/bash
set -e

# Run DB seeding first (idempotent, skips if already seeded)
echo "Ensuring DB is seeded..."
python pipeline/01_init_db.py
python pipeline/02_seed_db.py --days 7 || true
python pipeline/04_score_engine.py || true

# Start the continuous background pipeline (live simulator + orchestrator)
echo "Starting background pipeline orchestrator..."
python pipeline/06_run_pipeline.py --demo_speed > pipeline_output.log 2>&1 &

# Start the FastAPI server on port 7860 (Hugging Face Spaces default port)
# The API also serves the frontend from the root `/` path
echo "Starting FastAPI Server..."
exec uvicorn api:app --host 0.0.0.0 --port 7860
