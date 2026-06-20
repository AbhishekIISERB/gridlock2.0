"""
06_run_pipeline.py — Orchestrator: ingestion → scoring → alerts in a loop
Usage:
  python pipeline/06_run_pipeline.py                  # default: batch=500, tick=30s
  python pipeline/06_run_pipeline.py --demo_speed     # batch=5000, tick=10s
  python pipeline/06_run_pipeline.py --batch_size 1000 --tick_sec 20
"""
import time, sys, os, argparse, sqlite3, subprocess
from datetime import datetime

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(PIPELINE_DIR, '..', 'db', 'gridlock.db')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',  type=int, default=500)
parser.add_argument('--tick_sec',    type=int, default=30)
parser.add_argument('--window_days', type=int, default=30)
parser.add_argument('--demo_speed',  action='store_true',
                    help='batch_size=5000, tick_sec=10 for demo')
args = parser.parse_args()

if args.demo_speed:
    args.batch_size = 5000
    args.tick_sec   = 10

# Add pipeline dir to path so we can import score/alert functions
sys.path.insert(0, PIPELINE_DIR)
from importlib.util import spec_from_file_location, module_from_spec

def _load(fname):
    path = os.path.join(PIPELINE_DIR, fname)
    spec = spec_from_file_location(fname, path)
    mod  = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

score_mod = _load('04_score_engine.py')
alert_mod = _load('05_alert_engine.py')

print("=" * 60)
print("GRIDLOCK AI — LIVE PIPELINE ORCHESTRATOR")
print(f"  batch_size  : {args.batch_size:,} violations/tick")
print(f"  tick_sec    : {args.tick_sec}s")
print(f"  window_days : {args.window_days} days rolling window")
print("=" * 60)

tick = 0
while True:
    tick += 1
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ══════ TICK {tick} ══════")

    # ── Step 1: Ingest next batch (subprocess to avoid CSV reload cost) ──
    try:
        result = subprocess.run(
            [sys.executable,
             os.path.join(PIPELINE_DIR, '03_ingestion_sim.py'),
             '--batch_size', str(args.batch_size)],
            capture_output=True, text=True, cwd=os.path.join(PIPELINE_DIR, '..')
        )
        out = result.stdout.strip()
        if out: print(out)
        if result.returncode != 0 and result.stderr:
            print(f"  [ingest ERR] {result.stderr.strip()[:200]}")
    except Exception as e:
        print(f"  [ingest] Exception: {e}")

    # ── Step 2: Recalculate scores ────────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        score_mod.run_score_engine(window_days=args.window_days, conn=conn)
        conn.close()
    except Exception as e:
        print(f"  [score_engine] {e}")

    # ── Step 3: Check for alerts ──────────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        alert_mod.run_alert_engine(conn=conn)
        conn.close()
    except Exception as e:
        print(f"  [alert_engine] {e}")

    print(f"  Sleeping {args.tick_sec}s ...")
    time.sleep(args.tick_sec)
