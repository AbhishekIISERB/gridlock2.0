"""
03_ingestion_sim.py — Replay historical violations as a live stream
Usage: python pipeline/03_ingestion_sim.py [--batch_size N]

Each call ingests the next BATCH_SIZE violations (ordered by created_datetime)
and stamps them with the current wall-clock time as ingested_at.
"""
import sqlite3, pandas as pd, os, argparse
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db', 'gridlock.db')
CSV     = os.path.join(os.path.dirname(__file__), '..', 'parking_clustered.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=500,
                    help='Violations to ingest per tick (default 500, use 5000 for demo)')
args = parser.parse_args()

conn = sqlite3.connect(DB_PATH, timeout=10)
conn.execute("PRAGMA journal_mode=WAL")

# Read current ingestion state
state = pd.read_sql("SELECT * FROM ingestion_state WHERE id=1", conn)
if state.empty:
    print("ERROR: ingestion_state is empty — run 02_seed_db.py first")
    conn.close()
    exit(1)

last_dt       = state.iloc[0]['last_ingested_datetime']
total_so_far  = int(state.iloc[0]['total_rows_ingested'])

print(f"  Resuming from: {last_dt} (total ingested so far: {total_so_far:,})")

# Load CSV (only columns we need, filter to rows after last checkpoint)
COLS = [
    'id', 'latitude', 'longitude', 'vehicle_type', 'violation_type',
    'police_station', 'junction_name', 'hour', 'is_peak',
    'lanes_blocked', 'vtype_severity', 'compound_score', 'impact',
    'validation_weight', 'scita_weight', 'cluster', 'created_datetime',
]
df = pd.read_csv(CSV, usecols=[c for c in COLS if c in
                  pd.read_csv(CSV, nrows=0).columns], low_memory=False)

# Fill missing score columns
for col, default in [('validation_weight', 1.0), ('scita_weight', 1.0),
                     ('lanes_blocked', 1.0), ('vtype_severity', 1.0),
                     ('compound_score', 1.0), ('impact', 1.0)]:
    if col not in df.columns:
        df[col] = default

df['created_datetime'] = pd.to_datetime(df['created_datetime'], utc=True, errors='coerce')
df = df[df['created_datetime'].notna()].sort_values('created_datetime')

# Filter to rows after the last checkpoint
df = df[df['created_datetime'].astype(str) > last_dt]

if df.empty:
    print("  ✓ All violations already ingested — replay complete!")
    conn.close()
    exit(0)

# Take next batch
batch = df.head(args.batch_size).copy()
now_str = datetime.now(timezone.utc).isoformat()
batch['ingested_at']      = now_str
batch['created_datetime'] = batch['created_datetime'].astype(str)

# Insert into DB (ignore duplicates on id)
batch.to_sql('violations', conn, if_exists='append', index=False,
             method='multi')

new_last_dt    = batch['created_datetime'].max()
new_total      = total_so_far + len(batch)

conn.execute("""
    UPDATE ingestion_state
    SET last_ingested_datetime = ?, total_rows_ingested = ?
    WHERE id = 1
""", (new_last_dt, new_total))
conn.commit()
conn.close()

print(f"  ✓ Ingested {len(batch):,} new violations (total: {new_total:,})")
print(f"    New checkpoint: {new_last_dt}")
