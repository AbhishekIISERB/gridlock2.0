"""
02_seed_db.py — Load parking_clustered.csv into violations table
Run once after 01_init_db.py: python pipeline/02_seed_db.py [--days N]

--days N  : only seed the first N days of data (default: all)
            Use --days 7 so the live simulator has fresh data to add.
"""
import sqlite3, pandas as pd, os, argparse
from datetime import datetime, timedelta

BATCH  = 10_000
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db', 'gridlock.db')
CSV     = os.path.join(os.path.dirname(__file__), '..', 'parking_clustered.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--days', type=int, default=None,
                    help='Only seed first N days (leave empty for all rows)')
args = parser.parse_args()

print("Loading parking_clustered.csv ...")
df = pd.read_csv(CSV, low_memory=False)
print(f"  {len(df):,} rows loaded")

# Normalise datetime
df['created_datetime'] = pd.to_datetime(df['created_datetime'], utc=True, errors='coerce')
df = df[df['created_datetime'].notna()].copy()
df = df.sort_values('created_datetime')

if args.days:
    cutoff = df['created_datetime'].min() + timedelta(days=args.days)
    df = df[df['created_datetime'] <= cutoff]
    print(f"  Seeding first {args.days} days → {len(df):,} rows (cutoff: {cutoff})")

# Columns we need
COLS = [
    'id', 'latitude', 'longitude', 'vehicle_type', 'violation_type',
    'police_station', 'junction_name', 'hour', 'is_peak',
    'lanes_blocked', 'vtype_severity', 'compound_score', 'impact',
    'validation_weight', 'scita_weight', 'cluster', 'created_datetime',
]
# Fill missing score columns with defaults
for col, default in [('validation_weight', 1.0), ('scita_weight', 1.0),
                     ('lanes_blocked', 1.0), ('vtype_severity', 1.0),
                     ('compound_score', 1.0), ('impact', 1.0)]:
    if col not in df.columns:
        df[col] = default

df = df[[c for c in COLS if c in df.columns]].copy()

# ingested_at = created_datetime for historical seed
df['ingested_at'] = df['created_datetime'].astype(str)
df['created_datetime'] = df['created_datetime'].astype(str)

earliest = df['created_datetime'].min()

conn = sqlite3.connect(DB_PATH, timeout=10)
conn.execute("PRAGMA journal_mode=WAL")

# Clear existing data
conn.execute("DELETE FROM violations")
conn.execute("DELETE FROM ingestion_state")
conn.commit()

total = len(df)
inserted = 0
for start in range(0, total, BATCH):
    chunk = df.iloc[start:start + BATCH]
    chunk.to_sql('violations', conn, if_exists='append', index=False)
    inserted += len(chunk)
    if inserted % 50_000 == 0 or inserted == total:
        print(f"  Inserted {inserted:,} / {total:,} rows")

# Write initial ingestion state
conn.execute("""
    INSERT OR REPLACE INTO ingestion_state (id, last_ingested_datetime, total_rows_ingested)
    VALUES (1, ?, ?)
""", (earliest, inserted))
conn.commit()
conn.close()

print(f"\n✓ Seed complete — {inserted:,} violations in DB")
print(f"  ingestion_state.last_ingested_datetime = {earliest}")
