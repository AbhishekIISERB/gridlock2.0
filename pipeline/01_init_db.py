"""
01_init_db.py — Create SQLite schema for Gridlock live pipeline
Run once: python pipeline/01_init_db.py
"""
import sqlite3, os

DB_DIR  = os.path.join(os.path.dirname(__file__), '..', 'db')
DB_PATH = os.path.join(DB_DIR, 'gridlock.db')

os.makedirs(DB_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

# WAL mode for concurrent reads during writes
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")

conn.executescript("""
-- Raw violations as they arrive (simulated stream)
CREATE TABLE IF NOT EXISTS violations (
    id                TEXT PRIMARY KEY,
    latitude          REAL,
    longitude         REAL,
    vehicle_type      TEXT,
    violation_type    TEXT,
    police_station    TEXT,
    junction_name     TEXT,
    hour              INTEGER,
    is_peak           INTEGER,
    lanes_blocked     REAL,
    vtype_severity    REAL,
    compound_score    REAL,
    impact            REAL,
    validation_weight REAL,
    scita_weight      REAL,
    cluster           INTEGER,
    created_datetime  TEXT,
    ingested_at       TEXT
);

-- Cluster scores recalculated on rolling 30-day window
CREATE TABLE IF NOT EXISTS cluster_scores (
    cluster           INTEGER,
    recalculated_at   TEXT,
    violation_count   INTEGER,
    total_impact      REAL,
    avg_lanes_blocked REAL,
    peak_pct          REAL,
    active_days       INTEGER,
    active_weeks      INTEGER,
    approved_pct      REAL,
    scita_pct         REAL,
    compound_pct      REAL,
    congestion_score  REAL,
    risk_tier         TEXT,
    zone_name         TEXT,
    police_station    TEXT,
    daily_rate        REAL,
    blockage_pct      REAL,
    enforcement_shift TEXT,
    PRIMARY KEY (cluster, recalculated_at)
);

-- Alert log — tier crossings
CREATE TABLE IF NOT EXISTS alerts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster        INTEGER,
    zone_name      TEXT,
    police_station TEXT,
    previous_tier  TEXT,
    new_tier       TEXT,
    congestion_score REAL,
    triggered_at   TEXT,
    acknowledged   INTEGER DEFAULT 0
);

-- Ingestion state — tracks replay position
CREATE TABLE IF NOT EXISTS ingestion_state (
    id                     INTEGER PRIMARY KEY DEFAULT 1,
    last_ingested_datetime TEXT,
    total_rows_ingested    INTEGER DEFAULT 0
);
""")

conn.commit()
conn.close()

print(f"✓ DB created at: {os.path.abspath(DB_PATH)}")
print("  Tables: violations, cluster_scores, alerts, ingestion_state")
