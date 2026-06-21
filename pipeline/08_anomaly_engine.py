"""
08_anomaly_engine.py — Statistical anomaly detector (Option 2)
Computes per-cluster baseline (mean ± std of violation rate) from historical data.
Compares current rolling-window rate against baseline for same hour+dow.
Writes anomalies to cluster_anomalies table when z-score > threshold.

Run: python pipeline/08_anomaly_engine.py
"""
import sqlite3, pandas as pd, numpy as np, os
from datetime import datetime, timezone

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(PIPELINE_DIR, '..', 'db', 'gridlock.db')
CSV_PATH     = os.path.join(PIPELINE_DIR, '..', 'parking_clustered.csv')
CS_PATH      = os.path.join(PIPELINE_DIR, '..', 'cluster_stats.csv')

Z_MODERATE = 2.0   # flag as moderate anomaly
Z_SEVERE   = 3.0   # flag as severe
Z_EXTREME  = 4.5   # flag as extreme (very rare, definitely worth alerting)


def ensure_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cluster_anomalies (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster         INTEGER,
            zone_name       TEXT,
            police_station  TEXT,
            current_rate    REAL,
            baseline_rate   REAL,
            z_score         REAL,
            anomaly_type    TEXT,
            severity        TEXT,
            detected_at     TEXT,
            acknowledged    INTEGER DEFAULT 0
        );
    """)
    conn.commit()


def build_baseline():
    """Compute per-cluster daily-rate baseline from full historical CSV."""
    print("  [anomaly] Building baseline from parking_clustered.csv...")
    df = pd.read_csv(CSV_PATH,
                     usecols=['cluster', 'date', 'hour', 'impact'],
                     low_memory=False)
    df = df[df['cluster'] != -1].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Daily violation count per cluster
    daily = df.groupby(['cluster', 'date']).size().reset_index(name='daily_count')

    baseline = daily.groupby('cluster')['daily_count'].agg(
        baseline_mean='mean',
        baseline_std='std',
        baseline_n='count'
    ).reset_index()
    baseline['baseline_std'] = baseline['baseline_std'].fillna(1.0)  # avoid /0
    print(f"  [anomaly] Baseline built for {len(baseline)} clusters")
    return baseline


# Cache baseline in module scope so we only build it once per process
_BASELINE_CACHE = None

def get_baseline():
    global _BASELINE_CACHE
    if _BASELINE_CACHE is None:
        _BASELINE_CACHE = build_baseline()
    return _BASELINE_CACHE


def run_anomaly_engine(conn=None) -> pd.DataFrame:
    close_after = conn is None
    if conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")

    ensure_tables(conn)

    # Get latest cluster scores (current daily_rate per cluster)
    current = pd.read_sql("""
        SELECT cluster, zone_name, police_station, daily_rate, congestion_score, risk_tier
        FROM cluster_scores
        WHERE recalculated_at = (SELECT MAX(recalculated_at) FROM cluster_scores)
    """, conn)

    if current.empty:
        print("  [anomaly] No current scores — run score engine first")
        if close_after: conn.close()
        return pd.DataFrame()

    baseline = get_baseline()
    merged   = current.merge(baseline, on='cluster', how='inner')

    now_str    = datetime.now(timezone.utc).isoformat()
    anomalies  = []

    # Zone metadata
    cs_meta = pd.read_csv(CS_PATH, usecols=['cluster', 'zone_name', 'police_station'])\
                .drop_duplicates('cluster')

    for _, row in merged.iterrows():
        current_rate  = row['daily_rate']
        baseline_mean = row['baseline_mean']
        baseline_std  = max(row['baseline_std'], 0.5)  # floor std to avoid inflation
        z = (current_rate - baseline_mean) / baseline_std

        if abs(z) < Z_MODERATE:
            continue  # within normal range

        anomaly_type = 'spike' if z > 0 else 'drop'
        if   abs(z) >= Z_EXTREME:  severity = 'extreme'
        elif abs(z) >= Z_SEVERE:   severity = 'severe'
        else:                       severity = 'moderate'

        zone_name = row.get('zone_name') or 'Unknown'
        police_st = row.get('police_station') or 'Unknown'

        conn.execute("""
            INSERT INTO cluster_anomalies
              (cluster, zone_name, police_station, current_rate, baseline_rate,
               z_score, anomaly_type, severity, detected_at, acknowledged)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (int(row['cluster']), zone_name, police_st,
              float(current_rate), float(baseline_mean),
              round(float(z), 2), anomaly_type, severity, now_str))

        icon = '🔴' if severity == 'extreme' else ('🟠' if severity == 'severe' else '🟡')
        direction = '↑' if anomaly_type == 'spike' else '↓'
        print(f"  [anomaly] {icon} {zone_name[:35]:35} {direction} z={z:.1f} | "
              f"now={current_rate:.1f}/day baseline={baseline_mean:.1f}/day [{severity}]")
        anomalies.append(row['cluster'])

    conn.commit()

    if not anomalies:
        print(f"  [anomaly] All {len(merged)} clusters within normal range (|z| < {Z_MODERATE})")

    if close_after: conn.close()
    return pd.DataFrame({'cluster': anomalies})


if __name__ == '__main__':
    run_anomaly_engine()
