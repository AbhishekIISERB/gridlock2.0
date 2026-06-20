"""
05_alert_engine.py — Tier crossing detector and alert writer
Usage: python pipeline/05_alert_engine.py

Compares latest vs previous cluster_scores, writes tier crossings to alerts table.
"""
import sqlite3, pandas as pd, os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db', 'gridlock.db')

TIER_ORDER = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}

SEVERITY_LABEL = {
    ('LOW',      'MEDIUM'):   'info',
    ('MEDIUM',   'HIGH'):     'warning',
    ('HIGH',     'CRITICAL'): 'urgent',
    ('CRITICAL', 'HIGH'):     'resolved',
    ('HIGH',     'MEDIUM'):   'resolved',
    ('MEDIUM',   'LOW'):      'resolved',
    ('CRITICAL', 'MEDIUM'):   'resolved',
    ('CRITICAL', 'LOW'):      'resolved',
}


def run_alert_engine(conn=None) -> list:
    close_after = conn is None
    if conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")

    # Get the two most recent recalculated_at timestamps
    times = pd.read_sql("""
        SELECT DISTINCT recalculated_at FROM cluster_scores
        ORDER BY recalculated_at DESC LIMIT 2
    """, conn)

    if len(times) < 2:
        print("  [alert_engine] Not enough score history yet — need ≥2 ticks")
        if close_after: conn.close()
        return []

    latest_t  = times.iloc[0]['recalculated_at']
    prev_t    = times.iloc[1]['recalculated_at']

    latest = pd.read_sql(
        "SELECT * FROM cluster_scores WHERE recalculated_at = ?", conn, params=(latest_t,))
    prev   = pd.read_sql(
        "SELECT * FROM cluster_scores WHERE recalculated_at = ?", conn, params=(prev_t,))

    merged = latest.merge(
        prev[['cluster', 'risk_tier', 'congestion_score']],
        on='cluster', suffixes=('_new', '_prev'), how='left'
    )
    # Fill clusters new to window (no previous score)
    merged['risk_tier_prev'] = merged['risk_tier_prev'].fillna('LOW')

    new_alerts = []
    now_str = datetime.now(timezone.utc).isoformat()

    for _, row in merged.iterrows():
        prev_tier = row['risk_tier_prev']
        new_tier  = row['risk_tier_new']

        if prev_tier == new_tier:
            continue  # no change

        # Also alert on daily_rate spike >25%
        rate_spike = False
        if 'daily_rate_prev' in row and pd.notna(row.get('daily_rate_prev')):
            if row['daily_rate_prev'] > 0:
                rate_spike = (row['daily_rate'] - row['daily_rate_prev']) / row['daily_rate_prev'] > 0.25

        conn.execute("""
            INSERT INTO alerts
              (cluster, zone_name, police_station, previous_tier, new_tier,
               congestion_score, triggered_at, acknowledged)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        """, (int(row['cluster']), row['zone_name'], row['police_station'],
              prev_tier, new_tier, float(row['congestion_score']), now_str))

        severity = SEVERITY_LABEL.get((prev_tier, new_tier), 'info')
        label    = f"{'🚨' if severity=='urgent' else '⚠️' if severity=='warning' else '✅' if severity=='resolved' else 'ℹ️'}"
        print(f"  [alert] {label} {row['zone_name']:35} {prev_tier} → {new_tier} "
              f"(score: {row['congestion_score']:.1f})")
        new_alerts.append(row['zone_name'])

    # Detect new top-10 entrants (not in previous top-10)
    prev_top10   = set(prev.nlargest(10, 'congestion_score')['cluster'])
    latest_top10 = set(latest.nlargest(10, 'congestion_score')['cluster'])
    new_entrants = latest_top10 - prev_top10
    for cluster_id in new_entrants:
        zone = latest[latest['cluster'] == cluster_id].iloc[0]
        # Only alert if not already alerted via tier change
        if zone['zone_name'] not in new_alerts:
            conn.execute("""
                INSERT INTO alerts
                  (cluster, zone_name, police_station, previous_tier, new_tier,
                   congestion_score, triggered_at, acknowledged)
                VALUES (?, ?, ?, 'NONE', ?, ?, ?, 0)
            """, (int(cluster_id), zone['zone_name'], zone['police_station'],
                  zone['risk_tier'], float(zone['congestion_score']), now_str))
            print(f"  [alert] 🆕 New top-10 hotspot: {zone['zone_name']} (score: {zone['congestion_score']:.1f})")

    conn.commit()
    if close_after:
        conn.close()

    if not new_alerts:
        print("  [alert_engine] No tier changes detected this tick")

    return new_alerts


if __name__ == '__main__':
    run_alert_engine()
