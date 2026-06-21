"""
04_score_engine.py — Rolling 30-day window cluster scorer
Usage: python pipeline/04_score_engine.py [--window_days N]

Queries violations ingested in the last N days (wall clock),
recomputes all 7 v4 score components, writes to cluster_scores table.
Returns the scored dataframe (also used by alert engine).
"""
import sqlite3, pandas as pd, numpy as np, os, argparse
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db', 'gridlock.db')

# V4 score weights — tuned based on operational priority:
# c_impact (total severity), c_lanes (traffic blockage) are the clearest proxy
# for actual traffic disruption. c_consistency (active_days) + c_persistence (active_weeks)
# determine chronic vs. one-off offenders. c_peak distinguishes rush-hour clusters.
# c_official and c_compound carry very little signal (only 4 cops + compound events)
# and have been redistributed to impact/lanes.
V4_WEIGHTS = {
    'c_impact':      0.28,  # was 0.20 — severity of each violation is the clearest proxy
    'c_lanes':       0.28,  # was 0.25 — lanes blocked = direct traffic disruption
    'c_peak':        0.18,  # was 0.15 — rush-hour offenders need priority enforcement
    'c_consistency': 0.20,  # was 0.25 — chronic zones (slightly down to give room to severity)
    'c_persistence': 0.05,  # was 0.10 — weeks-active matters but less than daily rate
    'c_official':    0.01,  # was 0.04 — very sparse signal (~4 approved violations)
    'c_compound':    0.00,  # was 0.01 — literally 0 variance; removed from scoring
}

parser = argparse.ArgumentParser()
parser.add_argument('--window_days', type=int, default=30)
args = parser.parse_args()


def mm(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / (rng if rng > 0 else 1)


def risk_tier(score: float) -> str:
    if score >= 75: return 'CRITICAL'
    if score >= 50: return 'HIGH'
    if score >= 25: return 'MEDIUM'
    return 'LOW'


def shift_label(peak_pct: float) -> str:
    # This dataset: 56.5% of violations happen 10pm-4am; is_peak = 0-6am + 7pm-11pm
    if peak_pct > 0.75: return 'Late night / early morning (7pm–6am)'
    if peak_pct > 0.40: return 'Evening + night (7pm–midnight)'
    return 'Daytime (6am–7pm)'


def run_score_engine(window_days: int = 30, conn=None) -> pd.DataFrame:
    close_after = conn is None
    if conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")

    now_str = datetime.now(timezone.utc).isoformat()

    df = pd.read_sql(f"""
        SELECT cluster, hour, is_peak, lanes_blocked, vtype_severity,
               compound_score, impact, validation_weight, scita_weight,
               created_datetime, ingested_at, police_station, junction_name
        FROM violations
        WHERE cluster != -1
          AND ingested_at >= datetime('now', '-{window_days} days')
    """, conn)

    if df.empty:
        print("  [score_engine] No violations in rolling window — skipping")
        if close_after: conn.close()
        return pd.DataFrame()

    # Parse dates for week/day calculations
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], utc=True, errors='coerce')
    df['date']  = df['created_datetime'].dt.date
    df['week']  = df['created_datetime'].dt.isocalendar().week

    # Load zone metadata from cluster_stats for names
    cs_path = os.path.join(os.path.dirname(__file__), '..', 'cluster_stats.csv')
    cs_meta = pd.read_csv(cs_path, usecols=['cluster', 'zone_name', 'lat', 'lon']
                          ).drop_duplicates('cluster')

    # Aggregate per cluster
    agg = df.groupby('cluster').agg(
        violation_count   = ('impact',            'count'),
        total_impact      = ('impact',            'sum'),
        avg_lanes_blocked = ('lanes_blocked',     'mean'),
        peak_pct          = ('is_peak',           'mean'),
        active_days       = ('date',              'nunique'),
        active_weeks      = ('week',              'nunique'),
        approved_pct      = ('validation_weight', lambda x: (x == 1.5).mean()),
        scita_pct         = ('scita_weight',      lambda x: (x == 1.2).mean()),
        compound_pct      = ('compound_score',    lambda x: (x > 1.0).mean()),
        police_station    = ('police_station',    lambda x: x.value_counts().index[0]),
    ).reset_index()

    # Score components (normalised across clusters in THIS window)
    agg['c_impact']      = mm(agg['total_impact'])
    agg['c_lanes']       = mm(agg['avg_lanes_blocked'])
    agg['c_peak']        = mm(agg['peak_pct'])
    agg['c_consistency'] = mm(agg['active_days'])
    agg['c_persistence'] = mm(agg['active_weeks'])
    agg['c_official']    = mm(agg['approved_pct'] * 0.6 + agg['scita_pct'] * 0.4)
    agg['c_compound']    = mm(agg['compound_pct'])

    raw_score = sum(agg[k] * w for k, w in V4_WEIGHTS.items())
    agg['congestion_score'] = (mm(raw_score) * 100).round(1)
    agg['risk_tier']        = agg['congestion_score'].apply(risk_tier)
    agg['enforcement_shift'] = agg['peak_pct'].apply(shift_label)
    agg['daily_rate']       = (agg['violation_count'] / agg['active_days']).round(1)
    agg['blockage_pct']     = ((agg['avg_lanes_blocked'] / 4) * 100).round(1)
    agg['recalculated_at']  = now_str

    # Merge zone names from cluster_stats
    agg = agg.merge(cs_meta, on='cluster', how='left')
    agg['zone_name'] = agg['zone_name'].fillna(agg['police_station'] + ' area')

    # Write to DB
    out_cols = [
        'cluster', 'recalculated_at', 'violation_count', 'total_impact',
        'avg_lanes_blocked', 'peak_pct', 'active_days', 'active_weeks',
        'approved_pct', 'scita_pct', 'compound_pct',
        'congestion_score', 'risk_tier', 'zone_name', 'police_station',
        'daily_rate', 'blockage_pct', 'enforcement_shift',
    ]
    agg[out_cols].to_sql('cluster_scores', conn, if_exists='append', index=False)
    
    # Clean up old scores to prevent DB bloat (keep last 6 hours of history for charts)
    conn.execute("DELETE FROM cluster_scores WHERE recalculated_at < datetime('now', '-6 hours')")
    conn.commit()

    top5 = agg.nlargest(5, 'congestion_score')[
        ['zone_name', 'congestion_score', 'risk_tier', 'violation_count']
    ]
    print(f"  [score_engine] Scored {len(agg)} clusters | top 5:")
    print(top5.to_string(index=False))

    if close_after:
        conn.close()
    return agg


if __name__ == '__main__':
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    run_score_engine(window_days=args.window_days, conn=conn)
    conn.close()
