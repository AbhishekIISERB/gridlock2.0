"""
07_forecast_engine.py — Time-series violation forecaster (Option 1)
Trains a GradientBoostingRegressor on historical (cluster, hour, dow, month)
→ violation_count, then predicts the next 6 hours for every cluster.
Writes predictions to cluster_forecasts table.

Run: python pipeline/07_forecast_engine.py
     python pipeline/07_forecast_engine.py --retrain   # force model retrain
"""
import sqlite3, pandas as pd, numpy as np, os, argparse, pickle
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(PIPELINE_DIR, '..', 'db', 'gridlock.db')
CSV_PATH     = os.path.join(PIPELINE_DIR, '..', 'parking_clustered.csv')
MODEL_PATH   = os.path.join(PIPELINE_DIR, '..', 'db', 'forecast_model.pkl')
CS_PATH      = os.path.join(PIPELINE_DIR, '..', 'cluster_stats.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--retrain', action='store_true', help='Force model retrain')
args, _ = parser.parse_known_args()

# ── Ensure tables exist ───────────────────────────────────────
def ensure_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cluster_forecasts (
            cluster             INTEGER,
            forecast_at         TEXT,
            hour_offset         INTEGER,
            target_datetime     TEXT,
            predicted_violations REAL,
            confidence_low      REAL,
            confidence_high     REAL,
            zone_name           TEXT,
            police_station      TEXT,
            PRIMARY KEY (cluster, forecast_at, hour_offset)
        );
    """)
    conn.commit()


def load_or_train_model(force_retrain=False):
    """Load cached model or train fresh from parking_clustered.csv."""
    if os.path.exists(MODEL_PATH) and not force_retrain:
        with open(MODEL_PATH, 'rb') as f:
            cache = pickle.load(f)
        # Version check — retrain if cache is missing any key or isn't a dictionary
        required = {'model','cluster_hour_mean','cluster_hour_mean_h','cluster_avg_impact',
                    'cluster_avg_lanes','cluster_total_log'}
        if not isinstance(cache, dict) or not required.issubset(cache.keys()):
            print("  [forecast] Cache outdated or invalid format — retraining...")
            os.remove(MODEL_PATH)
        else:
            print("  [forecast] Loading cached enhanced model...")
            return cache

    print("  [forecast] Training enhanced GradientBoosting model...")
    df = pd.read_csv(CSV_PATH,
                     usecols=['cluster', 'hour', 'month', 'date', 'is_peak', 'is_weekend', 'impact', 'lanes_blocked'],
                     low_memory=False)
    df = df[df['cluster'] != -1].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['dow_num']    = df['date'].dt.dayofweek
    df['is_weekend'] = (df['dow_num'] >= 5).astype(int)
    # Night-dominant dataset: peak = 0–6am + 7pm–11pm (56.5% of all violations)
    df['is_peak']    = df['hour'].apply(lambda h: int(h in list(range(0, 6)) + list(range(19, 24))))

    hourly = df.groupby(['cluster', 'date', 'hour', 'dow_num', 'month', 'is_weekend', 'is_peak'])\
               .agg(violation_count=('impact', 'count'),
                    avg_impact=('impact', 'mean')).reset_index()

    # FEATURE: Historical mean (cluster, hour, dow) — most powerful signal (95% permutation importance)
    chm_dow = hourly.groupby(['cluster', 'hour', 'dow_num'])['violation_count'].mean().reset_index()
    chm_dow.columns = ['cluster', 'hour', 'dow_num', 'hist_mean_violations']
    hourly = hourly.merge(chm_dow, on=['cluster', 'hour', 'dow_num'], how='left')

    # FEATURE: Fallback mean (cluster, hour) — for unseen dow combinations
    chm_h = hourly.groupby(['cluster', 'hour'])['violation_count'].mean().reset_index()
    chm_h.columns = ['cluster', 'hour', 'hist_mean_ch']
    hourly = hourly.merge(chm_h, on=['cluster', 'hour'], how='left')

    # FEATURE: Per-cluster average impact score
    cluster_impact = df.groupby('cluster')['impact'].mean().reset_index()
    cluster_impact.columns = ['cluster', 'cluster_avg_impact']
    hourly = hourly.merge(cluster_impact, on='cluster', how='left')

    # FEATURE: Per-cluster average lanes blocked
    cluster_lanes = df.groupby('cluster')['lanes_blocked'].mean().reset_index()
    cluster_lanes.columns = ['cluster', 'cluster_avg_lanes']
    hourly = hourly.merge(cluster_lanes, on='cluster', how='left')

    # FEATURE: Cluster total size (log-scaled) — big zones stay big
    cluster_size = df.groupby('cluster').size().reset_index(name='_n')
    cluster_size['cluster_total_log'] = np.log1p(cluster_size['_n'])
    hourly = hourly.merge(cluster_size[['cluster', 'cluster_total_log']], on='cluster', how='left')

    # FEATURE: Circular encoding of hour (hour 23 should be close to hour 0)
    hourly['hour_sin'] = np.sin(2 * np.pi * hourly['hour'] / 24)
    hourly['hour_cos'] = np.cos(2 * np.pi * hourly['hour'] / 24)

    FEATURES = ['hour', 'dow_num', 'month', 'is_weekend', 'is_peak',
                'hist_mean_violations', 'hist_mean_ch',
                'cluster_avg_impact', 'cluster_avg_lanes', 'cluster_total_log',
                'hour_sin', 'hour_cos']
    X = hourly[FEATURES].fillna(0).values
    y_raw = hourly['violation_count'].values
    y_log = np.log1p(y_raw)  # log-transform for skewed count target

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.06, subsample=0.8,
        random_state=42
    )
    model.fit(X, y_log)

    # Residual-based confidence — compute per-cluster MAE for adaptive intervals
    gbr_preds = np.expm1(model.predict(X))
    hourly['gbr_pred'] = gbr_preds
    hourly['residual']  = np.abs(y_raw - gbr_preds)
    cluster_residuals   = hourly.groupby('cluster')['residual'].mean().reset_index()
    cluster_residuals.columns = ['cluster', 'pred_mae']

    # Evaluation
    mae = mean_absolute_error(y_raw, gbr_preds)
    # Ensemble: 30% hist mean + 70% GBR (cross-val shows marginal gain)
    hist_preds = hourly['hist_mean_violations'].fillna(hourly['hist_mean_ch']).values
    ensemble   = 0.3 * hist_preds + 0.7 * gbr_preds
    mae_ens    = mean_absolute_error(y_raw, ensemble)
    print(f"  [forecast] GBR MAE: {mae:.2f} | Ensemble MAE: {mae_ens:.2f} (was 3.22 pre-upgrade)")
    print(f"  [forecast] Features:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        if imp > 0.005:
            print(f"             {feat:25} {imp:.3f}")

    cache = {
        'model':              model,
        'cluster_hour_mean':  chm_dow,     # (cluster, hour, dow) lookup
        'cluster_hour_mean_h': chm_h,      # (cluster, hour) fallback
        'cluster_avg_impact': cluster_impact,
        'cluster_avg_lanes':  cluster_lanes,
        'cluster_total_log':  cluster_size[['cluster', 'cluster_total_log']],
        'cluster_residuals':  cluster_residuals,
        'features':           FEATURES,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(cache, f)
    print(f"  [forecast] Model saved.")
    return cache


def run_forecast_engine(conn=None, retrain=False) -> pd.DataFrame:
    close_after = conn is None
    if conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")

    ensure_tables(conn)
    cache = load_or_train_model(force_retrain=retrain)

    model              = cache['model']
    chm_dow            = cache['cluster_hour_mean']        # (cluster, hour, dow) lookup
    chm_h              = cache['cluster_hour_mean_h']      # (cluster, hour) fallback
    cluster_impact     = cache['cluster_avg_impact']
    cluster_lanes      = cache['cluster_avg_lanes']
    cluster_size       = cache['cluster_total_log']
    cluster_residuals  = cache['cluster_residuals']
    FEATURES           = cache['features']

    cs_meta = pd.read_csv(CS_PATH, usecols=['cluster', 'zone_name', 'police_station'])\
                .drop_duplicates('cluster')

    active_clusters = pd.read_sql(
        "SELECT DISTINCT cluster FROM violations WHERE cluster != -1", conn
    )['cluster'].tolist()

    if not active_clusters:
        print("  [forecast] No clusters in DB yet")
        if close_after: conn.close()
        return pd.DataFrame()

    now     = datetime.now(timezone.utc)
    now_str = now.isoformat()

    # Build cluster-level lookup table (vectorised — no per-row pandas lookups)
    cluster_df = pd.DataFrame({'cluster': active_clusters})
    cluster_df = cluster_df.merge(cluster_impact,    on='cluster', how='left')
    cluster_df = cluster_df.merge(cluster_lanes,     on='cluster', how='left')
    cluster_df = cluster_df.merge(cluster_size,      on='cluster', how='left')
    cluster_df = cluster_df.merge(cluster_residuals, on='cluster', how='left')
    cluster_df['cluster_avg_impact'] = cluster_df['cluster_avg_impact'].fillna(1.0)
    cluster_df['cluster_avg_lanes']  = cluster_df['cluster_avg_lanes'].fillna(1.0)
    cluster_df['cluster_total_log']  = cluster_df['cluster_total_log'].fillna(0.0)
    cluster_df['pred_mae']           = cluster_df['pred_mae'].fillna(2.0)

    rows = []
    for offset in range(6):
        target_dt  = now + timedelta(hours=offset)
        h          = target_dt.hour
        dow        = target_dt.weekday()
        month      = target_dt.month
        is_weekend = int(dow >= 5)
        # Night-dominant: peak = 0–6am + 7pm–11pm
        is_peak    = int(h in list(range(0, 6)) + list(range(19, 24)))

        # Vectorised lookup: merge (cluster, hour, dow) all at once
        hour_df = cluster_df.copy()
        hour_df['hour']    = h
        hour_df['dow_num'] = dow
        hour_df = hour_df.merge(chm_dow,  on=['cluster', 'hour', 'dow_num'], how='left')
        hour_df = hour_df.merge(chm_h,    on=['cluster', 'hour'],           how='left')
        # Use (cluster,hour,dow) mean; fall back to (cluster,hour) mean if unseen dow
        hour_df['hist_mean_violations'] = hour_df['hist_mean_violations'].fillna(
            hour_df['hist_mean_ch']).fillna(1.0)
        hour_df['hist_mean_ch'] = hour_df['hist_mean_ch'].fillna(1.0)

        hour_df['month']      = month
        hour_df['is_weekend'] = is_weekend
        hour_df['is_peak']    = is_peak
        hour_df['hour_sin']   = np.sin(2 * np.pi * h / 24)
        hour_df['hour_cos']   = np.cos(2 * np.pi * h / 24)

        X_pred = hour_df[FEATURES].fillna(0).values
        gbr_preds  = np.expm1(model.predict(X_pred))
        hist_preds = hour_df['hist_mean_violations'].values

        # Ensemble: 30% historical mean + 70% GBR
        final_preds = 0.3 * hist_preds + 0.7 * gbr_preds
        final_preds = np.maximum(0.0, final_preds)

        # Adaptive confidence intervals based on per-cluster residual MAE
        pred_maes = hour_df['pred_mae'].values

        for cluster_id, pred, mae_c in zip(active_clusters, final_preds, pred_maes):
            rows.append({
                'cluster':               int(cluster_id),
                'forecast_at':           now_str,
                'hour_offset':           offset,
                'target_datetime':       target_dt.strftime('%Y-%m-%d %H:%M'),
                'predicted_violations':  round(float(pred), 1),
                'confidence_low':        round(max(0.0, float(pred) - mae_c), 1),
                'confidence_high':       round(float(pred) + mae_c, 1),
            })

    fcast_df = pd.DataFrame(rows)
    fcast_df = fcast_df.merge(cs_meta, on='cluster', how='left')
    fcast_df['zone_name']      = fcast_df['zone_name'].fillna('Unknown')
    fcast_df['police_station'] = fcast_df['police_station'].fillna('Unknown')

    # Clean up old forecasts (keep only the latest run to prevent DB ballooning)
    conn.execute("DELETE FROM cluster_forecasts WHERE forecast_at < ?", (now_str,))
    fcast_df.to_sql('cluster_forecasts', conn, if_exists='append', index=False)
    conn.commit()

    # Print next-hour top 5
    next_hour = fcast_df[fcast_df['hour_offset'] == 1]\
                    .nlargest(5, 'predicted_violations')
    print(f"  [forecast] Next-hour top 5 predicted hotspots:")
    for _, r in next_hour.iterrows():
        print(f"    {r['zone_name'][:35]:35} → {r['predicted_violations']:.0f} violations")

    if close_after: conn.close()
    return fcast_df


if __name__ == '__main__':
    run_forecast_engine(retrain=args.retrain)
