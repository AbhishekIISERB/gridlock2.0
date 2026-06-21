"""
api.py — FastAPI REST layer for the Gridlock AI frontend
Run: uvicorn api:app --host 0.0.0.0 --port 8502 --reload
"""
import sqlite3, os
from datetime import datetime
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd

DB_PATH   = os.path.join(os.path.dirname(__file__), 'db', 'gridlock.db')
CS_PATH   = os.path.join(os.path.dirname(__file__), 'cluster_stats.csv')
FRONT_DIR = os.path.join(os.path.dirname(__file__), 'frontend')

app = FastAPI(title="Gridlock AI API", version="1.0")

# Allow cross-origin from the frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


# ── Health ────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


# ── Ingestion state ────────────────────────────────────────────────
@app.get("/api/state")
def state():
    conn = get_conn()
    row = conn.execute("SELECT * FROM ingestion_state WHERE id=1").fetchone()
    conn.close()
    if not row:
        return {}
    return dict(row)


# ── KPIs ──────────────────────────────────────────────────────────
@app.get("/api/kpis")
def kpis():
    conn = get_conn()
    try:
        zones = pd.read_sql("""
            SELECT risk_tier, congestion_score FROM cluster_scores
            WHERE recalculated_at = (SELECT MAX(recalculated_at) FROM cluster_scores)
        """, conn)
        alerts_n = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE acknowledged=0"
        ).fetchone()[0]
        anomalies_n = conn.execute(
            "SELECT COUNT(*) FROM cluster_anomalies WHERE acknowledged=0"
        ).fetchone()[0] if _table_exists(conn, 'cluster_anomalies') else 0

        state_row = conn.execute(
            "SELECT total_rows_ingested, last_ingested_datetime FROM ingestion_state WHERE id=1"
        ).fetchone()

        return {
            "active_clusters": int(len(zones)),
            "critical_zones":  int((zones['risk_tier'] == 'CRITICAL').sum()),
            "high_zones":      int((zones['risk_tier'] == 'HIGH').sum()),
            "medium_zones":    int((zones['risk_tier'] == 'MEDIUM').sum()),
            "unacked_alerts":  int(alerts_n),
            "anomalies":       int(anomalies_n),
            "total_ingested":  int(state_row[0]) if state_row else 0,
            "last_ingested":   str(state_row[1])[:19] if state_row else "",
            "refresh_ts":      datetime.utcnow().isoformat(),
        }
    finally:
        conn.close()


def _table_exists(conn, name):
    return conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{name}'"
    ).fetchone() is not None


# ── Live zones ─────────────────────────────────────────────────────
@app.get("/api/zones")
def zones():
    conn = get_conn()
    try:
        cs = pd.read_csv(CS_PATH, usecols=['cluster','lat','lon']).drop_duplicates('cluster')
        df = pd.read_sql("""
            SELECT cluster, zone_name, congestion_score, risk_tier,
                   violation_count, daily_rate, blockage_pct,
                   enforcement_shift, police_station
            FROM cluster_scores
            WHERE recalculated_at = (
                SELECT MAX(recalculated_at) FROM cluster_scores cs2
                WHERE cs2.cluster = cluster_scores.cluster
            )
            ORDER BY congestion_score DESC
        """, conn)
        df = df.merge(cs, on='cluster', how='left')
        return df.dropna(subset=['lat','lon']).to_dict(orient='records')
    finally:
        conn.close()


# ── Heatmap violations ─────────────────────────────────────────────
@app.get("/api/heatmap")
def heatmap(hours: int = 24, mode: str = "volume"):
    conn = get_conn()
    try:
        df = pd.read_sql(f"""
            SELECT latitude, longitude, impact FROM violations
            WHERE ingested_at >= (
                SELECT datetime(MAX(ingested_at), '-{hours} hours') FROM violations
            )
            ORDER BY ingested_at DESC LIMIT 80000
        """, conn)
        if mode == "impact":
            df = df[df['impact'] >= 4.0].copy()
            df['weight'] = df['impact'] ** 2
        else:
            df['weight'] = 1.0
        df = df.dropna(subset=['latitude', 'longitude'])
        return {"points": df[['latitude','longitude','weight']].values.tolist()}
    finally:
        conn.close()


# ── Alerts ────────────────────────────────────────────────────────
@app.get("/api/alerts")
def alerts(limit: int = 20):
    conn = get_conn()
    rows = pd.read_sql("""
        SELECT id, zone_name, police_station, previous_tier, new_tier,
               congestion_score, triggered_at, acknowledged
        FROM alerts ORDER BY triggered_at DESC LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return rows.to_dict(orient='records')


@app.post("/api/alerts/{alert_id}/ack")
def ack_alert(alert_id: int):
    conn = get_conn()
    conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


# ── Anomalies ─────────────────────────────────────────────────────
@app.get("/api/anomalies")
def anomalies():
    conn = get_conn()
    try:
        rows = pd.read_sql("""
            SELECT id, cluster, zone_name, police_station, current_rate,
                   baseline_rate, z_score, anomaly_type, severity, detected_at
            FROM cluster_anomalies
            WHERE acknowledged=0
            ORDER BY z_score DESC LIMIT 15
        """, conn)
        conn.close()
        return rows.to_dict(orient='records')
    except Exception:
        conn.close()
        return []


@app.post("/api/anomalies/{anomaly_id}/ack")
def ack_anomaly(anomaly_id: int):
    conn = get_conn()
    conn.execute("UPDATE cluster_anomalies SET acknowledged=1 WHERE id=?", (anomaly_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


# ── Forecasts ─────────────────────────────────────────────────────
@app.get("/api/forecasts")
def forecasts():
    conn = get_conn()
    try:
        df = pd.read_sql("""
            SELECT f.cluster, f.zone_name, f.police_station, f.hour_offset,
                   f.target_datetime, f.predicted_violations,
                   f.confidence_low, f.confidence_high,
                   v.validation_status, v.llm_rationale
            FROM cluster_forecasts f
            LEFT JOIN forecast_validations v 
                   ON f.cluster = v.cluster 
                  AND f.forecast_at = v.forecast_at 
                  AND f.hour_offset = v.hour_offset
            WHERE f.forecast_at = (SELECT MAX(forecast_at) FROM cluster_forecasts)
              AND f.hour_offset BETWEEN 1 AND 3
            ORDER BY f.hour_offset ASC, f.predicted_violations DESC
        """, conn)
        conn.close()
        
        # Replace NaN values with None/empty strings so FastAPI JSON encoder doesn't crash
        df['validation_status'] = df['validation_status'].fillna('')
        df['llm_rationale'] = df['llm_rationale'].fillna('')
        
        return df.to_dict(orient='records')
    except Exception:
        conn.close()
        return []


# ── Deployment plan ────────────────────────────────────────────────
@app.get("/api/deployment")
def deployment():
    conn = get_conn()
    try:
        df = pd.read_sql("""
            SELECT police_station, zone_name, hour_offset,
                   SUM(predicted_violations) as total_pred
            FROM cluster_forecasts
            WHERE forecast_at = (SELECT MAX(forecast_at) FROM cluster_forecasts)
              AND hour_offset BETWEEN 1 AND 3
              AND police_station != 'Unknown'
            GROUP BY police_station, zone_name, hour_offset
            ORDER BY total_pred DESC
        """, conn)
        conn.close()

        # Group by station
        result = []
        for station, grp in df.groupby('police_station'):
            total = grp['total_pred'].sum()
            patrols = max(1, int(total / 15))
            top_zones = grp.groupby('zone_name')['total_pred'].sum().nlargest(3).reset_index()
            result.append({
                "police_station": station,
                "total_predicted": round(float(total), 0),
                "recommended_patrols": patrols,
                "top_zones": top_zones.to_dict(orient='records'),
            })
        result.sort(key=lambda x: -x['total_predicted'])
        return result[:6]
    except Exception:
        conn.close()
        return []


# ── Score history for charts ───────────────────────────────────────
@app.get("/api/score-history")
def score_history(top_n: int = 5):
    conn = get_conn()
    try:
        top_clusters = pd.read_sql("""
            SELECT cluster, zone_name, congestion_score FROM cluster_scores
            WHERE recalculated_at = (SELECT MAX(recalculated_at) FROM cluster_scores)
            ORDER BY congestion_score DESC LIMIT ?
        """, conn, params=(top_n,))

        if top_clusters.empty:
            conn.close()
            return []

        cids = tuple(top_clusters['cluster'].tolist())
        ph   = ','.join(['?' for _ in cids])
        hist = pd.read_sql(f"""
            SELECT cluster, zone_name, congestion_score, recalculated_at
            FROM cluster_scores
            WHERE cluster IN ({ph})
            ORDER BY recalculated_at ASC
        """, conn, params=cids)
        conn.close()
        return hist.to_dict(orient='records')
    except Exception:
        conn.close()
        return []


# ── Serve the frontend ─────────────────────────────────────────────
if os.path.isdir(FRONT_DIR):
    app.mount("/", StaticFiles(directory=FRONT_DIR, html=True), name="frontend")
