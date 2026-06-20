"""
streamlit_live.py — Live Gridlock AI Dashboard (polls SQLite every 30s)
Run: streamlit run streamlit_live.py
"""
import streamlit as st
import pandas as pd
import sqlite3
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime, timezone
import os, time

st.set_page_config(
    page_title="Gridlock AI — Live",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = os.path.join(os.path.dirname(__file__), 'db', 'gridlock.db')

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; color: #e2e8f0; }
  [data-testid="stSidebar"] { background: #161b22; }
  .metric-card { background:#1e2530; border-radius:10px; padding:16px;
                 border:1px solid rgba(255,255,255,0.08); margin-bottom:8px; }
  .alert-urgent  { background:#3d1515; border-left:4px solid #ef4444;
                   padding:10px; border-radius:6px; margin:6px 0; }
  .alert-warning { background:#2d2215; border-left:4px solid #f97316;
                   padding:10px; border-radius:6px; margin:6px 0; }
  .alert-info    { background:#152030; border-left:4px solid #3b82f6;
                   padding:10px; border-radius:6px; margin:6px 0; }
  .alert-resolved{ background:#152520; border-left:4px solid #22c55e;
                   padding:10px; border-radius:6px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ── DB helpers ─────────────────────────────────────────────────
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)

def db_exists():
    return os.path.exists(DB_PATH)

@st.cache_data(ttl=30)
def load_live_zones():
    conn = get_conn()
    try:
        df = pd.read_sql("""
            SELECT * FROM cluster_scores
            WHERE recalculated_at IN (
                SELECT MAX(recalculated_at) FROM cluster_scores cs2
                WHERE cs2.cluster = cluster_scores.cluster
            )
            ORDER BY congestion_score DESC
        """, conn)
    finally:
        conn.close()
    return df

@st.cache_data(ttl=30)
def load_recent_violations(hours=24):
    conn = get_conn()
    try:
        df = pd.read_sql(f"""
            SELECT * FROM violations
            WHERE ingested_at >= datetime('now', '-{hours} hours')
        """, conn)
    finally:
        conn.close()
    return df

@st.cache_data(ttl=10)
def load_unack_alerts():
    conn = get_conn()
    try:
        df = pd.read_sql("""
            SELECT * FROM alerts WHERE acknowledged = 0
            ORDER BY triggered_at DESC LIMIT 30
        """, conn)
    finally:
        conn.close()
    return df

@st.cache_data(ttl=30)
def load_ingestion_state():
    conn = get_conn()
    try:
        df = pd.read_sql("SELECT * FROM ingestion_state", conn)
    finally:
        conn.close()
    return df

@st.cache_data(ttl=30)
def load_score_history(top_cluster_ids):
    if not top_cluster_ids:
        return pd.DataFrame()
    ids = ','.join(str(c) for c in top_cluster_ids)
    conn = get_conn()
    try:
        df = pd.read_sql(f"""
            SELECT cluster, zone_name, congestion_score, recalculated_at
            FROM cluster_scores WHERE cluster IN ({ids})
            ORDER BY recalculated_at
        """, conn)
    finally:
        conn.close()
    return df

def ack_alert(alert_id):
    conn = get_conn()
    conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()
    st.cache_data.clear()

def score_color(s):
    if s >= 75: return '#ef4444'
    if s >= 50: return '#f97316'
    if s >= 25: return '#eab308'
    return '#22c55e'

# ── Guard: DB not ready ────────────────────────────────────────
if not db_exists():
    st.error("⚠️ Database not found. Run `python pipeline/01_init_db.py` then `02_seed_db.py` first.")
    st.stop()

# ── Sidebar: Alerts ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Gridlock AI Live")
    st.markdown("---")

    alerts = load_unack_alerts()
    if len(alerts) > 0:
        st.error(f"🚨 {len(alerts)} unacknowledged alert{'s' if len(alerts)>1 else ''}")
        for _, a in alerts.iterrows():
            pt = a['previous_tier']
            nt = a['new_tier']
            if nt == 'CRITICAL' and pt != 'CRITICAL':    css = 'alert-urgent'
            elif nt in ('HIGH','MEDIUM') and pt != 'NONE': css = 'alert-warning'
            elif pt in ('CRITICAL','HIGH') and TIER_ORDER.get(nt,0) < TIER_ORDER.get(pt,0):
                css = 'alert-resolved'
            else:
                css = 'alert-info'

            col1, col2 = st.columns([4, 1])
            col1.markdown(f"""
<div class="{css}">
  <b>{a['zone_name']}</b><br>
  <small>{pt} → {nt} | Score: {a['congestion_score']:.1f}</small><br>
  <small style='color:#94a3b8'>{str(a['triggered_at'])[:19]}</small>
</div>""", unsafe_allow_html=True)
            if col2.button("✓", key=f"ack_{a['id']}"):
                ack_alert(int(a['id']))
                st.rerun()
    else:
        st.success("✅ No active alerts")

    st.markdown("---")
    st.markdown("**Filters**")
    hour_filter = st.slider("Hours of violations to show", 1, 168, 24, step=1)
    show_heatmap = st.checkbox("Show heatmap layer", True)
    show_markers = st.checkbox("Show zone markers", True)

    st.markdown("---")
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

TIER_ORDER = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}

# ── Header KPIs ────────────────────────────────────────────────
state_df = load_ingestion_state()
if not state_df.empty:
    state = state_df.iloc[0]
    st.caption(
        f"🔴 **LIVE** &nbsp;|&nbsp; "
        f"**{int(state['total_rows_ingested']):,}** violations ingested &nbsp;|&nbsp; "
        f"Last: `{str(state['last_ingested_datetime'])[:19]}` &nbsp;|&nbsp; "
        f"Dashboard refreshed: `{datetime.now().strftime('%H:%M:%S')}`"
    )

zones = load_live_zones()
violations = load_recent_violations(hours=hour_filter)

# KPI row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Active clusters", len(zones))
col2.metric("CRITICAL zones",  int((zones['risk_tier'] == 'CRITICAL').sum()) if not zones.empty else 0)
col3.metric("HIGH zones",      int((zones['risk_tier'] == 'HIGH').sum()) if not zones.empty else 0)
col4.metric(f"Violations ({hour_filter}h)", f"{len(violations):,}")
col5.metric("Unacked alerts",  len(alerts))

st.markdown("---")

# ── Map ────────────────────────────────────────────────────────
st.subheader("🗺️ Live Enforcement Map")

BLR = [12.9716, 77.5946]
m   = folium.Map(location=BLR, zoom_start=12, tiles='CartoDB dark_matter')

# Heatmap layer
if show_heatmap and not violations.empty:
    heat_cols = ['latitude', 'longitude']
    if 'impact' in violations.columns:
        heat_cols.append('impact')
    heat_data = violations[heat_cols].dropna().values.tolist()
    if heat_data:
        HeatMap(heat_data, name='Live violations (heatmap)',
                min_opacity=0.3, radius=10, blur=14,
                gradient={0.2:'#440154', 0.45:'#31688e',
                          0.7:'#35b779', 1.0:'#fde725'}
                ).add_to(m)

# Zone markers
if show_markers and not zones.empty:
    # Load lat/lon from cluster_stats
    cs_path = os.path.join(os.path.dirname(__file__), 'cluster_stats.csv')
    if os.path.exists(cs_path):
        cs_geo = pd.read_csv(cs_path, usecols=['cluster', 'lat', 'lon'])
        zones  = zones.merge(cs_geo, on='cluster', how='left')

    fg = folium.FeatureGroup(name='Enforcement zones', show=True)
    for _, r in zones.head(20).iterrows():
        if pd.isna(r.get('lat')) or pd.isna(r.get('lon')):
            continue
        col    = score_color(r['congestion_score'])
        radius = 12 + (r['congestion_score'] / 100) * 22
        html = f"""
        <div style='font-family:system-ui;min-width:220px;font-size:13px'>
          <div style='background:{col};color:#fff;padding:6px 10px;
               border-radius:6px 6px 0 0;font-weight:600'>
            {r['zone_name']}
          </div>
          <div style='padding:8px 10px;background:#1e1e2e;color:#e2e8f0;
               border-radius:0 0 6px 6px'>
            <b>Score:</b> {r['congestion_score']}/100 &nbsp;
            <b>Tier:</b> {r['risk_tier']}<br>
            <b>Violations:</b> {int(r['violation_count']):,} &nbsp;
            <b>Daily rate:</b> {r['daily_rate']:.1f}/day<br>
            <b>Blockage:</b> {r['blockage_pct']:.1f}% &nbsp;
            <b>Shift:</b> {r['enforcement_shift']}
          </div>
        </div>"""
        folium.CircleMarker(
            location=[r['lat'], r['lon']],
            radius=radius, color=col,
            fill=True, fill_color=col, fill_opacity=0.75,
            popup=folium.Popup(html, max_width=260),
            tooltip=f"{r['zone_name']} | {r['risk_tier']} | {r['congestion_score']}/100"
        ).add_to(fg)
    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, width='100%', height=500)

st.markdown("---")

# ── Score trend chart ──────────────────────────────────────────
st.subheader("📈 Score Trends — Top 5 Zones")
if not zones.empty:
    top5_clusters = zones.nlargest(5, 'congestion_score')['cluster'].tolist()
    history = load_score_history(top5_clusters)
    if not history.empty:
        history['recalculated_at'] = pd.to_datetime(history['recalculated_at'], errors='coerce')
        pivot = history.pivot_table(
            index='recalculated_at', columns='zone_name',
            values='congestion_score', aggfunc='mean'
        )
        st.line_chart(pivot, use_container_width=True)
    else:
        st.info("Score history will appear after 2+ pipeline ticks.")

# ── Zone table ─────────────────────────────────────────────────
st.subheader("🏆 Current Zone Rankings")
if not zones.empty:
    display_cols = ['zone_name', 'congestion_score', 'risk_tier', 'violation_count',
                    'daily_rate', 'blockage_pct', 'enforcement_shift', 'police_station']
    display_cols = [c for c in display_cols if c in zones.columns]

    def color_tier(val):
        colors = {'CRITICAL':'#ef4444','HIGH':'#f97316','MEDIUM':'#eab308','LOW':'#22c55e'}
        return f'color: {colors.get(val,"#e2e8f0")}'

    styled = zones[display_cols].head(20).style.map(color_tier, subset=['risk_tier'])
    st.dataframe(styled, use_container_width=True, height=450)
else:
    st.info("Waiting for first score calculation...")

# ── Alert history table ────────────────────────────────────────
with st.expander("📋 Full alert log"):
    conn = get_conn()
    all_alerts = pd.read_sql(
        "SELECT * FROM alerts ORDER BY triggered_at DESC LIMIT 50", conn)
    conn.close()
    if not all_alerts.empty:
        st.dataframe(all_alerts, use_container_width=True)
    else:
        st.info("No alerts yet.")

# ── Auto-rerun every 30s ───────────────────────────────────────
time.sleep(0.5)  # small buffer
st.markdown(
    "<small style='color:#475569'>Auto-refreshes every 30 seconds via cache TTL. "
    "Use 'Refresh now' for immediate update.</small>",
    unsafe_allow_html=True
)
