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

# ── Custom CSS (World-Class UI) ────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
  
  /* Global Typography */
  html, body, [class*="css"] {
      font-family: 'Outfit', sans-serif;
  }
  
  /* Background & Sidebar */
  [data-testid="stAppViewContainer"] { 
      background: radial-gradient(circle at 50% 0%, #151b2b 0%, #0a0e17 100%); 
      color: #e2e8f0; 
  }
  [data-testid="stSidebar"] { 
      background: rgba(16, 21, 31, 0.6);
      backdrop-filter: blur(16px);
      border-right: 1px solid rgba(255,255,255,0.05);
  }
  
  /* Gradient Text */
  .gradient-text {
      background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 700;
  }

  /* Alerts */
  .alert-urgent { 
      background: linear-gradient(135deg, rgba(127,29,29,0.8), rgba(69,10,10,0.9)); 
      border-left: 4px solid #ef4444; padding: 12px; border-radius: 8px; margin: 8px 0;
      box-shadow: 0 4px 15px rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.2);
  }
  .alert-warning { 
      background: linear-gradient(135deg, rgba(124,45,18,0.8), rgba(67,20,7,0.9)); 
      border-left: 4px solid #f97316; padding: 12px; border-radius: 8px; margin: 8px 0;
      box-shadow: 0 4px 15px rgba(249,115,22,0.15); border: 1px solid rgba(249,115,22,0.2);
  }
  .alert-info { 
      background: linear-gradient(135deg, rgba(30,58,138,0.8), rgba(17,24,39,0.9)); 
      border-left: 4px solid #3b82f6; padding: 12px; border-radius: 8px; margin: 8px 0;
      box-shadow: 0 4px 15px rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.2);
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #475569; }
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
    """Returns violations from the last N hours of the DB's own timeline.
    Works whether data is freshly seeded (2023 dates) or truly live (today)."""
    conn = get_conn()
    try:
        df = pd.read_sql(f"""
            SELECT * FROM violations
            WHERE ingested_at >= (
                SELECT datetime(MAX(ingested_at), '-{hours} hours')
                FROM violations
            )
        """, conn)
    finally:
        conn.close()
    return df

@st.cache_data(ttl=30)
def load_violations_for_heatmap(hours=24, limit=100000):
    """All violations for heatmap (no cluster filter), filtered by hours slider.
    Time is relative to DB's own max ingested_at so seeded data works too."""
    conn = get_conn()
    try:
        df = pd.read_sql(f"""
            SELECT latitude, longitude, impact FROM violations
            WHERE ingested_at >= (
                SELECT datetime(MAX(ingested_at), '-{hours} hours')
                FROM violations
            )
            ORDER BY ingested_at DESC
            LIMIT {limit}
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

@st.cache_data(ttl=10)
def load_unack_anomalies():
    conn = get_conn()
    try:
        # Fails gracefully if table not created yet
        df = pd.read_sql("""
            SELECT * FROM cluster_anomalies WHERE acknowledged = 0
            ORDER BY detected_at DESC LIMIT 30
        """, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

@st.cache_data(ttl=30)
def load_forecasts():
    conn = get_conn()
    try:
        df = pd.read_sql("""
            SELECT * FROM cluster_forecasts
            WHERE forecast_at = (SELECT MAX(forecast_at) FROM cluster_forecasts)
              AND hour_offset > 0
            ORDER BY hour_offset ASC, predicted_violations DESC
        """, conn)
    except Exception:
        df = pd.DataFrame()
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

def ack_anomaly(anomaly_id):
    conn = get_conn()
    conn.execute("UPDATE cluster_anomalies SET acknowledged=1 WHERE id=?", (anomaly_id,))
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
        st.success("✅ No active tier alerts")

    anoms = load_unack_anomalies()
    if not anoms.empty:
        st.markdown("---")
        st.error(f"⚠️ {len(anoms)} statistical anomal{'ies' if len(anoms)>1 else 'y'}")
        for _, a in anoms.iterrows():
            if a['severity'] == 'extreme': css = 'alert-urgent'
            elif a['severity'] == 'severe': css = 'alert-warning'
            else: css = 'alert-info'
            
            icon = '↑' if a['anomaly_type'] == 'spike' else '↓'
            
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"""
<div class="{css}">
  <b>{a['zone_name']}</b><br>
  <small>{icon} {a['anomaly_type'].title()} ({a['severity']}) | z={a['z_score']:.1f}</small><br>
  <small>Rate: {a['current_rate']:.1f} vs avg {a['baseline_rate']:.1f}</small>
</div>""", unsafe_allow_html=True)
            if col2.button("✓", key=f"ack_anom_{a['id']}"):
                ack_anomaly(int(a['id']))
                st.rerun()

    st.markdown("---")
    st.markdown("**Filters**")
    hour_filter = st.slider("Hours of violations to show", 1, 168, 24, step=1)
    
    heatmap_type = st.radio("Heatmap Weight", ["Violation Volume", "Congestion Impact"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    show_heatmap = st.checkbox("Show heatmap layer", True)
    show_markers = st.checkbox("Show zone markers", True)
    show_pois    = st.checkbox("Show POIs (Commercial/Metro)", True)

    st.markdown("---")
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

TIER_ORDER = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}

# ── Header KPIs ────────────────────────────────────────────────
state_df = load_ingestion_state()
if not state_df.empty:
    state = state_df.iloc[0]
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px;">
        <div>
            <h1 style="margin-bottom: 0px;"><span class="gradient-text">Gridlock AI</span> <span style="font-weight:300; opacity:0.8;">Command Center</span></h1>
        </div>
        <div style="text-align: right; opacity: 0.7; font-size: 14px;">
            <span style="color:#ef4444; font-weight:700;">● LIVE</span> &nbsp;|&nbsp; 
            <b>{int(state['total_rows_ingested']):,}</b> violations &nbsp;|&nbsp; 
            Last: {str(state['last_ingested_datetime'])[:19]} &nbsp;|&nbsp; 
            Refreshed: {datetime.now().strftime('%H:%M:%S')}
        </div>
    </div>
    """, unsafe_allow_html=True)

zones = load_live_zones()
violations = load_recent_violations(hours=hour_filter)

def kpi_card(title, value, color_hex):
    return f"""
    <div style="background: linear-gradient(135deg, rgba(30,41,59,0.5) 0%, rgba(15,23,42,0.8) 100%); 
         border: 1px solid rgba(255,255,255,0.05); border-top: 3px solid {color_hex}; 
         border-radius: 12px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); backdrop-filter: blur(10px);
         margin-bottom: 20px;">
        <p style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 8px 0;">{title}</p>
        <h2 style="color: #f8fafc; font-size: 2.2rem; font-weight: 700; margin: 0; text-shadow: 0 0 15px {color_hex}40;">{value}</h2>
    </div>
    """

# KPI row
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.markdown(kpi_card("Active Clusters", len(zones), "#3b82f6"), unsafe_allow_html=True)
with col2: st.markdown(kpi_card("CRITICAL Zones", int((zones['risk_tier'] == 'CRITICAL').sum()) if not zones.empty else 0, "#ef4444"), unsafe_allow_html=True)
with col3: st.markdown(kpi_card("HIGH Zones", int((zones['risk_tier'] == 'HIGH').sum()) if not zones.empty else 0, "#f97316"), unsafe_allow_html=True)
with col4: st.markdown(kpi_card(f"Violations ({hour_filter}h)", f"{len(violations):,}", "#8b5cf6"), unsafe_allow_html=True)
with col5: st.markdown(kpi_card("Unacked Alerts", len(alerts), "#22c55e"), unsafe_allow_html=True)

st.markdown("---")

# ── Main Workspace Tabs ────────────────────────────────────────
tab_map, tab_analytics, tab_data = st.tabs([
    "🗺️ Live Operations & Deployments", 
    "📈 Analytics & Trends", 
    "🗄️ Data Explorer"
])

# ==========================================
# TAB 1: LIVE MAP & DEPLOYMENTS
# ==========================================
with tab_map:
    st.markdown("<br>", unsafe_allow_html=True)

BLR = [12.9716, 77.5946]
m   = folium.Map(location=BLR, zoom_start=12, tiles='CartoDB dark_matter')

# Heatmap layer — respects the hours slider
if show_heatmap:
    heatmap_data = load_violations_for_heatmap(hours=hour_filter)
    if not heatmap_data.empty:
        if heatmap_type == "Congestion Impact":
            # Aggressive filter: keep ONLY the top ~7% of worst choke points
            heatmap_data = heatmap_data[heatmap_data['impact'] >= 4.0].copy()
            heatmap_data['impact'] = heatmap_data['impact'] ** 2
            heat_cols = ['latitude', 'longitude', 'impact']
            # Stark, aggressive color gradient (Purple -> Neon Pink -> Bright White)
            heat_gradient = {0.2:'#2e004f', 0.4:'#71007a', 0.6:'#c4006c', 0.8:'#ff1a55', 1.0:'#ffffff'}
            h_radius = 22
            h_blur = 15
        else:
            heatmap_data['volume_weight'] = 1.0
            heat_cols = ['latitude', 'longitude', 'volume_weight']
            # Standard density gradient (Dark blue -> Green -> Yellow)
            heat_gradient = {0.2:'#0d0221', 0.4:'#0a1045', 0.6:'#31688e', 0.8:'#35b779', 1.0:'#fde725'}
            h_radius = 18
            h_blur = 22
            
        heat_data = heatmap_data[heat_cols].dropna().values.tolist()
        if heat_data:
            HeatMap(heat_data, name=f'{heatmap_type} (heatmap)',
                    min_opacity=0.35, radius=h_radius, blur=h_blur, max_zoom=13,
                    gradient=heat_gradient
                    ).add_to(m)

# POI Markers
if show_pois:
    POIS = [
        {"name": "Indiranagar Metro", "lat": 12.9783, "lon": 77.6387, "icon": "train"},
        {"name": "MG Road Metro", "lat": 12.9755, "lon": 77.6068, "icon": "train"},
        {"name": "Majestic Bus/Metro", "lat": 12.9757, "lon": 77.5729, "icon": "train"},
        {"name": "Manyata Tech Park", "lat": 13.0450, "lon": 77.6206, "icon": "building"},
        {"name": "Ecospace Tech Park", "lat": 12.9248, "lon": 77.6806, "icon": "building"},
        {"name": "KR Market", "lat": 12.9634, "lon": 77.5760, "icon": "shopping-cart"},
        {"name": "Koramangala BDA", "lat": 12.9284, "lon": 77.6262, "icon": "shopping-cart"}
    ]
    fg_poi = folium.FeatureGroup(name='Commercial / Transit Hubs', show=True)
    for poi in POIS:
        folium.Marker(
            location=[poi['lat'], poi['lon']],
            popup=f"<b>{poi['name']}</b>",
            tooltip=poi['name'],
            icon=folium.Icon(color='purple', icon=poi['icon'], prefix='fa')
        ).add_to(fg_poi)
    fg_poi.add_to(m)

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
    st_folium(m, width='100%', height=550, key=f"map_{hour_filter}_{show_heatmap}_{show_markers}_{show_pois}_{heatmap_type}")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI Forecast ────────────────────────────────────────────────
    fcast = load_forecasts()
    if not fcast.empty:
        st.subheader("🔮 AI Prediction: Next-Shift Hotspots")
        st.markdown("<p style='color:#94a3b8; font-size:14px; margin-top:-10px;'>GradientBoosting model forecasting next 3 hours based on historical hour/dow/month patterns.</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for offset in range(1, 4):
        hour_fcast = fcast[fcast['hour_offset'] == offset]
        if hour_fcast.empty: continue
        
        target_time = str(hour_fcast.iloc[0]['target_datetime'])[11:16]
        with cols[offset-1]:
            st.markdown(f"**In {offset} hour{'s' if offset>1 else ''} ({target_time})**")
            for _, r in hour_fcast.head(5).iterrows():
                pred = r['predicted_violations']
                if pred >= 20:   color = "red"
                elif pred >= 10: color = "orange"
                else:            color = "green"
                
                st.markdown(f"- :{color}[**{pred:.0f}**] violations expected at {r['zone_name']}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🚔 Actionable Deployment Plan (Next 3 Hours)")
    st.markdown("Recommended patrol allocation based on AI predictions.")
    
    # Aggregate predictions by police station
    station_preds = fcast.groupby('police_station')['predicted_violations'].sum().reset_index()
    station_preds = station_preds[station_preds['police_station'] != 'Unknown']
    station_preds = station_preds.sort_values('predicted_violations', ascending=False).head(5)
    
    if not station_preds.empty:
        col_deploy1, col_deploy2 = st.columns([2, 3])
        
        with col_deploy1:
            for _, r in station_preds.iterrows():
                total_pred = r['predicted_violations']
                # Simple rule: 1 patrol per 15 expected violations
                patrols = max(1, int(total_pred / 15))
                st.info(f"**{r['police_station']}**\n\nDeploy **{patrols} patrol{'s' if patrols > 1 else ''}** ({total_pred:.0f} expected)")
                
        with col_deploy2:
            # Show the top zones for the top station
            top_station = station_preds.iloc[0]['police_station']
            top_station_zones = fcast[fcast['police_station'] == top_station]
            top_station_zones = top_station_zones.groupby('zone_name')['predicted_violations'].sum().reset_index()
            top_station_zones = top_station_zones.sort_values('predicted_violations', ascending=False).head(3)
            
            st.markdown(f"**🎯 Priority targets for {top_station} patrols:**")
            for _, r in top_station_zones.iterrows():
                st.markdown(f"- 📍 {r['zone_name']} ({r['predicted_violations']:.0f} violations)")

# ==========================================
# TAB 2: ANALYTICS & TRENDS
# ==========================================
with tab_analytics:
    st.markdown("<br>", unsafe_allow_html=True)
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

# ==========================================
# TAB 3: DATA EXPLORER
# ==========================================
with tab_data:
    st.markdown("<br>", unsafe_allow_html=True)
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
    
    st.markdown("---")
    
    # ── Alert history table ────────────────────────────────────────
    st.subheader("📋 Full Alert Log")
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
