import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Gridlock AI — Parking Enforcement",
                   page_icon="🚔", layout="wide")

@st.cache_data
def load():
    df      = pd.read_csv("parking_clustered.csv", parse_dates=['created_datetime'])
    zones   = pd.read_csv("top_enforcement_zones.csv")
    clusters= pd.read_csv("cluster_stats.csv")
    return df, zones, clusters

df, zones, clusters = load()

# ── SIDEBAR ───────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Filters")
hour_range  = st.sidebar.slider("Hour of day", 0, 23, (0, 23))
risk_filter = st.sidebar.multiselect("Risk tier",
    options=['CRITICAL','HIGH','MEDIUM','LOW'],
    default=['CRITICAL','HIGH'])
show_top_n  = st.sidebar.slider("Top N zones on map", 5, 20, 10)

filt = df[df['hour'].between(hour_range[0], hour_range[1])]

# ── HEADER ────────────────────────────────────────────────────────────────
st.title("🚔 Gridlock AI — Parking Enforcement Intelligence")
st.caption("Flipkart × Bengaluru Traffic Police | Gridlock Hackathon 2.0")

# ── KPI CARDS ─────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Total violations",      f"{len(df):,}")
k2.metric("Hotspot clusters",      "189")
k3.metric("CRITICAL zones",        "3")
k4.metric("HIGH risk zones",       "74")
k5.metric("Overnight share",       "80.7%",  delta="vs 12.5% daytime")
k6.metric("Avg blockage (top 10)", "20.5%")
st.divider()

# ── MAP + ZONE TABLE ──────────────────────────────────────────────────────
col_map, col_tbl = st.columns([3, 2])

with col_map:
    st.subheader("Violation heatmap & enforcement zones")
    tier_colors = {'CRITICAL':'#ef4444','HIGH':'#f97316',
                   'MEDIUM':'#eab308','LOW':'#22c55e'}

    BLR = [12.9716, 77.5946]
    m   = folium.Map(location=BLR, zoom_start=12, tiles='CartoDB dark_matter')

    heat_data = filt[['latitude','longitude','impact']].dropna().values.tolist()
    if heat_data:
        HeatMap(heat_data, radius=10, blur=14,
                gradient={0.2:'#440154',0.45:'#31688e',
                          0.7:'#35b779',1.0:'#fde725'}).add_to(m)

    plot_zones = zones[zones['risk_tier'].isin(risk_filter)].head(show_top_n) \
                 if risk_filter else zones.head(show_top_n)

    for _, r in plot_zones.iterrows():
        col  = tier_colors.get(r['risk_tier'], '#94a3b8')
        size = 10 + (r['congestion_score'] / 100) * 22
        html = f"""
        <div style='font-family:system-ui;min-width:250px;font-size:13px'>
          <div style='background:{col};color:#fff;padding:6px 10px;
               border-radius:6px 6px 0 0;font-weight:600'>
            #{int(r['rank'])} &nbsp;{r['zone_name']}
            &nbsp;<span style='background:rgba(0,0,0,0.25);
            padding:1px 6px;border-radius:4px;font-size:11px'>{r['risk_tier']}</span>
          </div>
          <div style='padding:8px 10px;background:#1e1e2e;color:#e2e8f0;
               border-radius:0 0 6px 6px'>
            <table style='width:100%;font-size:12px'>
              <tr><td style='color:#94a3b8'>Police station</td>
                  <td style='text-align:right'><b>{r['police_station']}</b></td></tr>
              <tr><td style='color:#94a3b8'>Violations</td>
                  <td style='text-align:right'><b>{int(r['violation_count']):,}</b></td></tr>
              <tr><td style='color:#94a3b8'>Daily rate</td>
                  <td style='text-align:right'><b>{r['daily_rate']:.0f}/day</b></td></tr>
              <tr><td style='color:#94a3b8'>Congestion score</td>
                  <td style='text-align:right;color:{col}'><b>{r['congestion_score']}/100</b></td></tr>
              <tr><td style='color:#94a3b8'>Avg blockage</td>
                  <td style='text-align:right'><b>{r['blockage_pct']}%</b></td></tr>
              <tr><td style='color:#94a3b8'>Top vehicle</td>
                  <td style='text-align:right'><b>{r['top_vehicle']}</b></td></tr>
              <tr><td style='color:#94a3b8'>Deploy shift</td>
                  <td style='text-align:right'><b>{r['enforcement_shift']}</b></td></tr>
            </table>
          </div>
        </div>"""
        folium.CircleMarker(
            location=[r['lat'], r['lon']],
            radius=size, color=col,
            fill=True, fill_color=col, fill_opacity=0.8,
            popup=folium.Popup(html, max_width=290),
            tooltip=f"#{int(r['rank'])} {r['zone_name']} | {r['risk_tier']} | {r['congestion_score']}/100"
        ).add_to(m)
        folium.Marker(
            location=[r['lat'], r['lon']],
            icon=folium.DivIcon(
                html=f"<div style='font-size:9px;font-weight:700;color:#fff;"
                     f"text-shadow:0 0 3px #000'>#{int(r['rank'])}</div>",
                icon_size=(18,13), icon_anchor=(9,6))
        ).add_to(m)

    legend = """<div style='position:fixed;bottom:20px;left:20px;z-index:9999;
         background:rgba(15,15,30,0.88);color:#e2e8f0;padding:12px 16px;
         border-radius:10px;font-family:system-ui;font-size:12px;
         border:1px solid rgba(255,255,255,0.12)'>
      <b style='font-size:13px'>Risk tier</b><br>
      <span style='color:#ef4444'>&#9679;</span> CRITICAL<br>
      <span style='color:#f97316'>&#9679;</span> HIGH<br>
      <span style='color:#eab308'>&#9679;</span> MEDIUM<br>
      <span style='color:#22c55e'>&#9679;</span> LOW
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    st_folium(m, width=700, height=500)

with col_tbl:
    st.subheader("Enforcement zones")
    tier_emoji = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢'}
    disp = zones.copy()
    disp['tier'] = disp['risk_tier'].map(tier_emoji) + ' ' + disp['risk_tier']
    show = disp[['rank','zone_name','violation_count','daily_rate',
                 'congestion_score','blockage_pct','tier','enforcement_shift']]
    show.columns = ['#','Zone','Total','/day','Score','Block%','Tier','Shift']
    if risk_filter:
        show = show[disp['risk_tier'].isin(risk_filter)]
    st.dataframe(show.head(show_top_n), use_container_width=True, hide_index=True)

st.divider()

# ── SCORE COMPONENT RADAR (top 5 zones) ──────────────────────────────────
st.subheader("Score component breakdown — top 5 zones")
components = ['c_impact','c_lanes','c_peak','c_consistency','c_persistence']
labels     = ['Traffic impact','Vehicle severity','Peak timing',
              'Consistency','Persistence']
top5 = zones.head(5)

fig_radar = go.Figure()
for _, r in top5.iterrows():
    vals = [r[c] for c in components]
    vals += [vals[0]]  # close the loop
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=labels + [labels[0]],
        fill='toself',
        name=r['zone_name'][:25],
        opacity=0.7
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
    template='plotly_dark',
    legend=dict(orientation='h', y=-0.2),
    margin=dict(t=20,b=60),
    height=420
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── CHARTS ROW ────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Violations by hour")
    hourly = df.groupby('hour').size().reset_index(name='count')
    hourly['period'] = hourly['hour'].apply(
        lambda h: 'Overnight' if h in list(range(0,6))+list(range(19,24)) else 'Daytime'
    )
    fig = px.bar(hourly, x='hour', y='count', color='period',
                 color_discrete_map={'Overnight':'#f97316','Daytime':'#3b82f6'},
                 labels={'hour':'Hour','count':'Violations'},
                 template='plotly_dark')
    fig.update_layout(margin=dict(t=10,b=10), legend_title_text='')
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Daily rate vs congestion score (top 20)")
    fig2 = px.scatter(zones.head(20),
                      x='daily_rate', y='congestion_score',
                      size='violation_count', color='risk_tier',
                      hover_name='zone_name',
                      color_discrete_map={'CRITICAL':'#ef4444','HIGH':'#f97316',
                                          'MEDIUM':'#eab308','LOW':'#22c55e'},
                      labels={'daily_rate':'Violations/day',
                              'congestion_score':'Congestion score'},
                      template='plotly_dark')
    fig2.update_layout(margin=dict(t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ── DEPLOYMENT PLAN ───────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Enforcement deployment plan")
st.info(
    "**Model recommendation:** 80.7% of violations occur overnight (7pm–5am). "
    "3 CRITICAL + 74 HIGH zones identified. "
    "Prioritise top 5 zones with dedicated late-night units — "
    "estimated 20.5% avg carriageway recovery across critical zones."
)
rec = zones[['rank','zone_name','police_station','daily_rate',
             'congestion_score','blockage_pct','risk_tier','enforcement_shift']].head(15).copy()
rec.columns = ['#','Zone','Station','Viol/day','Score','Block%','Tier','Recommended shift']
st.dataframe(rec, use_container_width=True, hide_index=True)

# ── FOOTER ────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: DBSCAN clustering (eps=150m) + 5-component congestion scoring "
           "(impact 35% · vehicle severity 20% · peak timing 15% · "
           "consistency 20% · persistence 10%) | "
           "Data: Jan 2023 – Apr 2024 | Gridlock Hackathon 2.0")
