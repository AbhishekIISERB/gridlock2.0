import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap

EARTH_R = 6371000

df       = pd.read_csv("parking_clustered.csv")
zones    = pd.read_csv("top_enforcement_zones.csv")
clusters = pd.read_csv("cluster_stats.csv")

print(f"Plotting {len(df):,} violations, {len(zones)} top zones")

BLR = [12.9716, 77.5946]
m = folium.Map(location=BLR, zoom_start=12, tiles='CartoDB dark_matter')

# ── LAYER 1: FULL HEATMAP ─────────────────────────────────────────────────
heat_data = df[['latitude','longitude','impact']].dropna().values.tolist()
HeatMap(heat_data, name='All violations (heatmap)',
        min_opacity=0.3, radius=10, blur=14,
        gradient={0.2:'#440154',0.45:'#31688e',0.7:'#35b779',1.0:'#fde725'}
        ).add_to(m)

# ── LAYER 2: OVERNIGHT HEATMAP ────────────────────────────────────────────
overnight = df[df['hour'].isin(list(range(0,6))+list(range(19,24)))]
HeatMap(overnight[['latitude','longitude']].values.tolist(),
        name='Overnight only (7pm–5am)', show=False,
        radius=10, blur=14,
        gradient={0.3:'#03045e',0.6:'#0077b6',1.0:'#90e0ef'}
        ).add_to(m)

# ── LAYER 3: DAYTIME HEATMAP ──────────────────────────────────────────────
daytime = df[df['hour'].isin(range(7,19))]
HeatMap(daytime[['latitude','longitude']].values.tolist(),
        name='Daytime only (7am–7pm)', show=False,
        radius=10, blur=14,
        gradient={0.3:'#7f1d1d',0.6:'#f97316',1.0:'#fef08a'}
        ).add_to(m)

# ── LAYER 4: ENFORCEMENT ZONE MARKERS ─────────────────────────────────────
def score_color(s):
    if s >= 80: return '#ef4444'
    if s >= 60: return '#f97316'
    if s >= 40: return '#eab308'
    return '#22c55e'

zone_fg = folium.FeatureGroup(name='Top 20 enforcement zones', show=True)

for _, r in zones.iterrows():
    col    = score_color(r['congestion_score'])
    radius = 12 + (r['congestion_score'] / 100) * 22

    html = f"""
    <div style='font-family:system-ui,sans-serif;min-width:240px;font-size:13px'>
      <div style='background:{col};color:#fff;padding:6px 10px;border-radius:6px 6px 0 0;font-weight:600'>
        #{int(r['rank'])} &nbsp; {r['zone_name']}
      </div>
      <div style='padding:8px 10px;background:#1e1e2e;color:#e2e8f0;border-radius:0 0 6px 6px'>
        <table style='width:100%;border-collapse:collapse'>
          <tr><td style='color:#94a3b8'>Police station</td>
              <td style='text-align:right;font-weight:500'>{r['police_station']}</td></tr>
          <tr><td style='color:#94a3b8'>Violations</td>
              <td style='text-align:right;font-weight:500'>{int(r['violation_count']):,}</td></tr>
          <tr><td style='color:#94a3b8'>Congestion score</td>
              <td style='text-align:right;font-weight:600;color:{col}'>{r['congestion_score']}/100</td></tr>
          <tr><td style='color:#94a3b8'>Avg blockage</td>
              <td style='text-align:right;font-weight:500'>{r['blockage_pct']}%</td></tr>
          <tr><td style='color:#94a3b8'>Peak hour %</td>
              <td style='text-align:right;font-weight:500'>{r['peak_pct_pct']:.1f}%</td></tr>
          <tr><td style='color:#94a3b8'>Top vehicle</td>
              <td style='text-align:right;font-weight:500'>{r['top_vehicle']}</td></tr>
        </table>
      </div>
    </div>"""

    folium.CircleMarker(
        location=[r['lat'], r['lon']],
        radius=radius, color=col,
        fill=True, fill_color=col, fill_opacity=0.75,
        popup=folium.Popup(html, max_width=280),
        tooltip=f"#{int(r['rank'])} {r['zone_name']} | Score: {r['congestion_score']}"
    ).add_to(zone_fg)

    folium.Marker(
        location=[r['lat'], r['lon']],
        icon=folium.DivIcon(
            html=f"<div style='font-size:9px;font-weight:700;color:#fff;"
                 f"text-shadow:0 0 3px #000'>#{int(r['rank'])}</div>",
            icon_size=(18,13), icon_anchor=(9,6)
        )
    ).add_to(zone_fg)

zone_fg.add_to(m)

# ── LEGEND ────────────────────────────────────────────────────────────────
legend = """
<div style='position:fixed;bottom:28px;left:28px;z-index:9999;
     background:rgba(15,15,30,0.88);color:#e2e8f0;
     padding:14px 18px;border-radius:10px;font-family:system-ui;font-size:12px;
     border:1px solid rgba(255,255,255,0.12)'>
  <div style='font-weight:600;margin-bottom:8px;font-size:13px'>Congestion score</div>
  <div><span style='color:#ef4444;font-size:16px'>&#9679;</span>&nbsp; Critical &ge;80</div>
  <div><span style='color:#f97316;font-size:16px'>&#9679;</span>&nbsp; High 60–79</div>
  <div><span style='color:#eab308;font-size:16px'>&#9679;</span>&nbsp; Medium 40–59</div>
  <div><span style='color:#22c55e;font-size:16px'>&#9679;</span>&nbsp; Lower &lt;40</div>
  <div style='margin-top:8px;color:#64748b;font-size:10px'>Circle size = relative severity<br>Click circle for full details</div>
</div>"""
m.get_root().html.add_child(folium.Element(legend))

folium.LayerControl(collapsed=False).add_to(m)

m.save("parking_heatmap.html")
print("Saved: parking_heatmap.html — open in browser")
