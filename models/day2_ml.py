import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD CLEAN DATA ────────────────────────────────────────────────────
df = pd.read_csv("parking_clean.csv", parse_dates=['created_datetime'])
print(f"Loaded: {df.shape}")

# ── 2. FIX PEAK HOURS (data shows overnight peak, not rush hour) ──────────
# Real peaks: midnight-5am AND 7pm-midnight
df['is_peak'] = df['hour'].isin(list(range(0, 6)) + list(range(19, 24)))
print(f"Corrected peak violations: {df['is_peak'].sum()} ({df['is_peak'].mean()*100:.1f}%)")

# ── 3. DBSCAN CLUSTERING ──────────────────────────────────────────────────
# eps=0.0008 degrees ≈ 90 metres radius in Bengaluru
# min_samples=30 = at least 30 violations to form a hotspot
coords = df[['latitude', 'longitude']].values

print("\nRunning DBSCAN... (may take 30-60s on 291K rows)")
db = DBSCAN(eps=0.0008, min_samples=30, algorithm='ball_tree',
            metric='haversine', n_jobs=-1)

# Haversine needs radians
coords_rad = np.radians(coords)
labels = db.fit_predict(coords_rad)

df['cluster'] = labels
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise} ({n_noise/len(df)*100:.1f}%)")

# ── 4. CLUSTER STATS ──────────────────────────────────────────────────────
clustered = df[df['cluster'] != -1].copy()

cluster_stats = clustered.groupby('cluster').agg(
    violation_count   = ('id', 'count'),
    lat               = ('latitude', 'mean'),
    lon               = ('longitude', 'mean'),
    avg_severity      = ('severity_weight', 'mean'),
    peak_pct          = ('is_peak', 'mean'),
    unique_junctions  = ('junction_name', 'nunique'),
    top_junction      = ('junction_name', lambda x: x.value_counts().index[0]),
    top_vehicle       = ('vehicle_type', lambda x: x.value_counts().index[0]),
    police_station    = ('police_station', lambda x: x.value_counts().index[0]),
).reset_index()

# ── 5. CONGESTION IMPACT SCORE ────────────────────────────────────────────
# Score = violations × severity × peak concentration × density bonus
# All components normalized to [0,1] then combined

def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

cluster_stats['score_volume']   = minmax(cluster_stats['violation_count'])
cluster_stats['score_severity'] = minmax(cluster_stats['avg_severity'])
cluster_stats['score_peak']     = minmax(cluster_stats['peak_pct'])

# Weighted composite: volume matters most (50%), severity (30%), peak timing (20%)
cluster_stats['congestion_score'] = (
    0.50 * cluster_stats['score_volume'] +
    0.30 * cluster_stats['score_severity'] +
    0.20 * cluster_stats['score_peak']
)

# Normalize final score to 0-100
cluster_stats['congestion_score'] = (
    minmax(cluster_stats['congestion_score']) * 100
).round(1)

# ── 6. CARRIAGEWAY BLOCKAGE ESTIMATE ─────────────────────────────────────
# Rule-based: heavy vehicles block more lanes
# Bus/Truck/HGV avg = 2 lanes, Car/Van = 1 lane, 2-wheeler = 0.3 lanes
vehicle_lane_block = {
    'SCOOTER': 0.3, 'MOTOR CYCLE': 0.3, 'MOPED': 0.3,
    'CAR': 1.0, 'JEEP': 1.0,
    'PASSENGER AUTO': 0.8, 'GOODS AUTO': 0.8,
    'VAN': 1.5, 'TEMPO': 1.5, 'LGV': 1.5,
    'MAXI-CAB': 1.8,
    'PRIVATE BUS': 2.5, 'BUS (BMTC/KSRTC)': 2.5,
    'LORRY/GOODS VEHICLE': 2.5, 'HGV': 2.5,
}
df['lanes_blocked'] = df['vehicle_type'].map(vehicle_lane_block).fillna(1.0)

lane_stats = df[df['cluster'] != -1].groupby('cluster')['lanes_blocked'].mean()
cluster_stats = cluster_stats.merge(lane_stats.rename('avg_lanes_blocked'),
                                    on='cluster')
# Assume avg road = 4 lanes; blockage % = (avg_lanes_blocked / 4) * 100
cluster_stats['blockage_pct'] = (
    (cluster_stats['avg_lanes_blocked'] / 4) * 100
).round(1)

# ── 7. TOP ENFORCEMENT ZONES ──────────────────────────────────────────────
top_zones = cluster_stats.nlargest(20, 'congestion_score').copy()
top_zones['rank'] = range(1, len(top_zones) + 1)

print("\n=== TOP 15 ENFORCEMENT ZONES ===")
display_cols = ['rank', 'top_junction', 'police_station', 'violation_count',
                'congestion_score', 'blockage_pct', 'top_vehicle', 'peak_pct']
top_zones['peak_pct'] = (top_zones['peak_pct'] * 100).round(1)
print(top_zones[display_cols].head(15).to_string(index=False))

# ── 8. KPI SUMMARY FOR PITCH ──────────────────────────────────────────────
total = len(df)
top10_count = cluster_stats.nlargest(10, 'congestion_score')['violation_count'].sum()
print("\n=== PITCH KPIs ===")
print(f"Total parking violations analysed : {total:,}")
print(f"Hotspot clusters identified        : {n_clusters}")
print(f"Top 10 clusters capture            : {top10_count:,} violations "
      f"({top10_count/total*100:.1f}% of all)")
print(f"Overnight violations (7pm-5am)     : {df['is_peak'].sum():,} "
      f"({df['is_peak'].mean()*100:.1f}%)")
print(f"Avg carriageway blockage (top 10)  : "
      f"{cluster_stats.nlargest(10,'congestion_score')['blockage_pct'].mean():.1f}%")
print(f"Highest congestion score zone      : "
      f"{top_zones.iloc[0]['top_junction']} "
      f"(score: {top_zones.iloc[0]['congestion_score']})")

# ── 9. SAVE OUTPUTS ───────────────────────────────────────────────────────
cluster_stats.to_csv("cluster_stats.csv", index=False)
top_zones.to_csv("top_enforcement_zones.csv", index=False)
df[['id','latitude','longitude','cluster','congestion_score'
    if 'congestion_score' in df.columns else 'severity_weight']].to_csv(
    "violations_with_clusters.csv", index=False)

# Re-save df with cluster labels for heatmap
df.to_csv("parking_clustered.csv", index=False)

print("\nSaved:")
print("  cluster_stats.csv           — all cluster details")
print("  top_enforcement_zones.csv   — ranked top 20 zones")
print("  parking_clustered.csv       — full data with cluster labels")
