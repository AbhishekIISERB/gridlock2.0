import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')

EARTH_R = 6371000

def metres_to_rad(m):
    return m / EARTH_R

# ── 1. LOAD ────────────────────────────────────────────────────────────────
df = pd.read_csv("parking_clean.csv", parse_dates=['created_datetime'])
print(f"Loaded: {df.shape}")

df['is_peak'] = df['hour'].isin(list(range(0, 6)) + list(range(19, 24)))

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

# ── 2. DBSCAN ON SAMPLE ────────────────────────────────────────────────────
SAMPLE_N    = 60000
EPS_M       = 150
MIN_SAMPLES = 20

sample     = df.sample(n=SAMPLE_N, random_state=42).copy()
coords_rad = np.radians(sample[['latitude', 'longitude']].values)

print(f"Running DBSCAN (eps={EPS_M}m, min_samples={MIN_SAMPLES})...")
db = DBSCAN(eps=metres_to_rad(EPS_M), min_samples=MIN_SAMPLES,
            algorithm='ball_tree', metric='haversine')
sample['cluster'] = db.fit_predict(coords_rad)

n_clusters = len(set(sample['cluster'])) - (1 if -1 in sample['cluster'].values else 0)
print(f"Clusters found: {n_clusters}")

# ── 3. ASSIGN ALL POINTS — wider radius ───────────────────────────────────
centroids = (sample[sample['cluster'] != -1]
             .groupby('cluster')[['latitude','longitude']]
             .mean().reset_index())

tree      = BallTree(np.radians(centroids[['latitude','longitude']].values),
                     metric='haversine')
all_rad   = np.radians(df[['latitude','longitude']].values)
dist, idx = tree.query(all_rad, k=1)
dist_m    = dist.flatten() * EARTH_R

# Raise to 400m so dispersed violations still get attributed
ASSIGN_R_M = 400
df['cluster']         = np.where(dist_m <= ASSIGN_R_M,
                                  centroids['cluster'].values[idx.flatten()], -1)
df['dist_centroid_m'] = dist_m.round(1)

assigned = (df['cluster'] != -1).sum()
print(f"Assigned: {assigned:,} ({assigned/len(df)*100:.1f}%)")

# ── 4. BETTER ZONE LABELS ─────────────────────────────────────────────────
# Priority: named junction → police station area → lat/lon fallback
def best_zone_label(group):
    named = group['junction_name']
    named = named[named.str.strip() != 'No Junction']
    if len(named) > 0:
        top = named.value_counts().index[0]
        # Strip BTP code prefix for cleaner display
        label = top.split(' - ')[-1] if ' - ' in top else top
        return label
    # Fall back to police station
    ps = group['police_station'].value_counts().index[0]
    return f"{ps} area"

clustered = df[df['cluster'] != -1].copy()

# Build stats with better label
def top_junction_clean(x):
    named = x[x.str.strip() != 'No Junction']
    if len(named) > 0:
        raw = named.value_counts().index[0]
        return raw.split(' - ')[-1] if ' - ' in raw else raw
    return None  # will be filled below

def top_ps(x):
    return x.value_counts().index[0]

cluster_stats = clustered.groupby('cluster').agg(
    violation_count   = ('id',              'count'),
    lat               = ('latitude',        'mean'),
    lon               = ('longitude',       'mean'),
    avg_severity      = ('severity_weight', 'mean'),
    peak_pct          = ('is_peak',         'mean'),
    avg_lanes_blocked = ('lanes_blocked',   'mean'),
    junction_label    = ('junction_name',   top_junction_clean),
    police_station    = ('police_station',  top_ps),
    top_vehicle       = ('vehicle_type',    lambda x: x.value_counts().index[0]),
).reset_index()

# Fill nulls: use "police_station area" when no named junction
cluster_stats['zone_name'] = cluster_stats.apply(
    lambda r: r['junction_label'] if pd.notna(r['junction_label'])
              else f"{r['police_station']} area",
    axis=1
)

# ── 5. CONGESTION SCORE ────────────────────────────────────────────────────
def minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / (rng if rng > 0 else 1)

cluster_stats['congestion_score'] = (
    0.50 * minmax(cluster_stats['violation_count']) +
    0.30 * minmax(cluster_stats['avg_lanes_blocked']) +
    0.20 * minmax(cluster_stats['peak_pct'])
)
cluster_stats['congestion_score'] = (minmax(cluster_stats['congestion_score']) * 100).round(1)
cluster_stats['blockage_pct']     = ((cluster_stats['avg_lanes_blocked'] / 4) * 100).round(1)

# ── 6. TOP ZONES ──────────────────────────────────────────────────────────
top_zones = cluster_stats.nlargest(20, 'congestion_score').copy()
top_zones['rank'] = range(1, len(top_zones) + 1)
top_zones['peak_pct_pct'] = (top_zones['peak_pct'] * 100).round(1)

print("\n=== TOP 15 ENFORCEMENT ZONES ===")
cols = ['rank','zone_name','police_station','violation_count',
        'congestion_score','blockage_pct','top_vehicle','peak_pct_pct']
pd.set_option('display.max_colwidth', 35)
pd.set_option('display.width', 120)
print(top_zones[cols].head(15).to_string(index=False))

# ── 7. PITCH KPIs ─────────────────────────────────────────────────────────
top10 = cluster_stats.nlargest(10, 'congestion_score')
print("\n=== PITCH KPIs ===")
print(f"Total violations analysed          : {len(df):,}")
print(f"Hotspot clusters identified         : {n_clusters}")
print(f"Violations inside clusters          : {assigned:,} ({assigned/len(df)*100:.1f}%)")
print(f"Top 10 clusters capture             : {top10['violation_count'].sum():,} "
      f"({top10['violation_count'].sum()/len(df)*100:.1f}%)")
print(f"Overnight violations (7pm–5am)      : {df['is_peak'].sum():,} ({df['is_peak'].mean()*100:.1f}%)")
print(f"Avg carriageway blockage (top 10)   : {top10['blockage_pct'].mean():.1f}%")
print(f"#1 zone                             : {top_zones.iloc[0]['zone_name']}")
print(f"#1 congestion score                 : {top_zones.iloc[0]['congestion_score']}/100")

# ── 8. SAVE ───────────────────────────────────────────────────────────────
cluster_stats.to_csv("cluster_stats.csv", index=False)
top_zones.to_csv("top_enforcement_zones.csv", index=False)
df.to_csv("parking_clustered.csv", index=False)
print("\nSaved: cluster_stats.csv, top_enforcement_zones.csv, parking_clustered.csv")
