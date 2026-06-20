import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

EARTH_R = 6371000
def m2r(m): return m / EARTH_R

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════
df = pd.read_csv("parking_clean.csv", parse_dates=['created_datetime'])
print(f"Loaded: {df.shape}")

df['is_peak'] = df['hour'].isin(list(range(0,6)) + list(range(19,24)))

# ══════════════════════════════════════════════════════════════════════════
# 2. RICHER FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════

# 2a. Violation type severity (how much does THIS violation block traffic?)
vtype_severity = {
    # Full carriageway block
    'PARKING IN A MAIN ROAD': 3.0,
    'DOUBLE PARKING':         3.0,
    # Partial block
    'NO PARKING':             2.0,
    'WRONG PARKING':          1.5,
    # Footpath / secondary
    'PARKING ON FOOTPATH':    1.2,
    'PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC': 1.8,
}
def violation_severity(vtype_str):
    if not isinstance(vtype_str, str): return 1.0
    score = 1.0
    for k, v in vtype_severity.items():
        if k in vtype_str.upper():
            score = max(score, v)
    return score

df['vtype_severity'] = df['violation_type'].apply(violation_severity)

# 2b. Vehicle lane block weight (unchanged from v1)
vehicle_lane = {
    'SCOOTER':0.3,'MOTOR CYCLE':0.3,'MOPED':0.3,
    'CAR':1.0,'JEEP':1.0,
    'PASSENGER AUTO':0.8,'GOODS AUTO':0.8,
    'VAN':1.5,'TEMPO':1.5,'LGV':1.5,'MAXI-CAB':1.8,
    'PRIVATE BUS':2.5,'BUS (BMTC/KSRTC)':2.5,
    'LORRY/GOODS VEHICLE':2.5,'HGV':2.5,
}
df['lanes_blocked'] = df['vehicle_type'].map(vehicle_lane).fillna(1.0)

# 2c. COMBINED per-violation impact score
df['impact'] = df['vtype_severity'] * df['lanes_blocked']

# 2d. Time features
df['date']        = pd.to_datetime(df['created_datetime']).dt.date
df['week']        = pd.to_datetime(df['created_datetime']).dt.isocalendar().week.astype(int)

print("Feature engineering done.")

# ══════════════════════════════════════════════════════════════════════════
# 3. DBSCAN — same as v1 (proven to work)
# ══════════════════════════════════════════════════════════════════════════
SAMPLE_N    = 60000
EPS_M       = 150
MIN_SAMPLES = 20

sample     = df.sample(n=SAMPLE_N, random_state=42).copy()
coords_rad = np.radians(sample[['latitude','longitude']].values)

print(f"Running DBSCAN (eps={EPS_M}m, min_samples={MIN_SAMPLES})...")
db = DBSCAN(eps=m2r(EPS_M), min_samples=MIN_SAMPLES,
            algorithm='ball_tree', metric='haversine')
sample['cluster'] = db.fit_predict(coords_rad)

n_clusters = len(set(sample['cluster'])) - (1 if -1 in sample['cluster'].values else 0)
print(f"Clusters: {n_clusters}")

# Assign all points (400m radius)
centroids = (sample[sample['cluster'] != -1]
             .groupby('cluster')[['latitude','longitude']].mean().reset_index())
tree      = BallTree(np.radians(centroids[['latitude','longitude']].values),
                     metric='haversine')
dist, idx = tree.query(np.radians(df[['latitude','longitude']].values), k=1)
dist_m    = dist.flatten() * EARTH_R

df['cluster']         = np.where(dist_m <= 400,
                                  centroids['cluster'].values[idx.flatten()], -1)
df['dist_centroid_m'] = dist_m.round(1)
assigned = (df['cluster'] != -1).sum()
print(f"Assigned: {assigned:,} ({assigned/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# 4. CLUSTER-LEVEL FEATURE AGGREGATION (richer than v1)
# ══════════════════════════════════════════════════════════════════════════
clustered = df[df['cluster'] != -1].copy()

def top_junction_clean(x):
    named = x[x.str.strip() != 'No Junction']
    if len(named) > 0:
        raw = named.value_counts().index[0]
        return raw.split(' - ')[-1] if ' - ' in raw else raw
    return None

def top_val(x): return x.value_counts().index[0]

agg = clustered.groupby('cluster').agg(
    violation_count    = ('id',             'count'),
    lat                = ('latitude',       'mean'),
    lon                = ('longitude',      'mean'),
    # NEW: sum of per-violation impact (vtype × lanes)
    total_impact       = ('impact',         'sum'),
    avg_impact         = ('impact',         'mean'),
    avg_lanes_blocked  = ('lanes_blocked',  'mean'),
    peak_pct           = ('is_peak',        'mean'),
    # NEW: how many unique days does this zone appear? (consistency)
    active_days        = ('date',           'nunique'),
    # NEW: how many unique weeks? (persistence across months)
    active_weeks       = ('week',           'nunique'),
    junction_label     = ('junction_name',  top_junction_clean),
    police_station     = ('police_station', top_val),
    top_vehicle        = ('vehicle_type',   top_val),
).reset_index()

# Zone label
agg['zone_name'] = agg.apply(
    lambda r: r['junction_label'] if pd.notna(r['junction_label'])
              else f"{r['police_station']} area", axis=1)

# ══════════════════════════════════════════════════════════════════════════
# 5. IMPROVED CONGESTION SCORE
# ══════════════════════════════════════════════════════════════════════════
# v1 used: 50% volume + 30% lanes + 20% peak
# v2 uses: 5 dimensions, each normalised 0→1

def mm(s):
    rng = s.max() - s.min()
    return (s - s.min()) / (rng if rng > 0 else 1)

# Component 1 — Total traffic impact (volume × severity × lanes)
agg['c_impact']      = mm(agg['total_impact'])

# Component 2 — Vehicle severity (heavy vehicles = more blockage)
agg['c_lanes']       = mm(agg['avg_lanes_blocked'])

# Component 3 — Peak hour concentration (overnight = enforcement gap)
agg['c_peak']        = mm(agg['peak_pct'])

# Component 4 — Consistency (active across many days = chronic problem)
agg['c_consistency'] = mm(agg['active_days'])

# Component 5 — Persistence (active across many weeks = structural issue)
agg['c_persistence'] = mm(agg['active_weeks'])

# Weighted composite — judges want "quantify impact", so impact leads
WEIGHTS = {
    'c_impact':      0.35,   # primary: actual traffic disruption
    'c_lanes':       0.20,   # vehicle heaviness
    'c_peak':        0.15,   # enforcement timing gap
    'c_consistency': 0.20,   # chronic vs one-off
    'c_persistence': 0.10,   # structural vs seasonal
}
agg['raw_score'] = sum(agg[k] * w for k, w in WEIGHTS.items())
agg['congestion_score'] = (mm(agg['raw_score']) * 100).round(1)

# Blockage estimate
agg['blockage_pct'] = ((agg['avg_lanes_blocked'] / 4) * 100).round(1)

# Daily violation rate (chronic intensity metric)
agg['daily_rate'] = (agg['violation_count'] / agg['active_days']).round(1)

# ══════════════════════════════════════════════════════════════════════════
# 6. TEMPORAL PROFILE PER CLUSTER (for pitch + dashboard)
# ══════════════════════════════════════════════════════════════════════════
# Peak shift classification
def shift_label(peak_pct):
    if peak_pct > 0.85: return 'Late night (11pm–4am)'
    if peak_pct > 0.60: return 'Evening + night (7pm–midnight)'
    return 'Mixed / daytime'

agg['enforcement_shift'] = agg['peak_pct'].apply(shift_label)

# ══════════════════════════════════════════════════════════════════════════
# 7. RISK TIER
# ══════════════════════════════════════════════════════════════════════════
def risk_tier(score):
    if score >= 75: return 'CRITICAL'
    if score >= 50: return 'HIGH'
    if score >= 25: return 'MEDIUM'
    return 'LOW'

agg['risk_tier'] = agg['congestion_score'].apply(risk_tier)

tier_counts = agg['risk_tier'].value_counts()
print("\nRisk tier distribution:")
print(tier_counts)

# ══════════════════════════════════════════════════════════════════════════
# 8. TOP ZONES
# ══════════════════════════════════════════════════════════════════════════
top_zones = agg.nlargest(20, 'congestion_score').copy()
top_zones['rank'] = range(1, len(top_zones) + 1)
top_zones['peak_pct_pct'] = (top_zones['peak_pct'] * 100).round(1)

print("\n=== TOP 15 ENFORCEMENT ZONES (v2) ===")
cols = ['rank','zone_name','police_station','violation_count','daily_rate',
        'congestion_score','blockage_pct','risk_tier','enforcement_shift']
pd.set_option('display.max_colwidth', 32)
pd.set_option('display.width', 140)
print(top_zones[cols].head(15).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# 9. PITCH KPIs
# ══════════════════════════════════════════════════════════════════════════
top10 = agg.nlargest(10, 'congestion_score')
critical = agg[agg['risk_tier'] == 'CRITICAL']
high     = agg[agg['risk_tier'] == 'HIGH']

print("\n=== PITCH KPIs (v2) ===")
print(f"Total violations analysed           : {len(df):,}")
print(f"Hotspot clusters identified          : {n_clusters}")
print(f"CRITICAL zones                       : {len(critical)}")
print(f"HIGH risk zones                      : {len(high)}")
print(f"Violations inside clusters           : {assigned:,} ({assigned/len(df)*100:.1f}%)")
print(f"Top 10 total impact score            : {top10['total_impact'].sum():,.0f}")
print(f"Overnight violations (7pm–5am)       : {df['is_peak'].sum():,} ({df['is_peak'].mean()*100:.1f}%)")
print(f"Avg blockage % (top 10)              : {top10['blockage_pct'].mean():.1f}%")
print(f"Avg daily violation rate (#1 zone)   : {top_zones.iloc[0]['daily_rate']:.0f}/day")
print(f"#1 zone                              : {top_zones.iloc[0]['zone_name']}")
print(f"#1 congestion score                  : {top_zones.iloc[0]['congestion_score']}/100")
print(f"#1 enforcement shift                 : {top_zones.iloc[0]['enforcement_shift']}")

# ══════════════════════════════════════════════════════════════════════════
# 10. SAVE
# ══════════════════════════════════════════════════════════════════════════
agg.to_csv("cluster_stats.csv", index=False)
top_zones.to_csv("top_enforcement_zones.csv", index=False)
df.to_csv("parking_clustered.csv", index=False)

# Score comparison: show how v2 reranks vs v1
print("\n=== SCORE COMPONENT BREAKDOWN (top 10) ===")
comp_cols = ['zone_name','c_impact','c_lanes','c_peak',
             'c_consistency','c_persistence','congestion_score']
print(top_zones[comp_cols].head(10).round(3).to_string(index=False))

print("\nSaved: cluster_stats.csv, top_enforcement_zones.csv, parking_clustered.csv")
