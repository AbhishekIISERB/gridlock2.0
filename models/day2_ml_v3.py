import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
import ast, re, warnings
warnings.filterwarnings('ignore')

EARTH_R = 6371000
def m2r(m): return m / EARTH_R

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA (not the cleaned version — we need all original columns)
# ══════════════════════════════════════════════════════════════════════════
print("Loading raw dataset...")
raw = pd.read_csv("jan_to_may_police_violation_anonymized.csv", low_memory=False)
# parse_dates silently fails on the '+00' tz-suffix in this dataset;
# pd.to_datetime with utc=True handles it correctly.
for col in ['created_datetime', 'modified_datetime', 'validation_timestamp']:
    if col in raw.columns:
        raw[col] = pd.to_datetime(raw[col], utc=True, errors='coerce')
print(f"Raw shape: {raw.shape}")

# Bengaluru bounding box filter
BLR = dict(lat_min=12.75, lat_max=13.15, lon_min=77.35, lon_max=77.85)
raw = raw[
    raw['latitude'].between(BLR['lat_min'], BLR['lat_max']) &
    raw['longitude'].between(BLR['lon_min'], BLR['lon_max'])
]
print(f"After geo filter: {len(raw):,}")

# ══════════════════════════════════════════════════════════════════════════
# 2. USE CORRECTED VEHICLE TYPE WHERE AVAILABLE (research report insight)
# ══════════════════════════════════════════════════════════════════════════
# updated_vehicle_type is officer-verified — more accurate than original
raw['final_vehicle_type'] = raw['updated_vehicle_type'].fillna(raw['vehicle_type'])
print(f"Records with corrected vehicle type: {raw['updated_vehicle_type'].notna().sum():,} "
      f"({raw['updated_vehicle_type'].notna().mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# 3. VALIDATION STATUS WEIGHT (research report insight)
# approved = confirmed by officer = higher confidence
# rejected = false positive = should be downweighted
# NaN / processing = uncertain
# ══════════════════════════════════════════════════════════════════════════
validation_weight = {
    'approved':    1.5,   # confirmed violation — upweight
    'rejected':    0.0,   # false positive — exclude
    'processing':  0.8,
    'duplicate':   0.3,
}
raw['validation_weight'] = raw['validation_status'].map(validation_weight).fillna(1.0)

# Drop rejected violations entirely — they are false positives
before = len(raw)
raw = raw[raw['validation_weight'] > 0].copy()
print(f"After dropping rejected violations: {len(raw):,} (removed {before-len(raw):,} false positives)")

# ══════════════════════════════════════════════════════════════════════════
# 4. SCITA FLAG — officially significant violations
# ══════════════════════════════════════════════════════════════════════════
# data_sent_to_scita=True means BTP flagged this to the city traffic system
raw['scita_weight'] = raw['data_sent_to_scita'].map({True: 1.2, False: 0.9})

# ══════════════════════════════════════════════════════════════════════════
# 5. VIOLATION TYPE SEVERITY (explode multi-label strings)
# ══════════════════════════════════════════════════════════════════════════
def extract_violations(s):
    if not isinstance(s, str): return []
    return re.findall(r'"([^"]+)"', s)

vtype_severity = {
    'PARKING IN A MAIN ROAD':                    3.0,
    'DOUBLE PARKING':                            3.0,
    'PARKING OPPOSITE TO ANOTHER PARKED VEHICLE':2.5,
    'PARKING NEAR ROAD CROSSING':                2.2,
    'PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS': 2.2,
    'PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC':   2.0,
    'NO PARKING':                                2.0,
    'PARKING OTHER THAN BUS STOP':               1.8,
    'H T V PROHIBITED':                          1.8,
    'WRONG PARKING':                             1.5,
    'PARKING ON FOOTPATH':                       1.2,
}

def violation_severity(vtype_str):
    viols = extract_violations(vtype_str)
    if not viols: return 1.0
    return max(vtype_severity.get(v, 1.0) for v in viols)

def violation_count_score(vtype_str):
    """Multi-violation records are more severe"""
    n = len(extract_violations(vtype_str))
    return 1.0 + (n - 1) * 0.3  # each extra violation adds 30% weight

raw['vtype_severity']   = raw['violation_type'].apply(violation_severity)
raw['compound_score']   = raw['violation_type'].apply(violation_count_score)

# ══════════════════════════════════════════════════════════════════════════
# 6. VEHICLE LANE BLOCK (use corrected vehicle type)
# ══════════════════════════════════════════════════════════════════════════
vehicle_lane = {
    'SCOOTER':0.3,'MOTOR CYCLE':0.3,'MOPED':0.3,
    'CAR':1.0,'JEEP':1.0,
    'PASSENGER AUTO':0.8,'GOODS AUTO':0.8,
    'VAN':1.5,'TEMPO':1.5,'LGV':1.5,'MAXI-CAB':1.8,
    'PRIVATE BUS':2.5,'BUS (BMTC/KSRTC)':2.5,
    'LORRY/GOODS VEHICLE':2.5,'HGV':2.5,'TANKER':3.0,
}
raw['lanes_blocked'] = raw['final_vehicle_type'].map(vehicle_lane).fillna(1.0)

# ══════════════════════════════════════════════════════════════════════════
# 7. DEVICE_ID PATTERN — static camera vs patrol (research report insight)
# ══════════════════════════════════════════════════════════════════════════
# A device recording 500+ violations is a fixed camera, not a patrol
device_counts = raw['device_id'].value_counts()
raw['device_type'] = raw['device_id'].map(
    lambda d: 'fixed_camera' if device_counts.get(d, 0) >= 200 else 'patrol'
)
fixed_pct = (raw['device_type'] == 'fixed_camera').mean() * 100
print(f"Fixed camera violations : {fixed_pct:.1f}%")
print(f"Patrol violations       : {100-fixed_pct:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# 8. COMPOSITE PER-VIOLATION IMPACT SCORE
# ══════════════════════════════════════════════════════════════════════════
raw['impact'] = (
    raw['vtype_severity'] *
    raw['lanes_blocked'] *
    raw['compound_score'] *
    raw['validation_weight'] *
    raw['scita_weight']
)

# ══════════════════════════════════════════════════════════════════════════
# 9. TEMPORAL FEATURES
# ══════════════════════════════════════════════════════════════════════════
# Drop rows where datetime parsing failed (NaT) — they can't contribute temporal features
nat_count = raw['created_datetime'].isna().sum()
if nat_count:
    print(f"Dropping {nat_count:,} rows with unparseable created_datetime")
    raw = raw[raw['created_datetime'].notna()].copy()

raw['hour']     = raw['created_datetime'].dt.hour
raw['dow']      = raw['created_datetime'].dt.day_name()
raw['month']    = raw['created_datetime'].dt.month
raw['date']     = raw['created_datetime'].dt.date
raw['week']     = raw['created_datetime'].dt.isocalendar().week.astype('Int64')
raw['is_peak']  = raw['hour'].isin(list(range(0,6)) + list(range(19,24)))
raw['is_weekend'] = raw['created_datetime'].dt.dayofweek >= 5

print(f"\nFinal feature-engineered dataset: {len(raw):,} rows")

# ══════════════════════════════════════════════════════════════════════════
# 10. DBSCAN CLUSTERING
# ══════════════════════════════════════════════════════════════════════════
SAMPLE_N    = 60000
EPS_M       = 150
MIN_SAMPLES = 20

sample     = raw.sample(n=SAMPLE_N, random_state=42).copy()
coords_rad = np.radians(sample[['latitude','longitude']].values)

print(f"\nRunning DBSCAN (eps={EPS_M}m, min_samples={MIN_SAMPLES})...")
db = DBSCAN(eps=m2r(EPS_M), min_samples=MIN_SAMPLES,
            algorithm='ball_tree', metric='haversine')
sample['cluster'] = db.fit_predict(coords_rad)

n_clusters = len(set(sample['cluster'])) - (1 if -1 in sample['cluster'].values else 0)
print(f"Clusters: {n_clusters}")

centroids  = (sample[sample['cluster'] != -1]
              .groupby('cluster')[['latitude','longitude']].mean().reset_index())
tree       = BallTree(np.radians(centroids[['latitude','longitude']].values),
                      metric='haversine')
dist, idx  = tree.query(np.radians(raw[['latitude','longitude']].values), k=1)
dist_m     = dist.flatten() * EARTH_R

raw['cluster']          = np.where(dist_m <= 400,
                                    centroids['cluster'].values[idx.flatten()], -1)
raw['dist_centroid_m']  = dist_m.round(1)
assigned = (raw['cluster'] != -1).sum()
print(f"Assigned: {assigned:,} ({assigned/len(raw)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# 11. CLUSTER STATS — all new components
# ══════════════════════════════════════════════════════════════════════════
clustered = raw[raw['cluster'] != -1].copy()

def top_junction_clean(x):
    named = x[x.str.strip() != 'No Junction']
    if len(named) > 0:
        raw_val = named.value_counts().index[0]
        return raw_val.split(' - ')[-1] if ' - ' in raw_val else raw_val
    return None

def top_val(x): return x.value_counts().index[0]

agg = clustered.groupby('cluster').agg(
    violation_count       = ('id',                'count'),
    lat                   = ('latitude',          'mean'),
    lon                   = ('longitude',         'mean'),
    # v3 improvements
    total_impact          = ('impact',            'sum'),     # NEW: includes validation+scita weights
    avg_impact            = ('impact',            'mean'),
    avg_lanes_blocked     = ('lanes_blocked',     'mean'),
    peak_pct              = ('is_peak',           'mean'),
    weekend_pct           = ('is_weekend',        'mean'),    # NEW
    active_days           = ('date',              'nunique'),
    active_weeks          = ('week',              'nunique'),
    approved_pct          = ('validation_weight', lambda x: (x==1.5).mean()),  # NEW
    scita_pct             = ('scita_weight',      lambda x: (x==1.2).mean()),  # NEW
    fixed_camera_pct      = ('device_type',       lambda x: (x=='fixed_camera').mean()),  # NEW
    compound_pct          = ('compound_score',    lambda x: (x>1.0).mean()),   # NEW
    junction_label        = ('junction_name',     top_junction_clean),
    police_station        = ('police_station',    top_val),
    top_vehicle           = ('final_vehicle_type',top_val),   # uses corrected type
).reset_index()

agg['zone_name'] = agg.apply(
    lambda r: r['junction_label'] if pd.notna(r['junction_label'])
              else f"{r['police_station']} area", axis=1)

# ══════════════════════════════════════════════════════════════════════════
# 12. V3 CONGESTION SCORE — 7 components
# ══════════════════════════════════════════════════════════════════════════
def mm(s):
    rng = s.max() - s.min()
    return (s - s.min()) / (rng if rng > 0 else 1)

# Component 1: Validated impact (total_impact uses validation + scita weights)
agg['c_impact']      = mm(agg['total_impact'])

# Component 2: Vehicle severity
agg['c_lanes']       = mm(agg['avg_lanes_blocked'])

# Component 3: Peak timing
agg['c_peak']        = mm(agg['peak_pct'])

# Component 4: Chronicity (active days)
agg['c_consistency'] = mm(agg['active_days'])

# Component 5: Persistence (active weeks)
agg['c_persistence'] = mm(agg['active_weeks'])

# Component 6 NEW: Official confirmation rate (approved + SCITA)
agg['c_official']    = mm(agg['approved_pct'] * 0.6 + agg['scita_pct'] * 0.4)

# Component 7 NEW: Compound violation rate (multi-offense records)
agg['c_compound']    = mm(agg['compound_pct'])

WEIGHTS = {
    'c_impact':      0.30,
    'c_lanes':       0.15,
    'c_peak':        0.15,
    'c_consistency': 0.15,
    'c_persistence': 0.10,
    'c_official':    0.10,   # NEW
    'c_compound':    0.05,   # NEW
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"

agg['raw_score']        = sum(agg[k]*w for k,w in WEIGHTS.items())
agg['congestion_score'] = (mm(agg['raw_score']) * 100).round(1)
agg['blockage_pct']     = ((agg['avg_lanes_blocked'] / 4) * 100).round(1)
agg['daily_rate']       = (agg['violation_count'] / agg['active_days']).round(1)

def risk_tier(s):
    if s >= 75: return 'CRITICAL'
    if s >= 50: return 'HIGH'
    if s >= 25: return 'MEDIUM'
    return 'LOW'

def shift_label(p):
    if p > 0.85: return 'Late night (11pm–4am)'
    if p > 0.60: return 'Evening + night (7pm–midnight)'
    return 'Mixed / daytime'

agg['risk_tier']         = agg['congestion_score'].apply(risk_tier)
agg['enforcement_shift'] = agg['peak_pct'].apply(shift_label)

# Weekend bias flag — zones worse on weekends need different roster
agg['weekend_bias'] = (agg['weekend_pct'] > 0.55).map({True:'Weekend-heavy', False:'Weekday-uniform'})

# ══════════════════════════════════════════════════════════════════════════
# 13. TOP ZONES + OUTPUT
# ══════════════════════════════════════════════════════════════════════════
top_zones = agg.nlargest(20, 'congestion_score').copy()
top_zones['rank'] = range(1, len(top_zones)+1)
top_zones['peak_pct_pct'] = (top_zones['peak_pct']*100).round(1)

print("\n=== TOP 15 ENFORCEMENT ZONES (v3) ===")
cols = ['rank','zone_name','police_station','violation_count','daily_rate',
        'congestion_score','blockage_pct','risk_tier','enforcement_shift','weekend_bias']
pd.set_option('display.max_colwidth', 30)
pd.set_option('display.width', 160)
print(top_zones[cols].head(15).to_string(index=False))

print("\n=== SCORE COMPONENT BREAKDOWN (top 10) ===")
comp_cols = ['zone_name','c_impact','c_lanes','c_peak','c_consistency',
             'c_persistence','c_official','c_compound','congestion_score']
print(top_zones[comp_cols].head(10).round(3).to_string(index=False))

top10 = agg.nlargest(10, 'congestion_score')
print("\n=== PITCH KPIs (v3) ===")
print(f"Total violations (after removing rejected) : {len(raw):,}")
print(f"False positives removed (rejected)         : {before - len(raw):,}")
print(f"Clusters identified                         : {n_clusters}")
print(f"CRITICAL zones                              : {(agg['risk_tier']=='CRITICAL').sum()}")
print(f"HIGH risk zones                             : {(agg['risk_tier']=='HIGH').sum()}")
print(f"Violations in clusters                      : {assigned:,} ({assigned/len(raw)*100:.1f}%)")
print(f"Overnight violations (7pm–5am)              : {raw['is_peak'].sum():,} ({raw['is_peak'].mean()*100:.1f}%)")
print(f"Avg blockage % (top 10)                     : {top10['blockage_pct'].mean():.1f}%")
print(f"Fixed-camera vs patrol split                : {fixed_pct:.1f}% / {100-fixed_pct:.1f}%")
print(f"Violations sent to SCITA                    : {(raw['data_sent_to_scita']==True).sum():,} ({(raw['data_sent_to_scita']==True).mean()*100:.1f}%)")
print(f"#1 zone                                     : {top_zones.iloc[0]['zone_name']}")
print(f"#1 score                                    : {top_zones.iloc[0]['congestion_score']}/100")

agg.to_csv("cluster_stats.csv", index=False)
top_zones.to_csv("top_enforcement_zones.csv", index=False)
raw.to_csv("parking_clustered.csv", index=False)
print("\nSaved all outputs.")
