import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("parking_clustered.csv", parse_dates=['created_datetime'])
cs = pd.read_csv("cluster_stats.csv")
tz = pd.read_csv("top_enforcement_zones.csv")

print("="*60)
print("DEEP EDA — FULL DATASET NUANCE DISCOVERY")
print("="*60)

# ══════════════════════════════════════════════════════════════
# 1. VIOLATION TYPE DEEP DIVE
# ══════════════════════════════════════════════════════════════
print("\n── 1. VIOLATION TYPE ANALYSIS ──")

# Parse the JSON-like violation_type strings
import re
def extract_violations(s):
    if not isinstance(s, str): return []
    return re.findall(r'"([^"]+)"', s)

df['viol_list'] = df['violation_type'].apply(extract_violations)

# Explode to individual violations
exploded = df.explode('viol_list')
vtype_counts = exploded['viol_list'].value_counts()
print("\nAll unique violation types:")
print(vtype_counts.to_string())

# Multi-violation records
df['viol_count'] = df['viol_list'].apply(len)
print(f"\nRecords with 1 violation   : {(df['viol_count']==1).sum():,} ({(df['viol_count']==1).mean()*100:.1f}%)")
print(f"Records with 2 violations  : {(df['viol_count']==2).sum():,} ({(df['viol_count']==2).mean()*100:.1f}%)")
print(f"Records with 3+ violations : {(df['viol_count']>=3).sum():,} ({(df['viol_count']>=3).mean()*100:.1f}%)")

# Most dangerous combos
combos = df[df['viol_count']>=2]['violation_type'].value_counts().head(10)
print("\nTop 10 multi-violation combos:")
print(combos.to_string())

# ══════════════════════════════════════════════════════════════
# 2. TEMPORAL NUANCES
# ══════════════════════════════════════════════════════════════
print("\n── 2. TEMPORAL NUANCES ──")

df['hour']        = df['created_datetime'].dt.hour
df['month']       = df['created_datetime'].dt.month
df['day_of_week'] = df['created_datetime'].dt.day_name()
df['date']        = df['created_datetime'].dt.date
df['week']        = df['created_datetime'].dt.isocalendar().week.astype(int)

# Monthly trend — is it getting worse?
monthly = df.groupby('month').size()
monthly_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',11:'Nov',12:'Dec'}
print("\nViolations by month:")
for m, c in monthly.items():
    print(f"  {monthly_names.get(m,m):>3} : {c:,}")

# Day of week pattern
print("\nViolations by day of week:")
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow = df['day_of_week'].value_counts().reindex(dow_order)
for d, c in dow.items():
    bar = '█' * int(c/1000)
    print(f"  {d:>12} : {c:,}  {bar}")

# Hour buckets
print("\nHour buckets:")
buckets = {
    'Late night  (12am–3am)': list(range(0,3)),
    'Pre-dawn    (3am–6am)' : list(range(3,6)),
    'Morning     (6am–9am)' : list(range(6,9)),
    'Mid-morning (9am–12pm)': list(range(9,12)),
    'Afternoon   (12pm–5pm)': list(range(12,17)),
    'Evening     (5pm–8pm)' : list(range(17,20)),
    'Night       (8pm–12am)': list(range(20,24)),
}
for label, hours in buckets.items():
    count = df[df['hour'].isin(hours)]['id'].count()
    pct   = count / len(df) * 100
    print(f"  {label} : {count:,} ({pct:.1f}%)")

# Week-over-week trend
print("\nWeekly violation counts (trend):")
weekly = df.groupby('week').size().reset_index(name='count')
weekly['trend'] = weekly['count'].diff().apply(
    lambda x: '↑' if x>500 else ('↓' if x<-500 else '→') if pd.notna(x) else ''
)
print(weekly.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 3. VEHICLE TYPE NUANCES
# ══════════════════════════════════════════════════════════════
print("\n── 3. VEHICLE TYPE NUANCES ──")

veh = df.groupby('vehicle_type').agg(
    count        = ('id','count'),
    peak_pct     = ('is_peak','mean'),
    avg_lanes    = ('lanes_blocked','mean'),
).sort_values('count', ascending=False)
veh['pct'] = (veh['count']/len(df)*100).round(1)
veh['peak_pct'] = (veh['peak_pct']*100).round(1)
print(veh.head(15).to_string())

# 2-wheelers dominate — break them down
two_wheelers = ['SCOOTER','MOTOR CYCLE','MOPED']
tw = df[df['vehicle_type'].isin(two_wheelers)]
print(f"\n2-wheelers total: {len(tw):,} ({len(tw)/len(df)*100:.1f}% of all violations)")
print(f"2-wheelers in clustered zones: {(tw['cluster']!=-1).sum():,} ({(tw['cluster']!=-1).mean()*100:.1f}%)")

# Heavy vehicles analysis
heavy = df[df['vehicle_type'].isin(['LORRY/GOODS VEHICLE','HGV','PRIVATE BUS','BUS (BMTC/KSRTC)'])]
print(f"\nHeavy vehicles: {len(heavy):,} ({len(heavy)/len(df)*100:.1f}%)")
print(f"Heavy vehicle peak hour %: {heavy['is_peak'].mean()*100:.1f}%")
print(f"Top heavy vehicle zones:")
print(heavy.groupby('police_station').size().sort_values(ascending=False).head(8).to_string())

# ══════════════════════════════════════════════════════════════
# 4. GEOGRAPHIC NUANCES
# ══════════════════════════════════════════════════════════════
print("\n── 4. GEOGRAPHIC NUANCES ──")

# Police station analysis
ps_stats = df.groupby('police_station').agg(
    count      = ('id','count'),
    peak_pct   = ('is_peak','mean'),
    unique_junctions = ('junction_name','nunique'),
    clusters   = ('cluster', lambda x: (x!=-1).mean()),
).sort_values('count', ascending=False)
ps_stats['peak_pct'] = (ps_stats['peak_pct']*100).round(1)
ps_stats['cluster_rate'] = (ps_stats['clusters']*100).round(1)
print("\nTop 20 police stations:")
print(ps_stats.head(20).to_string())

# Zone density — how many violations per sq km?
print("\n── Zone density (violations per sq km estimate) ──")
for _, row in tz.head(10).iterrows():
    area_sqkm = 3.14159 * (0.15**2)  # pi * r^2 where r=150m=0.15km
    density = row['violation_count'] / area_sqkm
    print(f"  #{int(row['rank'])} {row['zone_name'][:35]:35} : {density:,.0f} viol/km²")

# ══════════════════════════════════════════════════════════════
# 5. REPEAT OFFENDER ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n── 5. REPEAT OFFENDER ANALYSIS ──")

if 'vehicle_number' in df.columns:
    vehicle_freq = df.groupby('vehicle_number').size().reset_index(name='appearances')
    print(f"Unique vehicles         : {len(vehicle_freq):,}")
    print(f"Vehicles caught 1x      : {(vehicle_freq['appearances']==1).sum():,} ({(vehicle_freq['appearances']==1).mean()*100:.1f}%)")
    print(f"Vehicles caught 2–5x    : {vehicle_freq['appearances'].between(2,5).sum():,}")
    print(f"Vehicles caught 6–10x   : {vehicle_freq['appearances'].between(6,10).sum():,}")
    print(f"Vehicles caught 10x+    : {(vehicle_freq['appearances']>10).sum():,}")
    print(f"Max appearances         : {vehicle_freq['appearances'].max()}")

    top_offenders = vehicle_freq.nlargest(10,'appearances')
    print("\nTop 10 repeat offenders:")
    print(top_offenders.to_string(index=False))

    # Do repeat offenders cluster?
    repeat_vehicles = vehicle_freq[vehicle_freq['appearances']>=5]['vehicle_number']
    repeat_df = df[df['vehicle_number'].isin(repeat_vehicles)]
    print(f"\nRepeat offenders (5+) in clustered zones: {(repeat_df['cluster']!=-1).mean()*100:.1f}%")
    print(f"vs all violations in clustered zones   : {(df['cluster']!=-1).mean()*100:.1f}%")
    print("\nRepeat offender top zones:")
    print(repeat_df.groupby('police_station').size().sort_values(ascending=False).head(8).to_string())
else:
    print("vehicle_number column not present in dataset — skipping repeat offender analysis")

# ══════════════════════════════════════════════════════════════
# 6. JUNCTION vs MID-BLOCK SPLIT
# ══════════════════════════════════════════════════════════════
print("\n── 6. JUNCTION vs MID-BLOCK ANALYSIS ──")

df['is_junction'] = df['junction_name'].str.strip() != 'No Junction'
print(f"At named junctions : {df['is_junction'].sum():,} ({df['is_junction'].mean()*100:.1f}%)")
print(f"Mid-block          : {(~df['is_junction']).sum():,} ({(~df['is_junction']).mean()*100:.1f}%)")

# Do junction violations have different severity?
j_sev  = df[df['is_junction']]['lanes_blocked'].mean()
mb_sev = df[~df['is_junction']]['lanes_blocked'].mean()
print(f"\nAvg lanes blocked at junction  : {j_sev:.3f}")
print(f"Avg lanes blocked mid-block    : {mb_sev:.3f}")
print(f"Junction violations are {j_sev/mb_sev:.2f}x more severe than mid-block")

# Junction violations by hour
print("\nJunction violations by hour bucket:")
for label, hours in buckets.items():
    j_count  = df[df['hour'].isin(hours) & df['is_junction']]['id'].count()
    mb_count = df[df['hour'].isin(hours) & ~df['is_junction']]['id'].count()
    print(f"  {label} : junction={j_count:,}  mid-block={mb_count:,}")

# ══════════════════════════════════════════════════════════════
# 7. VALIDATION STATUS NUANCE
# ══════════════════════════════════════════════════════════════
print("\n── 7. VALIDATION STATUS ──")
if 'validation_status' in df.columns:
    print(df['validation_status'].value_counts(dropna=False).to_string())
    approved = df[df['validation_status']=='approved']
    print(f"\nApproved violations: {len(approved):,} ({len(approved)/len(df)*100:.1f}%)")
    print(f"Approved in clusters: {(approved['cluster']!=-1).mean()*100:.1f}%")
    print(f"Unapproved in clusters: {(df[df['validation_status']!='approved']['cluster']!=-1).mean()*100:.1f}%")
else:
    print("validation_status not in clustered file")

# ══════════════════════════════════════════════════════════════
# 8. DATA QUALITY FINDINGS
# ══════════════════════════════════════════════════════════════
print("\n── 8. DATA QUALITY FINDINGS ──")
print(f"Total records         : {len(df):,}")
print(f"Date range            : {df['created_datetime'].min()} → {df['created_datetime'].max()}")
print(f"Months covered        : {df['month'].nunique()} months")
print(f"Unique junctions      : {df['junction_name'].nunique()}")
print(f"Unique police stations: {df['police_station'].nunique()}")
print(f"Unique vehicle types  : {df['vehicle_type'].nunique()}")
if 'vehicle_number' in df.columns:
    print(f"Unique vehicles       : {df['vehicle_number'].nunique():,}")
else:
    print("Unique vehicles       : N/A (vehicle_number not in dataset)")

print("\n✓ Deep EDA complete")
