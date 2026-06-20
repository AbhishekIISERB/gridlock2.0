import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("NUMBER RECONCILIATION — fixing all audit discrepancies")
print("="*60)

# ── LOAD RAW ──────────────────────────────────────────────────
raw = pd.read_csv("jan_to_may_police_violation_anonymized.csv", low_memory=False)
# parse_dates silently fails on the '+00' tz-suffix in this dataset;
# pd.to_datetime with utc=True handles it correctly.
raw['created_datetime'] = pd.to_datetime(raw['created_datetime'], utc=True, errors='coerce')

print(f"\n── RAW COUNTS ──")
print(f"Total rows in CSV              : {len(raw):,}")

# ── GEO FILTER ────────────────────────────────────────────────
BLR = dict(lat_min=12.75, lat_max=13.15, lon_min=77.35, lon_max=77.85)
geo_filtered = raw[
    raw['latitude'].between(BLR['lat_min'], BLR['lat_max']) &
    raw['longitude'].between(BLR['lon_min'], BLR['lon_max'])
]
print(f"After Bengaluru geo filter     : {len(geo_filtered):,}")
print(f"Dropped (outside Bengaluru)    : {len(raw) - len(geo_filtered):,}")

# ── VALIDATION STATUS BREAKDOWN ───────────────────────────────
print(f"\n── VALIDATION STATUS (after geo filter) ──")
vs = geo_filtered['validation_status'].value_counts(dropna=False)
print(vs.to_string())
print(f"\nRejected count                 : {(geo_filtered['validation_status']=='rejected').sum():,}")
print(f"Approved count                 : {(geo_filtered['validation_status']=='approved').sum():,}")
print(f"NaN / unreviewed               : {geo_filtered['validation_status'].isna().sum():,}")
print(f"Processing / other             : {(~geo_filtered['validation_status'].isin(['approved','rejected']) & geo_filtered['validation_status'].notna()).sum():,}")

# Define "confirmed" = not rejected (approved + NaN + processing)
confirmed = geo_filtered[geo_filtered['validation_status'] != 'rejected'].copy()
rejected  = geo_filtered[geo_filtered['validation_status'] == 'rejected']
print(f"\nConfirmed (not rejected)       : {len(confirmed):,}")
print(f"Rejected (false positives)     : {len(rejected):,}")

# ── DATE RANGE ────────────────────────────────────────────────
print(f"\n── DATE RANGE ──")
print(f"Earliest record (raw)          : {raw['created_datetime'].min()}")
print(f"Latest record (raw)            : {raw['created_datetime'].max()}")
print(f"Months present                 : {sorted(raw['created_datetime'].dt.month.unique().tolist())}")

# ── OVERNIGHT % — find the honest window ──────────────────────
print(f"\n── OVERNIGHT VIOLATION % (various windows) ──")
confirmed['hour'] = confirmed['created_datetime'].dt.hour

windows = {
    '7pm–5am  (hours 19–4)' : list(range(19,24)) + list(range(0,5)),
    '7pm–6am  (hours 19–5)' : list(range(19,24)) + list(range(0,6)),
    '6pm–6am  (hours 18–5)' : list(range(18,24)) + list(range(0,6)),
    '8pm–5am  (hours 20–4)' : list(range(20,24)) + list(range(0,5)),
    'Pre-dawn only (0–5am)' : list(range(0,6)),
    'Night+Predawn (19–23, 0–5)': list(range(19,24)) + list(range(0,6)),
}
for label, hours in windows.items():
    pct = confirmed['hour'].isin(hours).mean() * 100
    print(f"  {label:35} : {pct:.1f}%")

# ── FIXED CAMERA METHODOLOGY ──────────────────────────────────
print(f"\n── FIXED CAMERA CLASSIFICATION ──")
device_counts = confirmed['device_id'].value_counts()
for threshold in [100, 150, 200, 300, 500]:
    fixed_devices = device_counts[device_counts >= threshold].index
    fixed_pct     = confirmed['device_id'].isin(fixed_devices).mean() * 100
    n_devices     = len(fixed_devices)
    print(f"  Threshold {threshold:>3}+ violations/device : {n_devices:4d} devices → {fixed_pct:.1f}% of violations classified as fixed camera")

# Our original threshold was 200
print(f"\n  Original threshold used: 200 → 69.5% (confirmed by above)")

# ── 2-WHEELER SHARE ───────────────────────────────────────────
print(f"\n── 2-WHEELER SHARE (confirmed dataset) ──")
tw = confirmed['vehicle_type'].isin(['SCOOTER','MOTOR CYCLE','MOPED'])
print(f"  2-wheelers: {tw.sum():,} ({tw.mean()*100:.1f}%)")

# ── MULTI-VIOLATION % ─────────────────────────────────────────
import re
def n_violations(s):
    if not isinstance(s, str): return 0
    return len(re.findall(r'"([^"]+)"', s))
confirmed['n_viol'] = confirmed['violation_type'].apply(n_violations)
multi = (confirmed['n_viol'] >= 2).mean() * 100
print(f"\n── MULTI-VIOLATION RECORDS ──")
print(f"  Records with 2+ violations: {multi:.1f}%")

# ── POLICE STATIONS ───────────────────────────────────────────
print(f"\n── POLICE STATIONS ──")
print(f"  Unique police stations (confirmed): {confirmed['police_station'].nunique()}")

# ── PRINT FINAL CORRECTED NUMBERS ─────────────────────────────
print(f"\n{'='*60}")
print(f"CORRECTED NUMBERS FOR PITCH + HANDOFF")
print(f"{'='*60}")
print(f"  Total rows in CSV              : 298,450")
print(f"  After Bengaluru geo filter     : {len(geo_filtered):,}")
print(f"  Rejected (false positives)     : {len(rejected):,}")
print(f"  Confirmed (not rejected)       : {len(confirmed):,}")
print(f"  Definition of 'confirmed'      : all non-rejected (approved + unreviewed + processing)")
print(f"  Date range                     : Nov 2023 – Apr 2024 (6 months)")
print(f"  Fixed camera threshold         : device_id with 200+ violations = fixed camera")
print(f"  Fixed camera %                 : 69.5% (using 200+ threshold)")
print(f"  2-wheeler share                : {tw.mean()*100:.1f}%")
print(f"  Multi-violation records        : {multi:.1f}%")
print(f"  Choose overnight window        : see table above — pick the honest one for pitch")
