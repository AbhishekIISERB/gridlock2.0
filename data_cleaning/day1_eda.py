import pandas as pd
import numpy as np

# ── 1. LOAD ────────────────────────────────────────────────────────────────
df = pd.read_csv("jan_to_may_police_violation_anonymized.csv")
print(f"Loaded: {df.shape}")

# ── 2. FILTER TO PARKING VIOLATIONS ───────────────────────────────────────
parking_keywords = ['park', 'parking', 'obstruct', 'illegal stop', 'no parking',
                    'footpath', 'carriageway', 'blocking']
mask = df['violation_type'].str.lower().str.contains(
    '|'.join(parking_keywords), na=False
)
park = df[mask].copy()
print(f"\nParking violations: {len(park)} of {len(df)} ({len(park)/len(df)*100:.1f}%)")
print("\nTop 20 parking violation types:")
print(park['violation_type'].value_counts().head(20))

# ── 3. GEO VALIDATION ─────────────────────────────────────────────────────
# Bengaluru bounding box
BLR = dict(lat_min=12.75, lat_max=13.15, lon_min=77.35, lon_max=77.85)
before = len(park)
park = park[
    park['latitude'].between(BLR['lat_min'], BLR['lat_max']) &
    park['longitude'].between(BLR['lon_min'], BLR['lon_max'])
]
print(f"\nAfter geo filter: {len(park)} (dropped {before - len(park)} out-of-bounds)")

# ── 4. DATETIME FEATURES ──────────────────────────────────────────────────
park['created_datetime'] = pd.to_datetime(park['created_datetime'], errors='coerce')
park = park.dropna(subset=['created_datetime'])

park['hour']        = park['created_datetime'].dt.hour
park['day_of_week'] = park['created_datetime'].dt.day_name()
park['month']       = park['created_datetime'].dt.month
park['date']        = park['created_datetime'].dt.date
park['is_peak']     = park['hour'].isin(range(8, 12)) | park['hour'].isin(range(17, 21))
park['is_weekend']  = park['created_datetime'].dt.dayofweek >= 5

print(f"\nDate range: {park['created_datetime'].min()} → {park['created_datetime'].max()}")
print(f"Peak hour violations: {park['is_peak'].sum()} ({park['is_peak'].mean()*100:.1f}%)")

# ── 5. VEHICLE TYPE SEVERITY WEIGHT ───────────────────────────────────────
print("\nVehicle types in parking violations:")
print(park['vehicle_type'].value_counts().head(15))

severity_map = {
    'TRUCK': 3.0, 'BUS': 3.0, 'LMV': 2.0, 'HMV': 3.0,
    'MAXI-CAB': 2.5, 'TAXI': 1.5, 'AUTO': 1.2,
    'MOTOR CYCLE': 0.5, 'TWO WHEELER': 0.5,
    'CAR': 1.5, 'JEEP': 1.5, 'VAN': 2.0,
}
park['severity_weight'] = park['vehicle_type'].map(severity_map).fillna(1.0)

# ── 6. JUNCTION / LOCATION QUALITY ────────────────────────────────────────
park['has_junction'] = park['junction_name'].notna()
print(f"\nViolations with known junction: {park['has_junction'].sum()} "
      f"({park['has_junction'].mean()*100:.1f}%)")
print("\nTop 20 junctions by violation count:")
print(park['junction_name'].value_counts().head(20))

# ── 7. POLICE STATION COVERAGE ────────────────────────────────────────────
print("\nTop 15 police stations by parking violations:")
print(park['police_station'].value_counts().head(15))

# ── 8. HOURLY + DAILY PATTERN ─────────────────────────────────────────────
print("\nViolations by hour (top 5):")
print(park['hour'].value_counts().sort_index())

print("\nViolations by day of week:")
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
print(park['day_of_week'].value_counts().reindex(dow_order))

# ── 9. SAVE CLEAN FILE ────────────────────────────────────────────────────
keep_cols = [
    'id', 'latitude', 'longitude', 'location', 'vehicle_type', 'violation_type',
    'offence_code', 'created_datetime', 'police_station', 'junction_name',
    'hour', 'day_of_week', 'month', 'date', 'is_peak', 'is_weekend',
    'severity_weight', 'has_junction'
]
park_clean = park[keep_cols].reset_index(drop=True)
park_clean.to_csv("parking_clean.csv", index=False)
print(f"\nSaved parking_clean.csv → {len(park_clean)} rows, {len(park_clean.columns)} cols")
print("\nColumn summary:")
print(park_clean.dtypes)
