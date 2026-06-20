"""
day3_deployment.py  —  Gridlock Hackathon 2.0
=============================================
Prescriptive deployment recommendation engine.

Input  : cluster_stats.csv (v5, from day2_ml_v5_features.py)
Output : deployment_plan.csv  — one row per zone, fully prescribed

Logic
-----
Each zone gets:
  1. patrol_type    — Static post / Mobile patrol / Tow unit / Camera monitor
  2. officer_count  — based on tier + zone type
  3. shift_window   — derived from enforcement_shift
  4. expected_impact — estimated % blockage reduction if deployed
  5. deployment_priority — final integer rank for commander

Rules (evidence-based from the dataset findings)
-------------------------------------------------
CRITICAL + point zone         → 2-officer static post + tow unit on call
CRITICAL + corridor zone      → mobile patrol (2 officers, continuous circuit)
HIGH     + metro/event        → 1-officer static post (peak hours only)
HIGH     + general            → mobile patrol (1 officer)
MEDIUM                        → camera monitor (flag for reactive response)
LOW                           → no deployment recommended

Corridor detection: zones where zone_name contains 'Road' / 'Main' / 'Cross'
OR where the cluster spans >400m (proxy: avg_lanes_blocked < 1.5 AND violation_count > 200)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── LOAD ──────────────────────────────────────────────────────
print("Loading cluster_stats.csv ...")
cs = pd.read_csv("cluster_stats.csv")
print(f"  {len(cs)} clusters | v5 score present: {'congestion_score_v5' in cs.columns}")

if 'congestion_score_v5' not in cs.columns:
    print("ERROR: Run day2_ml_v5_features.py first.")
    exit(1)

# ── CORRIDOR DETECTION ────────────────────────────────────────
corridor_keywords = ['road','main','cross','street','avenue','highway','layout','nagar']

def is_corridor(row):
    name_lower = str(row.get('zone_name', '')).lower()
    if any(k in name_lower for k in corridor_keywords):
        return True
    # Proxy: spread-out zone with many violations but low avg lanes blocked
    if row.get('avg_lanes_blocked', 2) < 1.5 and row.get('violation_count', 0) > 200:
        return True
    return False

cs['is_corridor'] = cs.apply(is_corridor, axis=1)

# ── SHIFT WINDOW MAPPING ──────────────────────────────────────
def shift_to_window(shift_label):
    if 'Late night' in str(shift_label):
        return '11:00 PM – 5:00 AM'
    if 'Evening' in str(shift_label):
        return '7:00 PM – 1:00 AM'
    return '6:00 AM – 10:00 AM, 5:00 PM – 9:00 PM'

# ── DEPLOYMENT RULES ──────────────────────────────────────────
def assign_deployment(row):
    tier          = row.get('risk_tier_v5', row.get('risk_tier', 'LOW'))
    is_corr       = row['is_corridor']
    near_metro    = row.get('near_metro', False)
    near_event    = row.get('near_event_venue', False)
    daily_rate    = row.get('daily_rate', 0)
    blockage      = row.get('blockage_pct', 0)
    shift         = row.get('enforcement_shift', 'Mixed / daytime')

    # ── CRITICAL ──────────────────────────────────────────────
    if tier == 'CRITICAL':
        if is_corr:
            return dict(
                patrol_type      = 'Mobile patrol',
                officer_count    = 3,
                tow_unit         = True,
                shift_window     = shift_to_window(shift),
                expected_impact  = '35–50% blockage reduction',
                deployment_note  = 'Continuous circuit; coordinate with tow unit at corridor entry points',
            )
        else:
            return dict(
                patrol_type      = 'Static post + Tow unit',
                officer_count    = 2,
                tow_unit         = True,
                shift_window     = shift_to_window(shift),
                expected_impact  = '50–65% blockage reduction',
                deployment_note  = 'Static post at junction; tow unit on 15-min standby',
            )

    # ── HIGH ──────────────────────────────────────────────────
    if tier == 'HIGH':
        if near_metro or near_event:
            return dict(
                patrol_type      = 'Static post (peak only)',
                officer_count    = 2 if daily_rate > 30 else 1,
                tow_unit         = daily_rate > 40,
                shift_window     = shift_to_window(shift),
                expected_impact  = '30–45% blockage reduction',
                deployment_note  = 'Deploy at metro exit / venue gate; peak-hours only to maximise coverage',
            )
        elif is_corr:
            return dict(
                patrol_type      = 'Mobile patrol',
                officer_count    = 2,
                tow_unit         = False,
                shift_window     = shift_to_window(shift),
                expected_impact  = '25–40% blockage reduction',
                deployment_note  = 'Circuit patrol covering full corridor; radio coordination with adjacent zones',
            )
        else:
            return dict(
                patrol_type      = 'Mobile patrol',
                officer_count    = 1,
                tow_unit         = False,
                shift_window     = shift_to_window(shift),
                expected_impact  = '20–35% blockage reduction',
                deployment_note  = 'Single-officer mobile; escalate to static if violations exceed 20/night',
            )

    # ── MEDIUM ────────────────────────────────────────────────
    if tier == 'MEDIUM':
        return dict(
            patrol_type      = 'Camera monitor',
            officer_count    = 0,
            tow_unit         = False,
            shift_window     = 'Reactive (on alert)',
            expected_impact  = '10–20% blockage reduction',
            deployment_note  = 'Flag for reactive dispatch; review monthly for tier upgrade',
        )

    # ── LOW ───────────────────────────────────────────────────
    return dict(
        patrol_type      = 'No deployment',
        officer_count    = 0,
        tow_unit         = False,
        shift_window     = '—',
        expected_impact  = 'Negligible',
        deployment_note  = 'Monitor via camera only',
    )

# ── APPLY DEPLOYMENT RULES ────────────────────────────────────
print("Assigning deployment plans ...")
deploy_cols = cs.apply(assign_deployment, axis=1, result_type='expand')
cs = pd.concat([cs, deploy_cols], axis=1)

# ── PRIORITY SCORE ────────────────────────────────────────────
# Rank: CRITICAL first, then by congestion_score_v5 desc
tier_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
cs['tier_order'] = cs['risk_tier_v5'].map(tier_order).fillna(3)
cs_sorted = cs.sort_values(['tier_order', 'congestion_score_v5'],
                           ascending=[True, False]).reset_index(drop=True)
cs_sorted['deployment_priority'] = cs_sorted.index + 1

# ── SUMMARY STATS ─────────────────────────────────────────────
print("\n=== DEPLOYMENT SUMMARY ===")
patrol_counts = cs_sorted['patrol_type'].value_counts()
for pt, cnt in patrol_counts.items():
    print(f"  {pt:<30} : {cnt} zones")

total_officers = cs_sorted['officer_count'].sum()
tow_zones      = cs_sorted['tow_unit'].sum()
print(f"\n  Total officers needed (all shifts)  : {total_officers}")
print(f"  Zones requiring tow unit            : {tow_zones}")
print(f"  Metro-context deployments           : {(cs_sorted['near_metro'] & cs_sorted['officer_count'] > 0).sum()}")
print(f"  Event-context deployments           : {(cs_sorted['near_event_venue'] & cs_sorted['officer_count'] > 0).sum()}")

# ── TOP 15 DEPLOYMENT TABLE ───────────────────────────────────
top15 = cs_sorted[cs_sorted['tier_order'] <= 1].head(15)
print("\n=== TOP 15 DEPLOYMENT PLAN ===")
disp_cols = ['deployment_priority','zone_name','risk_tier_v5','patrol_type',
             'officer_count','tow_unit','shift_window','context_tags']
pd.set_option('display.max_colwidth', 28)
pd.set_option('display.width', 200)
print(top15[disp_cols].to_string(index=False))

# ── SAVE ──────────────────────────────────────────────────────
output_cols = [
    'deployment_priority','zone_name','police_station',
    'risk_tier_v5','congestion_score_v5','daily_rate','blockage_pct',
    'patrol_type','officer_count','tow_unit','shift_window',
    'expected_impact','deployment_note',
    'context_tags','near_metro','near_event_venue',
    'metro_dist_m','event_dist_m','enforcement_shift',
    'is_corridor','cluster_lat','cluster_lon',
]
output_cols = [c for c in output_cols if c in cs_sorted.columns]
cs_sorted[output_cols].to_csv("deployment_plan.csv", index=False)
print("\nSaved → deployment_plan.csv")
print("Run day3_streamlit_v2.py next:  streamlit run day3_streamlit_v2.py")
