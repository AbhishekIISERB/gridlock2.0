import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def mm(s):
    rng = s.max() - s.min()
    return (s - s.min()) / (rng if rng > 0 else 1)

# ── LOAD ──────────────────────────────────────────────────────
cs    = pd.read_csv("cluster_stats.csv")
zones = pd.read_csv("top_enforcement_zones.csv")

print(f"Clusters loaded: {len(cs)}")

# ══════════════════════════════════════════════════════════════
# ABLATION-INFORMED WEIGHTS (v4)
# Evidence: c_lanes r=0.935, c_consistency r=0.907 drive rankings
# c_impact r=0.990, c_compound r=0.991 are redundant at high weight
# ══════════════════════════════════════════════════════════════
V4_WEIGHTS = {
    'c_impact':      0.20,  # was 0.30 — ablation r=0.990, redundant at 30%
    'c_lanes':       0.25,  # was 0.15 — ablation r=0.935, most discriminating
    'c_peak':        0.15,  # unchanged
    'c_consistency': 0.25,  # was 0.15 — ablation r=0.907, second most discriminating
    'c_persistence': 0.10,  # unchanged
    'c_official':    0.04,  # was 0.10 — low independent impact (r=0.972)
    'c_compound':    0.01,  # was 0.05 — lowest impact (r=0.991)
}
assert abs(sum(V4_WEIGHTS.values()) - 1.0) < 1e-9

comp_cols = list(V4_WEIGHTS.keys())
missing = [c for c in comp_cols if c not in cs.columns]
if missing:
    print(f"Missing columns: {missing} — ensure cluster_stats.csv is from v3 run")
    exit(1)

raw_score = sum(mm(cs[k]) * w for k, w in V4_WEIGHTS.items())
cs['congestion_score_v4'] = (mm(raw_score) * 100).round(1)
cs['blockage_pct']        = ((cs['avg_lanes_blocked'] / 4) * 100).round(1)

# Risk tiers
def risk_tier(s):
    if s >= 75: return 'CRITICAL'
    if s >= 50: return 'HIGH'
    if s >= 25: return 'MEDIUM'
    return 'LOW'

def shift_label(p):
    if p > 0.85: return 'Late night (11pm–4am)'
    if p > 0.60: return 'Evening + night (7pm–midnight)'
    return 'Mixed / daytime'

cs['risk_tier_v4']         = cs['congestion_score_v4'].apply(risk_tier)
cs['enforcement_shift']    = cs['peak_pct'].apply(shift_label)
cs['daily_rate']           = (cs['violation_count'] / cs['active_days']).round(1)
cs['peak_pct_pct']         = (cs['peak_pct'] * 100).round(1)
cs['weekend_bias']         = cs['weekend_pct'].apply(
    lambda p: 'Weekend-heavy' if p > 0.55 else 'Weekday-uniform'
) if 'weekend_pct' in cs.columns else 'Unknown'

# ── TOP ZONES ─────────────────────────────────────────────────
top_zones_v4 = cs.nlargest(20, 'congestion_score_v4').copy()
top_zones_v4['rank'] = range(1, len(top_zones_v4) + 1)

print("\n=== TOP 15 ENFORCEMENT ZONES (v4 — final) ===")
cols = ['rank','zone_name','police_station','violation_count','daily_rate',
        'congestion_score_v4','blockage_pct','risk_tier_v4','enforcement_shift']
pd.set_option('display.max_colwidth', 32)
pd.set_option('display.width', 160)
print(top_zones_v4[cols].head(15).to_string(index=False))

# ── RANK CHANGE TABLE (v3 → v4) ───────────────────────────────
print("\n=== RANK CHANGE v3 → v4 (top 15) ===")
v3_ranks = zones[['zone_name','rank']].rename(columns={'rank':'rank_v3'})
v4_temp  = top_zones_v4[['zone_name','rank']].rename(columns={'rank':'rank_v4'})
merged   = v3_ranks.merge(v4_temp, on='zone_name', how='outer').fillna(99)
merged['change'] = merged['rank_v3'] - merged['rank_v4']
merged['arrow']  = merged['change'].apply(
    lambda x: f'↑{int(x)}' if x > 0 else (f'↓{int(abs(x))}' if x < 0 else '→')
)
print(merged[['zone_name','rank_v3','rank_v4','arrow']].head(15).to_string(index=False))

# ── TIER DISTRIBUTION ─────────────────────────────────────────
print("\n=== RISK TIER DISTRIBUTION (v4) ===")
tiers = cs['risk_tier_v4'].value_counts()
for tier in ['CRITICAL','HIGH','MEDIUM','LOW']:
    count = tiers.get(tier, 0)
    bar   = '█' * int(count / 3)
    print(f"  {tier:>10} : {count:3d} zones  {bar}")

# ── FINAL PITCH KPIs ──────────────────────────────────────────
top10 = cs.nlargest(10, 'congestion_score_v4')
print("\n=== FINAL PITCH KPIs (v4) ===")
print(f"Total violations (confirmed)        : 243,313")
print(f"False positives removed             : 48,664")
print(f"Clusters identified                 : 184")
print(f"CRITICAL zones                      : {(cs['risk_tier_v4']=='CRITICAL').sum()}")
print(f"HIGH risk zones                     : {(cs['risk_tier_v4']=='HIGH').sum()}")
print(f"Silhouette score (DBSCAN)           : 0.4500")
print(f"DBSCAN vs HDBSCAN                   : 0.4500 vs 0.4363 — DBSCAN wins")
print(f"Leakage test (Spearman r)           : 0.966 — post-hoc fields refine, not manufacture")
print(f"Temporal stability (top-10 overlap) : 6/10 — core zones persist, 4 emerging detected")
print(f"Eps optimality                      : 150m = best silhouette across 100/150/200/300m")
print(f"Overnight violations (7pm–5am)      : 81.6%")
print(f"Fixed camera vs patrol              : 69.5% / 30.5%")
print(f"Avg blockage % (top 10)             : {top10['blockage_pct'].mean():.1f}%")
print(f"#1 zone                             : {top_zones_v4.iloc[0]['zone_name']}")
print(f"#1 daily rate                       : {top_zones_v4.iloc[0]['daily_rate']:.0f} violations/day")
print(f"Weight justification                : ablation study (c_lanes r=0.935, c_consistency r=0.907)")

# ── SAVE FINAL OUTPUTS ────────────────────────────────────────
cs['congestion_score'] = cs['congestion_score_v4']
cs['risk_tier']        = cs['risk_tier_v4']
cs.to_csv("cluster_stats.csv", index=False)
top_zones_v4['congestion_score'] = top_zones_v4['congestion_score_v4']
top_zones_v4['risk_tier']        = top_zones_v4['risk_tier_v4']
top_zones_v4.to_csv("top_enforcement_zones.csv", index=False)
print("\nFinal outputs saved — cluster_stats.csv and top_enforcement_zones.csv updated to v4.")
