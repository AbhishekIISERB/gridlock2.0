import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

EARTH_R = 6371000
def m2r(m): return m / EARTH_R

df       = pd.read_csv("parking_clustered.csv", parse_dates=['created_datetime'])
clusters = pd.read_csv("cluster_stats.csv")
zones    = pd.read_csv("top_enforcement_zones.csv")

print("="*60)
print("EXTENDED VALIDATION v2 — addressing research report concerns")
print("="*60)

# ══════════════════════════════════════════════════════════════
# 1. LEAKAGE DEFENCE — score with vs without post-hoc fields
# ══════════════════════════════════════════════════════════════
print("\n── 1. LEAKAGE SENSITIVITY TEST ──")
print("Does removing post-hoc fields (validation, SCITA) change rankings?")

# Score without official confirmation (c_official removed)
def mm(s):
    rng = s.max() - s.min()
    return (s - s.min()) / (rng if rng > 0 else 1)

# v3 score (full)
v3_score = clusters['congestion_score'].copy()

# Score without c_official and c_compound (pre-hoc only)
if all(c in clusters.columns for c in ['c_impact','c_lanes','c_peak','c_consistency','c_persistence']):
    raw_no_official = (
        0.35 * mm(clusters['c_impact']) +
        0.20 * mm(clusters['c_lanes']) +
        0.175 * mm(clusters['c_peak']) +
        0.175 * mm(clusters['c_consistency']) +
        0.10  * mm(clusters['c_persistence'])
    )
    score_no_official = (mm(raw_no_official) * 100).round(1)

    # Rank correlation
    corr, pval = spearmanr(v3_score, score_no_official)
    print(f"\n  Spearman rank correlation (full v3 vs no post-hoc): r={corr:.4f}, p={pval:.4f}")
    print(f"  Interpretation: {'Rankings stable — post-hoc fields refine but do not determine order' if corr > 0.9 else 'Rankings change significantly — post-hoc fields have strong influence'}")

    # Check if top 10 zones stay the same
    top10_full      = set(clusters.nlargest(10,'congestion_score')['cluster'])
    clusters_temp   = clusters.copy()
    clusters_temp['score_no_official'] = score_no_official
    top10_no_official = set(clusters_temp.nlargest(10,'score_no_official')['cluster'])
    overlap = len(top10_full & top10_no_official)
    print(f"  Top 10 zones overlap (full vs no post-hoc): {overlap}/10")
    print(f"  Conclusion: {'Post-hoc fields refine scores but do not manufacture hotspots' if overlap >= 8 else 'Post-hoc fields significantly alter top zones'}")
else:
    print("  Score components not found in cluster_stats — run day2_ml_v3.py first")

# ══════════════════════════════════════════════════════════════
# 2. TEMPORAL HOLDOUT — do early clusters predict late ones?
# ══════════════════════════════════════════════════════════════
print("\n── 2. TEMPORAL HOLDOUT VALIDATION ──")
print("Train on Nov–Jan, validate on Feb–Apr")

df['month'] = df['created_datetime'].dt.month

early = df[df['month'].isin([11, 12, 1])].copy()   # Nov–Jan
late  = df[df['month'].isin([2, 3, 4])].copy()     # Feb–Apr

print(f"\n  Early period (Nov–Jan): {len(early):,} violations")
print(f"  Late period  (Feb–Apr): {len(late):,} violations")

# Cluster on early period
sample_e   = early.sample(n=min(40000, len(early)), random_state=42)
coords_rad = np.radians(sample_e[['latitude','longitude']].values)

db_early = DBSCAN(eps=m2r(150), min_samples=20,
                  algorithm='ball_tree', metric='haversine')
sample_e['cluster'] = db_early.fit_predict(coords_rad)

n_early = len(set(sample_e['cluster'])) - (1 if -1 in sample_e['cluster'].values else 0)
print(f"\n  Clusters from early period: {n_early}")

# Get early centroids
centroids_e = (sample_e[sample_e['cluster'] != -1]
               .groupby('cluster')[['latitude','longitude']].mean().reset_index())

# Assign late violations to early clusters
tree_e    = BallTree(np.radians(centroids_e[['latitude','longitude']].values),
                     metric='haversine')
dist, idx = tree_e.query(np.radians(late[['latitude','longitude']].values), k=1)
dist_m    = dist.flatten() * EARTH_R

late['early_cluster'] = np.where(dist_m <= 400,
                                  centroids_e['cluster'].values[idx.flatten()], -1)
late_assigned = (late['early_cluster'] != -1).mean() * 100
print(f"  Late violations matched to early clusters: {late_assigned:.1f}%")

# Do top early clusters still dominate in late period?
early_top = (sample_e[sample_e['cluster'] != -1]
             .groupby('cluster').size().nlargest(10).index.tolist())
late_top  = (late[late['early_cluster'] != -1]
             .groupby('early_cluster').size().nlargest(10).index.tolist())

overlap_t = len(set(early_top) & set(late_top))
print(f"  Top 10 early clusters still top 10 in late period: {overlap_t}/10")
print(f"  Temporal stability: {'Strong ✓ — hotspots persist across time periods' if overlap_t >= 7 else 'Moderate ⚠ — some zone turnover between periods'}")

# ══════════════════════════════════════════════════════════════
# 3. EPS SENSITIVITY TEST
# ══════════════════════════════════════════════════════════════
print("\n── 3. DBSCAN EPS SENSITIVITY ──")
print("Testing eps = 100m, 150m (current), 200m, 300m")

sample_s   = df.sample(n=30000, random_state=42)
coords_s   = np.radians(sample_s[['latitude','longitude']].values)
labels_s   = sample_s['cluster'].values  # from v3 run

results = []
for eps_m in [100, 150, 200, 300]:
    db_test = DBSCAN(eps=m2r(eps_m), min_samples=20,
                     algorithm='ball_tree', metric='haversine')
    lbls = db_test.fit_predict(coords_s)
    n_c  = len(set(lbls)) - (1 if -1 in lbls else 0)
    noise = (lbls == -1).sum()
    if n_c > 1:
        sil = silhouette_score(coords_s, lbls, sample_size=5000, random_state=42)
    else:
        sil = -1
    results.append({'eps_m': eps_m, 'clusters': n_c, 'noise': noise, 'silhouette': round(sil,4)})
    print(f"  eps={eps_m}m → clusters={n_c}, noise={noise}, silhouette={sil:.4f}")

best = max(results, key=lambda x: x['silhouette'])
print(f"\n  Best eps by silhouette: {best['eps_m']}m (score={best['silhouette']})")
best_eps = best['eps_m']
print(f"  Current eps=150m {'is optimal ✓' if best_eps == 150 else f'suboptimal — consider eps={best_eps}m'}")

# ══════════════════════════════════════════════════════════════
# 4. FEATURE ABLATION STUDY
# ══════════════════════════════════════════════════════════════
print("\n── 4. FEATURE ABLATION — contribution of each component ──")
print("Removing one component at a time, measuring rank shift vs full v3")

comp_weights = {
    'c_impact':      0.30,
    'c_lanes':       0.15,
    'c_peak':        0.15,
    'c_consistency': 0.15,
    'c_persistence': 0.10,
    'c_official':    0.10,
    'c_compound':    0.05,
}
comp_cols = list(comp_weights.keys())

if all(c in clusters.columns for c in comp_cols):
    full_score = clusters['congestion_score'].values
    print(f"\n  {'Component removed':25} {'Spearman r':>12} {'Top-10 overlap':>15} {'Verdict':>12}")
    print(f"  {'-'*65}")
    for drop_c in comp_cols:
        remaining = {k:v for k,v in comp_weights.items() if k != drop_c}
        total_w   = sum(remaining.values())
        ablated   = sum(clusters[k] * (v/total_w) for k,v in remaining.items())
        ablated_n = (mm(ablated) * 100).values

        r, _      = spearmanr(full_score, ablated_n)
        top10_f   = set(clusters.nlargest(10,'congestion_score')['cluster'])
        clusters['ablated'] = ablated_n
        top10_a   = set(clusters.nlargest(10,'ablated')['cluster'])
        ov        = len(top10_f & top10_a)
        verdict   = 'Low impact' if r > 0.97 else ('Medium' if r > 0.90 else 'High impact')
        print(f"  {drop_c:25} {r:>12.4f} {ov:>14}/10 {verdict:>12}")
else:
    print("  Component columns not found — ensure cluster_stats.csv is from v3 run")

# ══════════════════════════════════════════════════════════════
# 5. HDBSCAN COMPARISON (if available)
# ══════════════════════════════════════════════════════════════
print("\n── 5. HDBSCAN COMPARISON ──")
try:
    import hdbscan
    sample_h   = df.sample(n=30000, random_state=42)
    coords_h   = np.radians(sample_h[['latitude','longitude']].values)

    hdb = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10,
                           metric='haversine', cluster_selection_epsilon=m2r(150))
    h_labels = hdb.fit_predict(coords_h)
    n_hdb    = len(set(h_labels)) - (1 if -1 in h_labels else 0)
    noise_h  = (h_labels == -1).sum()
    sil_h    = silhouette_score(coords_h, h_labels, sample_size=5000) if n_hdb > 1 else -1

    print(f"  HDBSCAN → clusters={n_hdb}, noise={noise_h}, silhouette={sil_h:.4f}")
    print(f"  DBSCAN  → clusters=~184, silhouette=0.749 (on full dataset)")
    print(f"  {'HDBSCAN better' if sil_h > 0.749 else 'DBSCAN better or equal'} for this dataset")
except ImportError:
    print("  hdbscan not installed — run: pip install hdbscan")
    print("  DBSCAN silhouette=0.749 stands as current best")

# ══════════════════════════════════════════════════════════════
# 6. FINAL VALIDATION SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("VALIDATION SUMMARY — USE IN PITCH")
print("="*60)
print(f"  Silhouette score (v3)           : 0.749  [STRONG]")
print(f"  Davies-Bouldin (v3)             : 0.341  [EXCELLENT]")
print(f"  Leakage test                    : see rank correlation above")
print(f"  Temporal stability              : see top-10 overlap above")
print(f"  Optimal eps                     : see sensitivity test above")
print(f"  Most impactful component        : see ablation above")
print(f"  HDBSCAN comparison              : see above")
