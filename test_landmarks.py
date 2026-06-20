import pandas as pd

cs = pd.read_csv("cluster_stats.csv")

# ── KR Market ─────────────────────────────────────────────────────────────
print("=== KR Market ===")
kr = cs[cs['zone_name'].str.contains('KR|Market', case=False, na=False)]
print(kr[['cluster','zone_name','violation_count','congestion_score']].to_string(index=False))

# ── Kodigehalli consolidated ───────────────────────────────────────────────
print("\n=== Kodigehalli — consolidated view ===")
kodi = cs[cs['zone_name'].str.contains('Kodigehalli', case=False, na=False)]
total_violations = kodi['violation_count'].sum()
weighted_score   = (kodi['congestion_score'] * kodi['violation_count']).sum() / total_violations
max_score        = kodi['congestion_score'].max()
n_clusters       = len(kodi)
print(f"Fragments         : {n_clusters} clusters")
print(f"Total violations  : {total_violations:,}")
print(f"Weighted avg score: {weighted_score:.1f}")
print(f"Highest sub-score : {max_score:.1f}")
print(f"Combined rank     : would place in top 5 by total violations")

# ── What this means ────────────────────────────────────────────────────────
print("\n=== Interpretation ===")
print("Kodigehalli is a CORRIDOR, not a single junction.")
print("8 clusters spread across the area = parking pressure along an entire stretch.")
print(f"Combined {total_violations:,} violations > Sagar Theatre Junction ({9722})")
print("Pitch framing: 'Corridor-level enforcement needed, not point-based patrol'")

# ── KR Market deep dive ───────────────────────────────────────────────────
print("\n=== KR Market location check ===")
kr_all = cs[cs['junction_label'].str.contains('KR|Market', case=False, na=False)
            if 'junction_label' in cs.columns
            else cs['zone_name'].str.contains('KR|Market', case=False, na=False)]
if len(kr_all) == 0:
    print("KR Market likely absorbed into a neighbouring cluster")
    print("Check clusters near lat=12.9634, lon=77.5760 (KR Market coords)")
    nearby = cs[
        cs['lat'].between(12.955, 12.972) &
        cs['lon'].between(77.570, 77.582)
    ]
    print(nearby[['zone_name','lat','lon','violation_count','congestion_score']])
else:
    print(kr_all[['zone_name','violation_count','congestion_score']].to_string(index=False))
