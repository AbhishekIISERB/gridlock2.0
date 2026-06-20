"""
10_validate_live.py — Compare live rolling scores vs batch top_enforcement_zones.csv
Run after full CSV replay: python pipeline/10_validate_live.py
"""
import sqlite3, pandas as pd, os
from scipy.stats import spearmanr

DB_PATH  = os.path.join(os.path.dirname(__file__), '..', 'db', 'gridlock.db')
BATCH_CSV = os.path.join(os.path.dirname(__file__), '..', 'top_enforcement_zones.csv')

conn = sqlite3.connect(DB_PATH, timeout=10)

live = pd.read_sql("""
    SELECT cluster, zone_name, congestion_score AS congestion_score_live,
           risk_tier, violation_count, daily_rate
    FROM cluster_scores
    WHERE recalculated_at = (SELECT MAX(recalculated_at) FROM cluster_scores)
    ORDER BY congestion_score_live DESC
""", conn)
conn.close()

batch = pd.read_csv(BATCH_CSV)
batch = batch.rename(columns={'congestion_score': 'congestion_score_batch'})

merged = live.merge(batch[['cluster', 'congestion_score_batch', 'zone_name']],
                    on='cluster', suffixes=('_live','_batch'))

if len(merged) < 5:
    print(f"Only {len(merged)} common clusters — replay may not be complete yet")
else:
    r, p = spearmanr(merged['congestion_score_live'], merged['congestion_score_batch'])
    print("=" * 55)
    print("LIVE vs BATCH VALIDATION")
    print("=" * 55)
    print(f"Common clusters      : {len(merged)}")
    print(f"Spearman r           : {r:.4f}  (target: >0.90)")
    print(f"p-value              : {p:.4e}")
    print(f"Status               : {'✓ PASS' if r > 0.90 else '✗ FAIL'}")
    print()
    print("Top 10 live vs batch scores:")
    out = merged[['zone_name_live','congestion_score_live','congestion_score_batch']]\
        .nlargest(10, 'congestion_score_live')
    out.columns = ['Zone', 'Live score', 'Batch score']
    print(out.to_string(index=False))
