import pandas as pd
import numpy as np

cs = pd.read_csv("cluster_stats.csv")
def mm(s): return (s-s.min())/(s.max()-s.min()+1e-9)

# Ablation-informed weights
cs['score_v4'] = (
    0.20 * mm(cs['c_impact']) +
    0.25 * mm(cs['c_lanes']) +       # was 15%
    0.15 * mm(cs['c_peak']) +
    0.25 * mm(cs['c_consistency']) + # was 15%
    0.10 * mm(cs['c_persistence']) +
    0.04 * mm(cs['c_official']) +
    0.01 * mm(cs['c_compound'])
)
cs['score_v4'] = (mm(cs['score_v4'])*100).round(1)

# Compare top 10
top_v3 = cs.nlargest(10,'congestion_score')[['zone_name','congestion_score']]
top_v4 = cs.nlargest(10,'score_v4')[['zone_name','score_v4']]
print("v3 top 10:"); print(top_v3.to_string(index=False))
print("\nv4 top 10:"); print(top_v4.to_string(index=False))