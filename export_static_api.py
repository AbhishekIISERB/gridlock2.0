import os
import json
import requests

API_BASE = 'http://localhost:8502/api'
OUT_DIR = os.path.join(os.path.dirname(__file__), 'frontend', 'api')

os.makedirs(OUT_DIR, exist_ok=True)

endpoints = {
    'kpis': 'kpis',
    'zones': 'zones',
    'alerts': 'alerts',
    'anomalies': 'anomalies',
    'forecasts': 'forecasts',
    'deployment': 'deployment',
    'score-history': 'score-history',
    'heatmap_volume': 'heatmap?hours=24&mode=volume',
    'heatmap_impact': 'heatmap?hours=24&mode=impact'
}

print(f"Exporting live database state to {OUT_DIR}...")

for name, query in endpoints.items():
    try:
        resp = requests.get(f'{API_BASE}/{query}')
        resp.raise_for_status()
        data = resp.json()
        
        out_path = os.path.join(OUT_DIR, f'{name}.json')
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f" ✓ Exported {name}.json")
    except Exception as e:
        print(f" ✗ Failed to export {name}: {e}")

print("Done! The frontend folder is now ready to be deployed as a static site to Vercel.")
