import os
import json
import requests

API_BASE = 'http://localhost:8502/api'
OUT_DIR = os.path.join(os.path.dirname(__file__), 'frontend', 'api')

os.makedirs(OUT_DIR, exist_ok=True)

endpoints = [
    'kpis',
    'zones',
    'alerts',
    'anomalies',
    'forecasts',
    'deployment',
    'score-history'
]

print(f"Exporting live database state to {OUT_DIR}...")

for ep in endpoints:
    try:
        resp = requests.get(f'{API_BASE}/{ep}')
        resp.raise_for_status()
        data = resp.json()
        
        out_path = os.path.join(OUT_DIR, f'{ep}.json')
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f" ✓ Exported {ep}.json")
    except Exception as e:
        print(f" ✗ Failed to export {ep}: {e}")

print("Done! The frontend folder is now ready to be deployed as a static site to Vercel.")
