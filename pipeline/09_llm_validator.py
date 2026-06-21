import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime
import json
from dotenv import load_dotenv

# Load .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

try:
    from groq import Groq
except ImportError:
    Groq = None

def run_llm_validator(conn=None):
    if not Groq:
        print("  [llm_validator] groq library not installed.")
        return
        
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("  [llm_validator] GROQ_API_KEY not found in environment or .env. Skipping validation.")
        return

    client = Groq(api_key=api_key)

    close_conn = False
    if not conn:
        db_path = os.path.join(os.path.dirname(__file__), '..', 'db', 'gridlock.db')
        conn = sqlite3.connect(db_path, timeout=10)
        close_conn = True

    try:
        # Get the latest forecast timestamp
        latest_ts = conn.execute("SELECT MAX(forecast_at) FROM cluster_forecasts").fetchone()[0]
        if not latest_ts:
            return

        # Pull top 3 critical predictions for the next hour
        df = pd.read_sql(f"""
            SELECT f.cluster, f.forecast_at, f.hour_offset, f.target_datetime, 
                   f.predicted_violations, f.zone_name, f.police_station,
                   s.daily_rate, s.congestion_score
            FROM cluster_forecasts f
            LEFT JOIN (
                SELECT cluster, daily_rate, congestion_score 
                FROM cluster_scores 
                WHERE recalculated_at = (SELECT MAX(recalculated_at) FROM cluster_scores)
            ) s ON f.cluster = s.cluster
            WHERE f.forecast_at = ? AND f.hour_offset = 1
            ORDER BY f.predicted_violations DESC
            LIMIT 3
        """, conn, params=(latest_ts,))

        if df.empty:
            return

        for _, row in df.iterrows():
            # Check if we already validated this
            existing = conn.execute("""
                SELECT 1 FROM forecast_validations 
                WHERE cluster=? AND forecast_at=? AND hour_offset=?
            """, (int(row['cluster']), latest_ts, int(row['hour_offset']))).fetchone()
            
            if existing:
                continue

            # Construct the prompt
            prompt = f"""You are a Police Command Center Validation AI for Bangalore traffic police.
Review the following machine learning forecast:
- Zone: {row['zone_name']} (Station: {row['police_station']})
- Time of Prediction: {row['target_datetime']}
- ML Predicted Violations in this 1 hour: {row['predicted_violations']:.1f}
- Zone's Historical AVERAGE daily rate (24h): {row['daily_rate']:.1f}
- Current Live Congestion Score: {row['congestion_score']}/100

Task:
1. Determine if this forecast is realistic ('Valid') or a statistical anomaly/error ('Suspect'). Note: If 1-hour prediction is more than 50% of the entire daily rate, it might be Suspect unless congestion score is very high (90+).
2. Write a 1-sentence tactical briefing for the patrol officer.

Respond ONLY in valid JSON format:
{{"status": "Valid" or "Suspect", "rationale": "Your 1-sentence tactical briefing."}}"""

            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                status = result.get('status', 'Valid')
                rationale = result.get('rationale', 'No rationale provided.')

                conn.execute("""
                    INSERT INTO forecast_validations 
                    (cluster, forecast_at, hour_offset, validation_status, llm_rationale)
                    VALUES (?, ?, ?, ?, ?)
                """, (int(row['cluster']), latest_ts, int(row['hour_offset']), status, rationale))
                conn.commit()

            except Exception as e:
                print(f"  [llm_validator] API Error for cluster {row['cluster']}: {e}")

    except Exception as e:
        print(f"  [llm_validator] General Error: {e}")
    finally:
        if close_conn:
            conn.close()

if __name__ == "__main__":
    run_llm_validator()
