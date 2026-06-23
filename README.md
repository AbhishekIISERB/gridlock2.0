# How to Run Belaku on Your Machine

## Prerequisites
- Python 3.10 or higher
- Git (if cloning from GitHub)

---

## Step 1: Get the Code

**Option A — Clone from GitHub:**
```bash
git clone https://github.com/AbhishekIISERB/gridlock2.0.git
cd gridlock2.0
```

**Option B — From ZIP:**
```bash
unzip gridlock2.0.zip
cd gridlock2.0
```

---

## Step 2: Place the Raw Data File

Place the provided raw dataset in the **root** of the `gridlock2.0/` folder:

```
gridlock2.0/
  jan_to_may_police_violation_anonymized.csv   ← place here
  cluster_stats.csv
  api.py
  pipeline/
  frontend/
  ...
```

---

## Step 3: Create a Virtual Environment and Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 4: Run the Preprocessing Script

This step reads the raw dataset, cleans it, engineers features, runs DBSCAN spatial clustering, and generates the `parking_clustered.csv` file that the pipeline depends on.

```bash
python pipeline/00_preprocess.py
```

**Expected output:**
```
Loading raw dataset...
Raw shape: (291957, ...)
After geo filter: 291957
After dropping rejected violations: 243218
Running DBSCAN (eps=150m, min_samples=20)...
Clusters: 184
Assigned: 238450 (98.0%)
...
Saved all outputs to project root.
  cluster_stats.csv
  parking_clustered.csv
```

> **Note:** This takes approximately 3–5 minutes to run. It only needs to be run once.

---

## Step 5: Add Your Groq API Key (for LLM Validation)

Create a file named `.env` in the root `gridlock2.0/` folder with the following content:

```
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free API key at: https://console.groq.com
> If you skip this step, everything still works — the LLM validation layer is simply skipped.

---

## Step 6: Run the Application

Run the one-command demo launcher from the root folder:

```bash
bash pipeline/demo_run.sh
```

This script will automatically:
1. Create the SQLite database and all tables
2. Seed the database with the first 7 days of data
3. Run the initial congestion impact score calculation
4. Start the live background ML pipeline (adds new simulated violations every 10 seconds)
5. Start the FastAPI REST server on **port 8502**

---

## Step 7: Open the Dashboard

Once the script prints `GRIDLOCK AI IS LIVE`, open your browser and go to:

```
http://localhost:8502
```

The dashboard will load with the live heatmap, AI predictions, and deployment plan already populated.

---

## Stopping the Application

Press `Ctrl+C` in the terminal where `demo_run.sh` is running. The script will automatically shut down the background pipeline and API server.

---

## Troubleshooting

**`ModuleNotFoundError`** — Make sure your virtual environment is activated (`source venv/bin/activate`) and you ran `pip install -r requirements.txt`.

**`parking_clustered.csv not found`** — Make sure you placed the file directly inside the `gridlock2.0/` root folder, not in a subfolder.

**Port 8502 already in use** — Kill any existing process on that port:
```bash
# Mac/Linux
lsof -ti:8502 | xargs kill -9
# Windows
netstat -ano | findstr :8502
taskkill /PID <PID> /F
```

**LLM validation not appearing** — Check that your `.env` file exists in the root folder and contains a valid `GROQ_API_KEY`. The LLM validation runs on the top 3 predicted hotspots only, and results appear on the next pipeline tick (~10 seconds after startup).
