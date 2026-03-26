from fastapi import FastAPI, HTTPException
from pathlib import Path
import os
import pandas as pd

app = FastAPI()

# Resolve the CSV path relative to this file's location so it works
# correctly regardless of the working directory inside the container.
BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "Backend" / "predictions.csv"

# Track whether the CSV loaded successfully so the health check and
# startup event can surface failures clearly instead of failing silently.
csv_loaded: bool = False
df: pd.DataFrame = pd.DataFrame()

try:
    df = pd.read_csv(csv_path)
    csv_loaded = True
    print(f"[startup] CSV loaded successfully from {csv_path} ({len(df)} rows)")
except Exception as e:
    print(f"[startup] ERROR: Failed to load CSV from {csv_path}: {e}")

@app.on_event("startup")
async def validate_csv():
    if not csv_loaded:
        raise RuntimeError(
            f"CSV file could not be loaded from '{csv_path}'. "
            "The app cannot serve predictions. Check the path and file contents."
        )
    print("[startup] CSV validation passed — app is ready to serve requests.")

@app.get("/health")
def health():
    if not csv_loaded:
        raise HTTPException(
            status_code=500,
            detail=f"CSV not loaded. Expected path: {csv_path}"
        )
    return {"status": "ok", "rows": len(df)}

@app.get("/")
def home():
    return {"message": "Fuel Prediction API Running"}

@app.get("/predict")
def predict(years: int = 5):
    if not csv_loaded:
        raise HTTPException(status_code=500, detail="Prediction data unavailable — CSV failed to load at startup.")
    periods = years * 12
    result = df.tail(periods)
    return result.to_dict(orient="records")