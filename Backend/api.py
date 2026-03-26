from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd

app = FastAPI()

# CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CSV
BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "predictions.csv"

df = None
csv_loaded = False

try:
    df = pd.read_csv(csv_path)
    csv_loaded = True
    print(f"[startup] Loaded CSV: {len(df)} rows")
except Exception as e:
    print(f"[startup] ERROR: {e}")

@app.get("/health")
def health():
    if not csv_loaded:
        raise HTTPException(status_code=500, detail="CSV not loaded")
    return {"status": "ok", "rows": len(df)}

@app.get("/")
def home():
    return {"message": "Fuel Prediction API Running"}

@app.get("/predict")
def predict(years: int = 5):
    if not csv_loaded:
        raise HTTPException(status_code=500, detail="Data not available")

    periods = years * 12
    result = df.tail(periods)

    return result.to_dict(orient="records")