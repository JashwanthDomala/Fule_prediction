from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd

app = FastAPI()

# ✅ Get correct path (same folder as this file)

BASE_DIR = Path(**file**).resolve().parent
csv_path = BASE_DIR / "predictions.csv"

# Global variables

df = None
csv_loaded = False

# 🔥 Load CSV

try:
df = pd.read_csv(csv_path)
csv_loaded = True
print(f"[startup] Loaded CSV from {csv_path} with {len(df)} rows")
except Exception as e:
print(f"[startup] ERROR loading CSV: {e}")

# 🔥 Validate at startup

@app.on_event("startup")
async def startup_check():
if not csv_loaded:
raise RuntimeError(
f"CSV file not found at {csv_path}. Please check deployment."
)
print("[startup] App is ready")

# ✅ Health check

@app.get("/health")
def health():
if not csv_loaded:
raise HTTPException(status_code=500, detail="CSV not loaded")
return {"status": "ok", "rows": len(df)}

# ✅ Home route

@app.get("/")
def home():
return {"message": "Fuel Prediction API Running"}

# ✅ Prediction route

@app.get("/predict")
def predict(years: int = 5):
if not csv_loaded:
raise HTTPException(
status_code=500,
detail="Prediction data unavailable"
)

```
periods = years * 12
result = df.tail(periods)

return result.to_dict(orient="records")
```
