from fastapi import FastAPI
from pathlib import Path
import os
import pandas as pd

app = FastAPI()

# Resolve the CSV path relative to this file's location so it works
# correctly regardless of the working directory inside the container.
BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "Backend" / "predictions.csv"

# Load precomputed data once at startup to avoid reloading on every request.
df = pd.read_csv(csv_path)

@app.get("/")
def home():
    return {"message": "Fuel Prediction API Running"}

@app.get("/predict")
def predict(years: int = 5):
    periods = years * 12
    result = df.tail(periods)
    return result.to_dict(orient="records")