from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    with open("model.pkl", "rb") as f:
        model, poly = pickle.load(f)

    df = pd.read_csv("petrol_price.csv")
    df['date'] = pd.to_datetime(df['date'], format='%Y_%b')

    df = df.sort_values('date')
    df = df.drop_duplicates(subset='date', keep='last')

    base_length = len(df)

    print("✅ Model loaded")

except Exception as e:
    print("❌ ERROR LOADING MODEL:", e)
    model = None


@app.get("/")
def home():
    return {"message": "API Running"}


@app.get("/predict_range")
def predict_range(years: int = 5):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        periods = years * 12

        future_index = np.arange(base_length, base_length + periods)

        # 🔥 FIX: reshape correctly
        future_index = future_index.reshape(-1, 1)

        X = poly.transform(future_index)

        preds = model.predict(X)

        future_dates = pd.date_range(
            start=pd.Timestamp.today(),
            periods=periods,
            freq='MS'
        )

        return [
            {"ds": str(d.date()), "yhat": float(p)}
            for d, p in zip(future_dates, preds)
        ]

    except Exception as e:
        print("❌ ERROR IN PREDICT:", e)
        raise HTTPException(status_code=500, detail=str(e))