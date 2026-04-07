from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

# ✅ CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD MODEL =================

try:
    # load trained model
    with open("model.pkl", "rb") as f:
        model, poly = pickle.load(f)

    # load original dataset (for time reference)
    df = pd.read_csv("petrol_price.csv")
    df.columns = df.columns.str.strip()

    df = df[['date', 'Delhi']]
    df.columns = ['date', 'price']

    df['date'] = pd.to_datetime(df['date'], format='%Y_%b')

    df = df.sort_values('date')
    df = df.drop_duplicates(subset='date', keep='last')

    base_length = len(df)
    start_date = df['date'].min()

    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ ERROR LOADING MODEL:", e)
    model = None


# ================= ROOT =================

@app.get("/")
def home():
    return {"message": "Fuel Prediction API Running 🚀"}


# ================= RANGE PREDICTION =================

@app.get("/predict_range")
def predict_range(years: int = 5):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        periods = years * 12

        future_index = np.arange(base_length, base_length + periods)
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
        print("❌ ERROR IN RANGE:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ================= DATE PREDICTION =================

@app.get("/predict_by_date")
def predict_by_date(date: str):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_date = pd.to_datetime(date)

        # convert date → time index
        months_diff = (input_date.year - start_date.year) * 12 + (input_date.month - start_date.month)

        X = poly.transform([[months_diff]])

        pred = model.predict(X)[0]

        return {
            "date": str(input_date.date()),
            "predicted_price": round(float(pred), 2)
        }

    except Exception as e:
        print("❌ ERROR IN DATE:", e)
        raise HTTPException(status_code=500, detail=str(e))