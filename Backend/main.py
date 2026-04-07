from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle

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
        model = pickle.load(f)
    print("Model loaded")
except:
    model = None


@app.get("/")
def home():
    return {"message": "Fuel Prediction API Running"}


@app.get("/predict_by_date")
def predict_by_date(date: str):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_date = pd.to_datetime(date).to_period('M').to_timestamp()

        future_df = pd.DataFrame({'ds': [input_date]})

        forecast = model.predict(future_df)

        price = float(forecast['yhat'].iloc[0])

        return {
            "date": str(input_date.date()),
            "predicted_price": round(price, 2)
        }

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict_range")
def predict_range(years: int = 5):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        future_dates = pd.date_range(
            start=pd.Timestamp.today().to_period('M').to_timestamp(),
            periods=years * 12,
            freq='MS'
        )

        future_df = pd.DataFrame({'ds': future_dates})

        forecast = model.predict(future_df)

        return forecast[['ds', 'yhat']].to_dict(orient="records")

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))