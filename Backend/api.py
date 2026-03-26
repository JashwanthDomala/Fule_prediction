from fastapi import FastAPI
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (ok for project)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model once
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Fuel Prediction API Running"}


@app.get("/predict")
def predict(years: int = 5):
    periods = years * 12

    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)

    result = forecast[['ds', 'yhat']].tail(periods)

    return result.to_dict(orient="records")