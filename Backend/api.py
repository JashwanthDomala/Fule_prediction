from fastapi import FastAPI
import pandas as pd

app = FastAPI()

# Load precomputed data
df = pd.read_csv("predictions.csv")

@app.get("/")
def home():
    return {"message": "Fuel Prediction API Running"}

@app.get("/predict")
def predict(years: int = 5):
    periods = years * 12
    result = df.tail(periods)
    return result.to_dict(orient="records")