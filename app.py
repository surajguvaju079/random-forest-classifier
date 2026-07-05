
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

model = joblib.load("models/churn_model.pkl")


class Driver(BaseModel):
    loads_completed:int
    last_login_days:int
    rating:float

@app.post("/predict")
def predict(driver:Driver):
    data = pd.DataFrame([driver.model_dump()])
    probabilities = model.predict_proba(data)
    prediction = model.predict(data)
    return {
        "churn":int(prediction[0]),
        "probability":probabilities
    }
    
