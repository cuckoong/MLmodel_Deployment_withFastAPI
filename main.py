from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from src.ml.data import process_data
from src.ml.model import inference

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global model, encoder, lb, categorical_features
    model = joblib.load("model/model.joblib")
    encoder = joblib.load("model/encoder.joblib")
    lb = joblib.load("model/lb.joblib")
    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]


class inputs(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(..., alias="native-country")
    capital_gain: float = Field(..., alias="capital-gain")
    capital_loss: float = Field(..., alias="capital-loss")
    hours_per_week: float = Field(..., alias="hours-per-week")

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                'race': 'White',
                'sex': 'Male',
                'native-country': 'United-States',
                'capital-gain': 2147,
                'capital-loss': 0,
                'hours-per-week': 40
            }
        }


@app.get("/")
def read_root():
    return {"msg": "Welcome to the model inference API"}


@app.post("/predict/")
async def predict(item: inputs):
    # convert input to dataframe
    X = pd.DataFrame([item.dict()])
    # process data
    X_prepared, _, _, _ = process_data(
        X,
        categorical_features=categorical_features,
        training=False,
        encoder=encoder,
        lb=None)

    # run inference
    _, labels = inference(model, X_prepared, lb)
    return {"prediction": labels[0]}
