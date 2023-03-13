from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

# load model
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
    age: float = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: float = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: float = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    native_country: str = Field(..., example="United-States", alias="native-country")
    capital_gain: float = Field(..., example=2147, alias="capital-gain")
    capital_loss: float = Field(..., example=0, alias="capital-loss")
    hours_per_week: float = Field(..., example=40, alias="hours-per-week")


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
