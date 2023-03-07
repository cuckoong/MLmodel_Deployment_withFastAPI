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

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class inputs(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float


@app.get("/")
def read_root():
    return {"Welcome to the model inference API"}


@app.post("/predict/")
async def predict(item: inputs):
    # convert input to dataframe
    X = pd.DataFrame([item.dict()])
    X.rename(columns={"native_country": "native-country",
                      "marital_status": "marital-status",
                      "capital_gain": "capital-gain",
                      "capital_loss": "capital-loss",
                      "hours_per_week": "hours-per-week"},
             inplace=True)

    # process data
    X_prepared, _, _, _ = process_data(
        X,
        categorical_features=categorical_features,
        training=False,
        encoder=encoder,
        lb=None)

    # run inference
    preds = model.predict(X_prepared)
    return {"prediction":preds.tolist()[0:]}

#
# {
#   "age": 39,
#   "workclass": "State-gov",
#   "fnlgt": 77516,
#   "education": "Bachelors",
#   "education_num": 13,
#   "marital_status": "Never-married",
#   "occupation": "Adm-clerical",
#   "relationship": "Not-in-family",
#   "race": "White",
#   "sex": "Male",
#   "native_country": "United-States",
#   "capital_gain": 2147,
#   "capital_loss": 0,
#   "hours_per_week": 40
# }
#
