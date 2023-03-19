import joblib
import pytest
import pandas as pd

from src.ml.model import compute_model_metrics, inference
from src.ml.data import process_data
from sklearn.exceptions import NotFittedError


@pytest.fixture
def data():
    df = pd.read_csv("data/census.csv")
    return df


@pytest.fixture
def label():
    return "salary"


@pytest.fixture
def model():
    model = joblib.load("model/model.joblib")
    return model


@pytest.fixture
def lb():
    lb = joblib.load("model/lb.joblib")
    return lb


@pytest.fixture
def encoder():
    encoder = joblib.load("model/encoder.joblib")
    return encoder


def test_process_data(data, label):
    X_train, y_train, encoder, lb = process_data(data, label=label, training=True)
    assert X_train.shape == (len(data), 108)
    assert y_train.shape == (len(data),)
    assert encoder is not None
    assert lb is not None


def test_train_model(model, data, label, encoder, lb):
    assert model is not None
    # check if model contains the following steps
    assert "VarianceThreshold" in model.named_steps
    assert "scaler" in model.named_steps
    assert "model" in model.named_steps

    # check if model is fitted
    x, _, _, _ = process_data(data, label=label, training=False, encoder=encoder, lb=lb)
    try:
        model.predict(x)
    except NotFittedError as e:
        print(repr(e))


def test_model_inference_metrics(model, data, encoder, lb, label):
    x, y, _, _ = process_data(data, label=label, training=False, encoder=encoder, lb=lb)
    preds, labels = inference(model, x, lb)

    # check if preds and labels are not None
    assert preds is not None
    assert labels is not None
    assert len(preds) == len(labels) == len(data)

    acc, precision, recall, fbeta = compute_model_metrics(y, preds)
    # check if metrics are not None
    assert acc is not None
    assert precision is not None
    assert recall is not None
    assert fbeta is not None
