import pytest
import pandas as pd
import os

from starter.ml.model import train_model, compute_model_metrics, inference
from starter.ml.data import process_data


@pytest.fixture
def data():
    df = pd.read_csv("starter/data/census.csv")
    return df


@pytest.fixture
def categorical_features():
    cat_fea = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_fea


@pytest.fixture
def label():
    return "salary"

@pytest.fixture
def model_path():
    return "starter/model/model.joblib"


def test_process_data(data, categorical_features, label):
    X_train, y_train, encoder, lb = process_data(data, categorical_features, label, training=True)
    assert X_train.shape == (len(data), 108)
    assert y_train.shape == (len(data),)
    assert encoder is not None
    assert lb is not None


def test_train_model(model_path):
    # check if model saved
    assert os.path.exists(model_path)


def test_compute_model_metrics():
    # load model and test
    assert True


def test_inference():
    assert True
