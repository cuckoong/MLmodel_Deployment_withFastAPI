import requests

def test_get_main():
    response = requests.get("http://localhost:8000/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Welcome to the model inference API"}


def test_post_predict_1():
    # invalid data
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
    }
    response = requests.post("http://localhost:8000/predict", json=data)
    assert response.status_code == 422


def test_post_predict_2():
    # valid data
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "capital-gain": 2147,
        "capital-loss": 0,
        "hours-per-week": 40
    }

    response = requests.post("http://localhost:8000/predict/", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}
