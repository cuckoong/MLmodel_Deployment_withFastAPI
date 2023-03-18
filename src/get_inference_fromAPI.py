import requests


def get_response_liveAPI(path, data):
    """
    Get response from live API
    Parameters
    ----------
    path: str
        path to the API
    data:
        data to be sent to the API
        example:
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

    Returns
    -------
    status_code: int
        status code of the response;
        200: success, 422: invalid data, 404: not found, 500: internal server error
    inference: str
        inference of the data, "<=50K" or ">50K"
    """
    response = requests.post(path, json=data)
    status_code = response.status_code
    inference = response.json()
    return status_code, inference


if __name__ == '__main__':
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

    status_code, inference = get_response_liveAPI(path='https://panda-demo-api.herokuapp.com/predict',
                                                  data=data)
    print(f'status code:{status_code}')
    print(f'inference:{inference}')
