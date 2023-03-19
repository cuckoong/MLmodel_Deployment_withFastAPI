# Project Summary
* This project is to build an API application that can be used to infer whether a person makes over 50K a year based
on census data. The API is built using FastAPI and deployed on Heroku. The model is trained using data from the census 
income dataset.

# Environment Setup
* Create a virtual environment
```CONDA CREATE -n <env_name> python=3.8```
* Activate the virtual environment
```CONDA ACTIVATE <env_name>```
* Install the required packages
```PIP INSTALL -r requirements.txt```
* Run the application locally with localhost and port as 8000
```uvicorn main:app --reload```
* Run the tests
```pytest -c test/pytest.ini```

# Model Information
* The model card is in the model_card.md file.

# Project Structure
```
├── data
│   ├── census_income.csv
├── model
│   ├── lb.joblib
│   ├── encoder.joblib
├── src
│   ├── ml
│   │   ├── model.py
│   │   ├── data.py
│   ├── train_model.py
│   ├── get_inference_fromAPI.py
├── test
│   ├── pytest.ini
│   ├── test_main.py
│   ├── test_ml.py
├── Procfile
├── Aptfile
├── main.py
├── requirements.txt
├── model_card.md
├── README.md
├── LICENSE.txt
├── outputs
│   ├── example.png
│   ├── live_post.png
│   ├── continuous_deployment.png
│   ├── live_get.png
│   ├── slice_output.txt
```