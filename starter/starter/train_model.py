# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

# get root path
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

# Add code to load in the data.
data = pd.read_csv("starter/data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)

# Compute and print the model's metrics.
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision}, Recall: {recall}, F1: {fbeta}")

# save model
import joblib
joblib.dump(model, "starter/model/model.joblib")

