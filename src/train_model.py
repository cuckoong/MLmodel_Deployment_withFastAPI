# Script to train machine learning model.
import os

import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.model import train_model, compute_model_metrics, inference, slicing_metrics
from ml.data import process_data

if __name__ == "__main__":
    # get root path of current project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    # set working directory to project root
    os.chdir(project_root)
    data = pd.read_csv("data/census.csv")

    # replace hyphens with underscores
    data.columns = [x.replace("-", "_") for x in data.columns]

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    # Train and save a model.
    model = train_model(X_train, y_train)

    # Compute and print the model's metrics.
    preds, labels = inference(model, X_test, lb)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision}, Recall: {recall}, F1: {fbeta}")

    # model performance of slicing data
    slicing_performance = slicing_metrics(model, test, 'sex', encoder, lb)

    # # save model
    import joblib
    joblib.dump(model, "model/model.joblib")
    joblib.dump(encoder, "model/encoder.joblib")
    joblib.dump(lb, "model/lb.joblib")
