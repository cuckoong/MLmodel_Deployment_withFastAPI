# Script to train machine learning model.
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.model import train_model, compute_model_metrics, inference, compute_slicing_metrics
from ml.data import process_data

# get root path of current project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# set working directory to project root
os.chdir(project_root)


def training(data_path="data/census.csv", test_size=0.2, target="salary", slicing_feature='sex'):
    """
    Train the model and save it to disk.
    Parameters
    ----------
    data_path : str, default="data/census.csv"
        Path to the data.
    test_size : float, default=0.2
        The proportion of the data to use for testing.
    target : str, default="salary"
        The target column name.
    slicing_feature: str, default='sex'
        The slicing feature column name.

    Returns
    -------

    """

    data = pd.read_csv(data_path)
    # replace hyphens with underscores
    data.columns = [x.replace("-", "_") for x in data.columns]

    # Split the data into train and test sets.
    train, test = train_test_split(data, test_size=test_size)

    X_train, y_train, encoder, lb = process_data(
        train, label=target, training=True
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
    accuracy, precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision}, Recall: {recall}, F1: {fbeta}, Accuracy: {fbeta}")

    # show model performance of slicing data
    compute_slicing_metrics(model, test, slicing_feature, encoder, lb)

    # # save model
    joblib.dump(model, "model/model.joblib")
    joblib.dump(encoder, "model/encoder.joblib")
    joblib.dump(lb, "model/lb.joblib")


if __name__ == "__main__":
    training()