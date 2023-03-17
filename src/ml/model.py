from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    steps = [
        ('VarianceThreshold', VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    pipe = Pipeline(steps=steps)
    pipe.fit(X_train, y_train)
    return pipe


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X, lb=None):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer used for label inverse transform.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    labels : np.array
        Inverse transformed labels from predictions.
    """
    preds = model.predict(X)
    labels = lb.inverse_transform(preds)
    return preds, labels


def slicing_metrics(model, df, feature, encoder, lb):
    """
    Compute metrics for a specific feature.
    Parameters
    ----------
    model: sklearn model
        Trained machine learning model.
    df: pd.DataFrame
        Data used for prediction, including features and labels.
    feature: str
        Feature to slice on.
    encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb: sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.

    Returns
    -------
    res: dict, {category: {precision: float, recall: float, fbeta: float}}
    """
    # check if feature is in categorical_features
    X, y, _, _ = process_data(
        df, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    preds = model.predict(X)
    X_feature = df[feature].copy()

    res = {}

    # Slice the data.
    for cat in X_feature.unique():
        y_slice = y[X_feature == cat]
        preds_slice = preds[X_feature == cat]
        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
        res[cat] = {"precision": precision, "recall": recall, "fbeta": fbeta}

    return res
