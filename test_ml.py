import pytest
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from ml.data import apply_label
from ml.model import compute_model_metrics, inference, train_model


def test_apply_label():
    """Tests that apply_label maps binary predictions to salary strings."""
    assert apply_label(np.array([1])) == ">50K"
    assert apply_label(np.array([0])) == "<=50K"



def test_train_and_inference():
    """Tests that train_model returns a GradientBoostingClassifier and inference returns expected predictions."""
    X_train = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, GradientBoostingClassifier)
    preds = inference(model, X_train)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y_train.shape



def test_compute_model_metrics():
    """Tests that compute_model_metrics returns expected precision, recall, and F1."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0
    assert recall == 0.5
    assert np.isclose(fbeta, 2 / 3)

