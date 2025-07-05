import sys
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

# Add the nested 'starter' directory to the python path to import ml libs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'starter')))

# Now you can import from the ml folder
from ml.data import process_data
from ml.model import train_model, inference

@pytest.fixture(scope="session")
def sample_data():
    """Fixture to provide sample data for testing."""
    data = {
        "age": [39, 50, 38, 53, 28],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private", "Private"],
        "fnlgt": [77516, 83311, 215646, 234721, 338409],
        "education": ["Bachelors", "Bachelors", "HS-grad", "11th", "Bachelors"],
        "education-num": [13, 13, 9, 7, 13],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Handlers-cleaners", "Prof-specialty"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband", "Wife"],
        "race": ["White", "White", "White", "Black", "Black"],
        "sex": ["Male", "Male", "Male", "Male", "Female"],
        "capital-gain": [2174, 0, 0, 0, 0],
        "capital-loss": [0, 0, 0, 0, 0],
        "hours-per-week": [40, 13, 40, 40, 40],
        "native-country": ["United-States", "United-States", "United-States", "United-States", "Cuba"],
        "salary": [">50K", "<=50K", "<=50K", "<=50K", "<=50K"]
    }
    return pd.DataFrame(data)

def test_process_data(sample_data):
    """Test the data processing function."""
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
    assert X.shape[0] == 5
    assert y.shape[0] == 5

def test_train_model(sample_data):
    """Test the model training function."""
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, _, _ = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression)

def test_inference(sample_data):
    """Test the inference function."""
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    # Process with a fitted encoder and lb
    X_train, y_train, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)

    # Use the same data for inference for simplicity
    X_inference, _, _, _ = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    
    preds = inference(model, X_inference)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_inference.shape[0]