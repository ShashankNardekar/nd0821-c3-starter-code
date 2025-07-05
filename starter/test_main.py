from fastapi.testclient import TestClient

# Import the app from your main.py file
from main import app 

# Create a TestClient instance
client = TestClient(app)

def test_get_root():
    """
    Test the GET endpoint on the root path.
    """
    response = client.get("/")
    assert response.status_code == 200 # Must test status code [cite: 84]
    assert response.json() == {"greeting": "Welcome to the census income classifier API!"} # Must test contents [cite: 84]

def test_post_predict_low_income():
    """
    Test the POST endpoint for a prediction of '<=50K'.
    """
    # This data point is known to result in a '<=50K' prediction
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}

def test_post_predict_high_income():
    """
    Test the POST endpoint for a prediction of '>50K'.
    """
    # This data point is known to result in a '>50K' prediction
    data = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}