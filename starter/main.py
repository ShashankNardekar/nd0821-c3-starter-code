# Put the code for your API here.
import sys
import os
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Add the nested 'starter' directory to the python path to import ml libs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'starter')))

# Now you can import from the ml folder
from ml.data import process_data
from ml.model import inference

# Pydantic model for input data
class CensusData(BaseModel):
    """Pydantic model for the input data for a single prediction."""
    age: int
    workclass: str
    fnlgt: int
    education: str
    # Use alias to handle feature names with hyphens [cite: 25]
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        """Pydantic config with an example for the API docs."""
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "age": 45,
                "workclass": "Private",
                "fnlgt": 215646,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Prof-specialty",
                "relationship": "Wife",
                "race": "White",
                "sex": "Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

# Initialize FastAPI app
app = FastAPI()

# Load model artifacts on startup
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

# Define categorical features
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]


@app.get("/")
async def root():
    """GET endpoint for the root, giving a welcome message."""
    return {"greeting": "Welcome to the census income classifier API!"}


@app.post("/predict/")
async def predict(data: CensusData):
    """POST endpoint to make model predictions."""
    # Convert Pydantic model to a DataFrame
    input_df = pd.DataFrame([data.dict(by_alias=True)])

    # Process the data
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Get model prediction
    prediction = inference(model, X)

    # Convert prediction back to original label
    predicted_label = lb.inverse_transform(prediction)[0]

    return {"prediction": predicted_label}