# Script to train machine learning model.
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split

# Import functions from the ml folder
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# --- Load Data ---
# Path is now relative to the main 'starter' directory
print("Loading cleaned data...")
data = pd.read_csv("data/cleaned_census.csv")

# --- Train-Test Split ---
print("Splitting data...")
train, test = train_test_split(data, test_size=0.20, random_state=42)

# --- Process Data ---
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

print("Processing training data...")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

print("Processing test data...")
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# --- Train Model ---
print("Training model...")
model = train_model(X_train, y_train)

# --- Evaluate Model ---
print("Evaluating model...")
preds = inference(model, X_test)
precision, recall, f1 = compute_model_metrics(y_test, preds)

print(f"\n--- Model Performance ---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"-----------------------\n")

# --- Save Artifacts ---
# Path is now relative to the main 'starter' directory
model_dir = "model"
print(f"Saving model and artifacts to {model_dir}...")
os.makedirs(model_dir, exist_ok=True) 
joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
joblib.dump(encoder, os.path.join(model_dir, 'encoder.pkl'))
joblib.dump(lb, os.path.join(model_dir, 'lb.pkl'))

print("Artifacts saved successfully.")