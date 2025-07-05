import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Add the nested 'starter' directory to the python path to import ml libs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'starter')))

# Now you can import from the ml folder
from ml.data import process_data
from ml.model import compute_model_metrics, inference

# Load the cleaned data
data = pd.read_csv("data/cleaned_census.csv")

# Split data to get the same test set as in training
_, test = train_test_split(data, test_size=0.20, random_state=42)

# Load the trained model and processors
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

# Define categorical features
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Open the output file
with open("slice_output.txt", "w") as f:
    f.write("Model Performance on Slices of 'education' feature\n")
    f.write("="*50 + "\n")

    # Iterate over each unique value in the 'education' column [cite: 69]
    for cls in test["education"].unique():
        f.write(f"\nSlice for education = '{cls}'\n")

        # Create a slice of the test data
        temp_df = test[test["education"] == cls]

        # Process the slice
        X_slice, y_slice, _, _ = process_data(
            temp_df,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Get predictions if the slice is not empty
        if X_slice.shape[0] > 0:
            preds = inference(model, X_slice)
            precision, recall, f1 = compute_model_metrics(y_slice, preds)

            # Write metrics to file
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall: {recall:.4f}\n")
            f.write(f"  F1 Score: {f1:.4f}\n")
        else:
            f.write("  No data in this slice.\n")

print("Slice performance analysis complete. Results saved to slice_output.txt")