import pandas as pd

# Define file paths
original_path = 'data/census.csv'
cleaned_path = 'data/cleaned_census.csv'

print(f"Reading data from: {original_path}")
print(f"Saving cleaned data to: {cleaned_path}")

# Clean the file
with open(original_path, 'r') as infile, open(cleaned_path, 'w') as outfile:
    for line in infile:
        outfile.write(line.replace(" ", ""))

print("\nCleaning complete.")
print(f"Cleaned data saved to {cleaned_path}")