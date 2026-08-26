import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("Loading data...")

# Load dataset
df = pd.read_csv("final_data.csv")

# --------------------------------------------------
# 1. CHECK TARGET
# --------------------------------------------------

if "price" not in df.columns:
    print("ERROR: 'price' column was not found.")
    print("Available columns:")
    print(df.columns.tolist())
    exit()

target = "price"

# --------------------------------------------------
# 2. SELECT NUMERIC FEATURES
# --------------------------------------------------

# Exclude target and any columns that leak the target price
exclude_cols = [target, 'payment_value', 'Unnamed: 0']

features = [
    col for col in numeric_cols
    if col not in exclude_cols
]
# --------------------------------------------------
# 3. CHECK FEATURES
# --------------------------------------------------

if not features:
    print("ERROR: No numeric features found.")
    exit()

print("\nFeatures used by the model:")
for feature in features:
    print(" -", feature)

# --------------------------------------------------
# 4. PREPARE DATA
# --------------------------------------------------

X = df[features + [target]].dropna()

y = X[target]

X = X.drop(columns=[target])

print(f"\nTraining on {len(X):,} rows of real transactional data...")
print(f"Number of features: {len(features)}")

# --------------------------------------------------
# 5. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 7. EVALUATE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("\nModel trained successfully!")
print(f"Mean Absolute Error: {mae:.2f}")

# --------------------------------------------------
# 8. SAVE MODEL + FEATURES
# --------------------------------------------------

model_package = {
    "model": model,
    "features": features
}

with open("trained_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("\nModel saved successfully!")
print("File: trained_model.pkl")
print("\nSaved features:")

for feature in features:
    print(" -", feature)