
import pandas as pd
import os

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

print("Dataset validation successful!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nColumns:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df["ProdTaken"].value_counts())

print("\nDataset summary:")
print(df.describe(include="all").transpose())
