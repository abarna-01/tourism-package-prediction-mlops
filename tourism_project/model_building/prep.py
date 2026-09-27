
import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the dataset
DATA_PATH = "tourism_project/data/tourism.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Remove unnecessary columns
# Unnamed: 0 -> accidental CSV index
# CustomerID -> unique identifier, not a predictive feature
columns_to_drop = ["Unnamed: 0", "CustomerID"]

df = df.drop(columns=columns_to_drop, errors="ignore")

print("Dataset shape after removing unnecessary columns:", df.shape)

# Separate features and target
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

# Stratified split to preserve the target class distribution
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save the splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("\nTrain/Test split completed successfully.")
print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape:", ytest.shape)

print("\nTraining target distribution:")
print(ytrain.value_counts(normalize=True))

print("\nTesting target distribution:")
print(ytest.value_counts(normalize=True))
