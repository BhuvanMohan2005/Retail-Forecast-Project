import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # Make sure OpenMP runtime is installed on Mac: brew install libomp
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# 1. Load the datasets
# -------------------------
train = pd.read_csv("/Users/gunupatibhuvan/MyFiles/MachineLearning/RetailForecastProject/train.csv")
test = pd.read_csv("/Users/gunupatibhuvan/MyFiles/MachineLearning/RetailForecastProject/test.csv")
features = pd.read_csv("/Users/gunupatibhuvan/MyFiles/MachineLearning/RetailForecastProject/features.csv")
stores = pd.read_csv("/Users/gunupatibhuvan/MyFiles/MachineLearning/RetailForecastProject/stores.csv")
sampleSubmission = pd.read_csv("/Users/gunupatibhuvan/MyFiles/MachineLearning/RetailForecastProject/sampleSubmission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# -------------------------
# 2. Merge datasets
# -------------------------
train = train.merge(features, on=["Store", "Date"], how="left")
train = train.merge(stores, on="Store", how="left")

test = test.merge(features, on=["Store", "Date"], how="left")
test = test.merge(stores, on="Store", how="left")

# -------------------------
# 3. Preprocessing
# -------------------------
# Convert Date to datetime
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

# Extract useful time features
for df in [train, test]:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Drop Date column
train = train.drop(columns=["Date"])
test = test.drop(columns=["Date"])

# -------------------------
# 4. Prepare features and target
# -------------------------
y = train["Weekly_Sales"]
X = train.drop(columns=["Weekly_Sales"])

# Drop rows with NaN in target (if any)
valid_idx = y.notna()
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# -------------------------
# 5. Fill missing values (numeric columns only)
# -------------------------
X_numeric = X.select_dtypes(include=np.number)
median_vals = X_numeric.median()
X.loc[:, X_numeric.columns] = X_numeric.fillna(median_vals)

test_numeric = test.select_dtypes(include=np.number)
median_test = test_numeric.median()
test.loc[:, test_numeric.columns] = test_numeric.fillna(median_test)

# For any remaining non-numeric missing values, fill with 0 (or choose appropriate method)
X = X.fillna(0)
test = test.fillna(0)

# -------------------------
# 6. One-Hot Encoding for categorical variables
# -------------------------
X = pd.get_dummies(X, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

# Align columns in train & test (important!)
X, test = X.align(test, join="left", axis=1, fill_value=0)

# -------------------------
# 7. Train/Test Split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 8. XGBoost Model
# -------------------------
model = XGBRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    verbosity=1,
    tree_method="hist"  # hist is usually faster, but you can omit this or change based on your setup
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)

# -------------------------
# 9. Evaluation
# -------------------------
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("✅ XGBoost Results:")
print("RMSE:", rmse)
print("R² Score:", r2)
print("Error %:", (rmse / y_val.mean()) * 100)

# -------------------------
# 10. Predict on Test (for submission)
# -------------------------
test_preds = model.predict(test)
submission = pd.DataFrame({
    "Id": sampleSubmission["Id"],  # Using the original sampleSubmission IDs
    "Weekly_Sales": test_preds
})

# Save to CSV
submission.to_csv("my_submission.csv", index=False)
print("📂 Submission file saved as my_submission.csv")
