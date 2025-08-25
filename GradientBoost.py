import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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

# Fill missing values
train = train.fillna(0)
test = test.fillna(0)

# -------------------------
# 4. Prepare features and target
# -------------------------
y = train["Weekly_Sales"]
X = train.drop(columns=["Weekly_Sales"])

# -------------------------
# 5. One-Hot Encoding for categorical variables
# -------------------------
X = pd.get_dummies(X, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

# Align columns in train & test (important!)
X, test = X.align(test, join="left", axis=1, fill_value=0)

# -------------------------
# 6. Train/Test Split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 7. Gradient Boosting Model
# -------------------------
gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gbr.fit(X_train, y_train)

# Predictions
y_pred = gbr.predict(X_val)

# -------------------------
# 8. Evaluation
# -------------------------
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("✅ Gradient Boosting Results:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Error %: {(rmse / y_val.mean()) * 100:.2f}%")

# -------------------------
# 9. Predict on Test (for submission)
# -------------------------
test_preds = gbr.predict(test)
submission = pd.DataFrame({
    "Id": test["Id"] if "Id" in test.columns else np.arange(len(test)),
    "Weekly_Sales": test_preds
})

# Save to CSV
submission.to_csv("my_submission_gradient_boost.csv", index=False)
print("📂 Submission file saved as my_submission_gradient_boost.csv")
