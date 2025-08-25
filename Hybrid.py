import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load datasets
train = pd.read_csv("/Users/gunupatibhuvan/MyFiles/MachineLearning/RetailForecastProject/train.csv")
test = pd.read_csv("/Users/gunupatibhuvan/MyFiles/MachineLearning/RetailForecastProject/test.csv")
features = pd.read_csv("/Users/gunupatibhuvan/MyFiles/MachineLearning/RetailForecastProject/features.csv")
stores = pd.read_csv("/Users/gunupatibhuvan/MyFiles/MachineLearning/RetailForecastProject/stores.csv")

# 2. Merge datasets
train = train.merge(features, on=["Store", "Date"], how="left")
train = train.merge(stores, on="Store", how="left")
test = test.merge(features, on=["Store", "Date"], how="left")
test = test.merge(stores, on="Store", how="left")

# 3. Preprocessing
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

for df in [train, test]:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

train = train.drop(columns=["Date"])
test = test.drop(columns=["Date"])

# Fill missing values with 0
train = train.fillna(0)
test = test.fillna(0)

# 4. Prepare features and target
y = train["Weekly_Sales"]
X = train.drop(columns=["Weekly_Sales"])

# 5. One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)
test = pd.get_dummies(test, drop_first=True)
X, test = X.align(test, join="left", axis=1, fill_value=0)

# 6. Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Initialize models
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1)

# 8. Train models
print("Training Random Forest...")
rf.fit(X_train, y_train)

print("Training Gradient Boosting...")
gbr.fit(X_train, y_train)

print("Training XGBoost...")
xgb.fit(X_train, y_train)

# 9. Predict on validation set
pred_rf = rf.predict(X_val)
pred_gbr = gbr.predict(X_val)
pred_xgb = xgb.predict(X_val)

# 10. Ensemble predictions (weighted average)
# You can tweak weights based on performance
weights = [0.3, 0.3, 0.4]  # sum should be 1
ensemble_pred = (weights[0]*pred_rf) + (weights[1]*pred_gbr) + (weights[2]*pred_xgb)

# 11. Evaluate ensemble
rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
r2 = r2_score(y_val, ensemble_pred)

print("✅ Hybrid Ensemble Results:")
print("RMSE:", rmse)
print("R² Score:", r2)
print("Error %:", (rmse / y_val.mean()) * 100)

# 12. Predict on test data using ensemble
test_pred_rf = rf.predict(test)
test_pred_gbr = gbr.predict(test)
test_pred_xgb = xgb.predict(test)

test_ensemble_pred = (weights[0]*test_pred_rf) + (weights[1]*test_pred_gbr) + (weights[2]*test_pred_xgb)

# 13. Create submission DataFrame
submission = pd.DataFrame({
    "Id": test["Id"] if "Id" in test.columns else np.arange(len(test)),
    "Weekly_Sales": test_ensemble_pred
})

submission.to_csv("my_submission_ensemble.csv", index=False)
print("📂 Ensemble submission saved as my_submission_ensemble.csv")