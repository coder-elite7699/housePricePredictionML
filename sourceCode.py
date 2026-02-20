import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test["Id"]

train = train.drop("Id", axis=1)
test = test.drop("Id", axis=1)

train = train.fillna(train.median(numeric_only=True))
test = test.fillna(test.median(numeric_only=True))

train = pd.get_dummies(train, columns=["FurnishingType"], drop_first=True)
test = pd.get_dummies(test, columns=["FurnishingType"], drop_first=True)

train, test = train.align(test, join="left", axis=1, fill_value=0)
test = test.drop("SalePrice", axis=1, errors="ignore")

X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE:", rmse)

model.fit(X, y)

test_predictions = model.predict(test)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")
