from LinearRegression import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

data = pd.read_csv('../possum.csv')
data = data.dropna()

# Data Preprocessing
X = data.drop(columns=['age'])
y = data['age']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# From scratch implementation
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("From Scratch Implementation:")
print(f"Root Mean Squared error: {np.sqrt(mean_squared_error(y_test, y_pred))}\n")

sklearn_model = SklearnLinearRegression()
sklearn_model.fit(X_train, y_train)
y_sklearn_pred = sklearn_model.predict(X_test)

print("Scikit-learn Implementation:")
print(f"Root Mean Squared error: {np.sqrt(mean_squared_error(y_test, y_sklearn_pred))}")

