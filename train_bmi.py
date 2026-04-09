import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
data = pd.read_csv("bmi_data.csv")

X = data[['weight', 'height']]
y = data['bmi']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= NN Model =================
nn = Sequential()
nn.add(Dense(8, activation='relu', input_dim=2))
nn.add(Dense(4, activation='relu'))
nn.add(Dense(1))

nn.compile(optimizer='adam', loss='mse')

nn.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

y_pred_nn = nn.predict(X_test).flatten()

# ================= Linear Regression =================
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

# ================= Metrics =================
print("\nNeural Network")
print("R2:", r2_score(y_test, y_pred_nn))
print("MAE:", mean_absolute_error(y_test, y_pred_nn))

print("\nLinear Regression")
print("R2:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))

# ================= Scatter Plot =================
plt.figure()
plt.scatter(y_test, y_pred_nn, label="NN")
plt.scatter(y_test, y_pred_lr, label="LR")

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])

plt.legend()
plt.title("Actual vs Predicted")
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.show()

# ================= Residual Plot =================
plt.figure()
plt.scatter(y_pred_nn, y_test - y_pred_nn, label="NN")
plt.scatter(y_pred_lr, y_test - y_pred_lr, label="LR")

plt.axhline(0)
plt.legend()
plt.title("Residual Plot")
plt.show()