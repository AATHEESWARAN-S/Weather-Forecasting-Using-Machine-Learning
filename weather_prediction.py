import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")

X = data[["MinTemp", "Humidity"]]
y = data["MaxTemp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predicted = model.predict(X_test)

plt.plot(predicted, label="Predicted Temperature")
plt.plot(y_test.values, label="Actual Temperature")
plt.legend()
plt.title("Weather Forecast Comparison")
plt.show()

print("Prediction Completed ✔")
