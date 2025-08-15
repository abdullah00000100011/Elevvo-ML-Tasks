import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("StudentsPerformance.csv")
df["average score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
X = df.drop(columns=["math score", "reading score", "writing score", "average score", "gender", "lunch"])
y = df["average score"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression:")
print(f"  MSE: {mse_linear:.2f}")
print(f"  R²:  {r2_linear:.2f}")

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
y_pred_poly = poly_model.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nPolynomial Regression (Degree 2):")
print(f"  MSE: {mse_poly:.2f}")
print(f"  R²:  {r2_poly:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, color='blue', alpha=0.6, label="Linear")
plt.scatter(y_test, y_pred_poly, color='green', alpha=0.4, label="Polynomial (deg 2)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Average Score")
plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted: Linear vs Polynomial")
plt.legend()
plt.grid(True)
plt.show()
