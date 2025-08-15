import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("covtype.data", header=None)
df.columns = [f"feature_{i}" for i in range(df.shape[1] - 1)] + ["Cover_Type"]

X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))
print("XGBoost Report:")
print(classification_report(y_test, y_pred_xgb))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8,6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8,6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens')
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

importances = rf.feature_importances_
indices = importances.argsort()[-10:][::-1]
plt.figure(figsize=(8,6))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=45)
plt.title("Top 10 Important Features - Random Forest")
plt.tight_layout()
plt.show()

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("Best Random Forest Params:", grid_rf.best_params_)
print("Tuned Random Forest Score:", grid_rf.score(X_test, y_test))

param_grid_xgb = {
    "n_estimators": [50, 100],
    "max_depth": [4, 8]
}
grid_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42), param_grid_xgb, cv=3, n_jobs=-1)
grid_xgb.fit(X_train, y_train)
print("Best XGBoost Params:", grid_xgb.best_params_)
print("Tuned XGBoost Score:", grid_xgb.score(X_test, y_test))
