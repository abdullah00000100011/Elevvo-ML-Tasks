# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(r"C:\Users\abdul\OneDrive\Desktop\Eleevo Pathway\Task 4\loan_approval_dataset.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

print("Columns in dataset:", df.columns)

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical features
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols.remove('loan_status')  # Exclude target column
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Encode target column
target_col = 'loan_status'
le_target = LabelEncoder()
df[target_col] = le_target.fit_transform(df[target_col])

# Split features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
import matplotlib.pyplot as plt
import seaborn as sns

feature_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title("Feature Importance")
plt.show()
