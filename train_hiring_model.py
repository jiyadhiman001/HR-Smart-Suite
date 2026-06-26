import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

DATA_PATH = "data/hiring_success.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

le = LabelEncoder()
df["EducationLevel"] = le.fit_transform(df["EducationLevel"])
encoders = {"EducationLevel": le}

X = df.drop(columns=["HiringSuccess"])
y = df["HiringSuccess"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300, max_depth=8, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Not Successful", "Successful"]))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

joblib.dump(model, f"{MODEL_DIR}/hiring_model.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/hiring_encoders.pkl")
joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/hiring_features.pkl")
print("Hiring model saved!")