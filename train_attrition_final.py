import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score
)
from xgboost import XGBClassifier

DATA_PATH = "data/attrition.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

categorical_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=42
)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1s[:-1])
best_threshold = float(thresholds[best_idx])

y_pred = (y_proba >= best_threshold).astype(int)
print(f"Threshold: {best_threshold:.3f}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

joblib.dump(model, f"{MODEL_DIR}/attrition_model.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/attrition_encoders.pkl")
joblib.dump(X.columns.tolist(), f"{MODEL_DIR}/attrition_features.pkl")
joblib.dump(best_threshold, f"{MODEL_DIR}/attrition_threshold.pkl")
print("Attrition model saved!")