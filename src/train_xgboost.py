from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "nids_clean.csv"
MODELS_DIR = PROJECT_DIR / "models"
data = pd.read_csv(INPUT_FILE, low_memory=False)
X = data.drop(columns=["Label", "Target"])
X = X.select_dtypes(include=["number"]).astype("float32")
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = XGBClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
print("Entraînement  XGBoosten cours...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nRapport :")
print(classification_report(y_test, y_pred, target_names=["Normal", "Suspect"]))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODELS_DIR / "xgboost_nids.pkl")
print("\nModèle sauvegardé : models/xgboost_nids.pkl")


