from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "nids_clean.csv"
MODELS_DIR = PROJECT_DIR / "models"

data = pd.read_csv(INPUT_FILE, low_memory=False)

X = data.drop(columns=["Label", "Target"])
y = data["Target"]

# Garde les caractéristiques numériques du trafic
X = X.select_dtypes(include=["number"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("Entraînement en cours...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nRapport :")
print(classification_report(y_test, y_pred, target_names=["Normal", "Suspect"]))

print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODELS_DIR / "random_forest_nids.pkl")

with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as file:
    json.dump(X.columns.tolist(), file)

print("\nModèle sauvegardé : models/random_forest_nids.pkl")