from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "nids_clean.csv"
MODELS_DIR = PROJECT_DIR / "models"

data = pd.read_csv(INPUT_FILE, low_memory=False)

X = data.drop(columns=["Label", "Target"])
X = X.select_dtypes(include=["number"]).astype("float32")
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Apprentissage uniquement avec le trafic normal
X_train_normal = X_train[y_train == 0]

model = IsolationForest(
    n_estimators=100,
    max_samples=10000,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

print("Entraînement Isolation Forest en cours...")
model.fit(X_train_normal)

# 1 = normal et -1 = anomalie dans Isolation Forest
predictions = model.predict(X_test)

# Conversion : 0 = Normal, 1 = Suspect
y_pred = (predictions == -1).astype(int)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nRapport :")
print(classification_report(y_test, y_pred, target_names=["Normal", "Suspect"]))

print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODELS_DIR / "isolation_forest_nids.pkl")

print("\nModèle sauvegardé : models/isolation_forest_nids.pkl")