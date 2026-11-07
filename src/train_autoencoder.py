from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

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

# L'autoencodeur apprend seulement le trafic normal
X_train_normal = X_train[y_train == 0]

# Limite pratique pour éviter un entraînement trop long sur CPU
if len(X_train_normal) > 300000:
    X_train_normal = X_train_normal.sample(n=300000, random_state=42)

X_normal_train, X_normal_val = train_test_split(
    X_train_normal,
    test_size=0.10,
    random_state=42
)

# Normalisation entre 0 et 1
scaler = MinMaxScaler()
X_normal_train = scaler.fit_transform(X_normal_train)
X_normal_val = scaler.transform(X_normal_val)
X_test_scaled = scaler.transform(X_test)

# Architecture de l'autoencodeur
input_layer = Input(shape=(X_normal_train.shape[1],))
encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)
encoded = Dense(16, activation="relu")(encoded)

decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(decoded)
output_layer = Dense(X_normal_train.shape[1], activation="sigmoid")(decoded)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("Entraînement de l'autoencodeur en cours...")
model.fit(
    X_normal_train,
    X_normal_train,
    validation_data=(X_normal_val, X_normal_val),
    epochs=20,
    batch_size=512,
    callbacks=[early_stop],
    verbose=1
)

# Seuil d'anomalie : 99 % des flux normaux doivent rester normaux
val_reconstruction = model.predict(X_normal_val, batch_size=1024, verbose=0)
val_errors = np.mean(np.square(X_normal_val - val_reconstruction), axis=1)
threshold = np.percentile(val_errors, 99)

# Classification des flux de test
test_reconstruction = model.predict(X_test_scaled, batch_size=1024, verbose=0)
test_errors = np.mean(np.square(X_test_scaled - test_reconstruction), axis=1)
y_pred = (test_errors > threshold).astype(int)

print(f"\nSeuil d'anomalie : {threshold:.8f}")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nRapport :")
print(classification_report(y_test, y_pred, target_names=["Normal", "Suspect"]))

print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# Sauvegarde
MODELS_DIR.mkdir(exist_ok=True)
model.save(MODELS_DIR / "autoencoder_nids.keras")
joblib.dump(scaler, MODELS_DIR / "autoencoder_scaler.pkl")

with open(MODELS_DIR / "autoencoder_config.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "threshold": float(threshold),
            "feature_columns": X.columns.tolist()
        },
        file,
        indent=2
    )

print("\nAutoencodeur sauvegardé dans models/.")