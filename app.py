# Importe Path pour gérer les chemins de fichiers.
from pathlib import Path

# Importe BytesIO pour télécharger le CSV depuis la mémoire.
from io import BytesIO

# Importe json pour lire les fichiers JSON.
import json

# Importe joblib pour charger les modèles enregistrés.
import joblib

# Importe NumPy pour les calculs numériques.
import numpy as np

# Importe Pandas pour lire les fichiers CSV.
import pandas as pd

# Importe les outils nécessaires de Flask.
from flask import Flask, render_template, request, send_file


# Crée l'application Flask.
app = Flask(__name__)

# Définit le dossier principal du projet.
PROJECT_DIR = Path(__file__).resolve().parent

# Définit le dossier des modèles.
MODELS_DIR = PROJECT_DIR / "models"


# Charge les colonnes utilisées pendant l'entraînement.
with open(MODELS_DIR / "feature_columns.json", "r", encoding="utf-8") as file:
    DEFAULT_FEATURE_COLUMNS = json.load(file)


# Définit les modèles disponibles dans le dashboard.
MODEL_CONFIGS = {
    "xgboost": {
        "label": "XGBoost",
        "file": "xgboost_nids.pkl",
        "type": "supervised",
        "metrics": {
            "accuracy": 99.93,
            "precision": 99.74,
            "recall": 99.88,
            "f1": 99.81
        }
    },
    "random_forest": {
        "label": "Random Forest",
        "file": "random_forest_nids.pkl",
        "type": "supervised",
        "metrics": {
            "accuracy": 99.89,
            "precision": 99.72,
            "recall": 99.74,
            "f1": 99.73
        }
    },
    "isolation_forest": {
        "label": "Isolation Forest",
        "file": "isolation_forest_nids.pkl",
        "type": "isolation",
        "metrics": {
            "accuracy": 85.95,
            "precision": 88.88,
            "recall": 32.69,
            "f1": 47.93
        }
    },
    "autoencoder": {
        "label": "Autoencodeur",
        "file": "autoencoder_nids.keras",
        "type": "autoencoder",
        "metrics": {
            "accuracy": 85.83,
            "precision": 88.83,
            "recall": 32.05,
            "f1": 47.10
        }
    }
}


# Définit les trois modèles affichés dans la comparaison.
COMPARISON_KEYS = [
    "random_forest",
    "xgboost",
    "isolation_forest"
]


# Crée un cache pour éviter de recharger les modèles.
MODEL_CACHE = {}

# Conserve le dernier CSV de résultats.
last_result_csv = None


# Transforme le label CIC-IDS2017 en nom d'attaque français.
def get_attack_type(label):

    # Transforme la valeur en texte minuscule.
    label = str(label).strip().lower()

    # Vérifie si le trafic est normal.
    if label == "benign":
        return "Trafic normal"

    # Vérifie si le trafic est une attaque DDoS.
    if "ddos" in label:
        return "DDoS — Déni de service distribué"

    # Vérifie si le trafic est une attaque DoS.
    if label.startswith("dos"):
        return "DoS — Déni de service"

    # Vérifie si le trafic est un scan de ports.
    if "portscan" in label:
        return "Balayage de ports"

    # Vérifie les attaques FTP-Patator, SSH-Patator ou Brute Force.
    if "patator" in label or "brute force" in label:
        return "Attaque par force brute"

    # Vérifie les attaques Web, XSS ou SQL Injection.
    if (
        "web attack" in label
        or "sql injection" in label
        or "xss" in label
    ):
        return "Attaque web"

    # Vérifie les attaques de type Botnet.
    if "bot" in label:
        return "Botnet"

    # Vérifie l'attaque Heartbleed.
    if "heartbleed" in label:
        return "Attaque Heartbleed"

    # Vérifie les attaques d'infiltration.
    if "infiltration" in label:
        return "Infiltration réseau"

    # Retourne le label original si aucune catégorie ne correspond.
    return str(label).title()


# Charge le modèle sélectionné par l'utilisateur.
def load_selected_model(model_key):

    # Retourne le modèle du cache s'il est déjà chargé.
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]

    # Récupère la configuration du modèle.
    config = MODEL_CONFIGS[model_key]

    # Vérifie si l'utilisateur a choisi l'autoencodeur.
    if config["type"] == "autoencoder":

        # Importe TensorFlow uniquement pour l'autoencodeur.
        from tensorflow.keras.models import load_model

        # Charge le modèle Deep Learning.
        model = load_model(MODELS_DIR / config["file"])

        # Charge le scaler utilisé durant l'entraînement.
        scaler = joblib.load(MODELS_DIR / "autoencoder_scaler.pkl")

        # Ouvre le fichier de configuration de l'autoencodeur.
        with open(
            MODELS_DIR / "autoencoder_config.json",
            "r",
            encoding="utf-8"
        ) as file:
            autoencoder_config = json.load(file)

        # Regroupe les fichiers nécessaires.
        artifacts = {
            "model": model,
            "scaler": scaler,
            "threshold": autoencoder_config["threshold"],
            "feature_columns": autoencoder_config["feature_columns"]
        }

    # Charge Random Forest, XGBoost ou Isolation Forest.
    else:

        # Charge le modèle choisi.
        model = joblib.load(MODELS_DIR / config["file"])

        # Regroupe le modèle et les colonnes.
        artifacts = {
            "model": model,
            "feature_columns": DEFAULT_FEATURE_COLUMNS
        }

    # Ajoute le modèle au cache.
    MODEL_CACHE[model_key] = artifacts

    # Retourne les éléments du modèle.
    return artifacts


# Lance la prédiction selon le type de modèle.
def predict_traffic(model_key, artifacts, X):

    # Récupère le type de modèle.
    model_type = MODEL_CONFIGS[model_key]["type"]

    # Récupère le modèle entraîné.
    model = artifacts["model"]

    # Cas XGBoost ou Forêt aléatoire.
    if model_type == "supervised":

        # Calcule le score de risque.
        risk_scores = model.predict_proba(X)[:, 1]

        # Classe les flux : 0 = Normal, 1 = Suspect.
        predictions = (risk_scores >= 0.5).astype(int)

    # Cas Forêt d'isolation.
    elif model_type == "isolation":

        # Isolation Forest retourne 1 pour normal et -1 pour anomalie.
        isolation_predictions = model.predict(X)

        # Transforme le résultat : 0 = Normal, 1 = Suspect.
        predictions = (isolation_predictions == -1).astype(int)

        # Calcule le score d'anomalie.
        anomaly_scores = -model.decision_function(X)

        # Transforme le score dans une échelle proche de 0 à 1.
        risk_scores = 1 / (
            1 + np.exp(-8 * anomaly_scores)
        )

    # Cas Autoencodeur.
    else:

        # Récupère le scaler.
        scaler = artifacts["scaler"]

        # Récupère le seuil d'anomalie.
        threshold = artifacts["threshold"]

        # Normalise les données.
        X_scaled = scaler.transform(X)

        # Reconstruit les données.
        reconstructed_data = model.predict(
            X_scaled,
            batch_size=1024,
            verbose=0
        )

        # Calcule l'erreur de reconstruction.
        reconstruction_errors = np.mean(
            np.square(X_scaled - reconstructed_data),
            axis=1
        )

        # Classe les flux selon le seuil.
        predictions = (
            reconstruction_errors > threshold
        ).astype(int)

        # Crée un score de risque entre 0 et 1.
        risk_scores = 1 - np.exp(
            -(reconstruction_errors / threshold)
        )

    # Retourne les prédictions et les scores.
    return predictions, risk_scores


# Crée la page principale.
@app.route("/", methods=["GET", "POST"])
def index():

    # Autorise la modification de la variable globale.
    global last_result_csv

    # Définit XGBoost comme modèle par défaut.
    selected_model_key = "xgboost"

    # Initialise les variables du template HTML.
    error = None
    suspect_results = None
    normal_results = None
    results = None
    stats = None

    # Vérifie si l'utilisateur a envoyé le formulaire.
    if request.method == "POST":

        # Récupère le modèle choisi.
        selected_model_key = request.form.get(
            "model_name",
            "xgboost"
        )

        # Vérifie que le modèle demandé existe.
        if selected_model_key not in MODEL_CONFIGS:
            selected_model_key = "xgboost"

        # Récupère le fichier CSV importé.
        uploaded_file = request.files.get("file")

        # Vérifie qu'un fichier est sélectionné.
        if not uploaded_file or uploaded_file.filename == "":
            error = "Veuillez sélectionner un fichier CSV."

        else:

            # Tente de lire et analyser le fichier.
            try:

                # Charge le modèle sélectionné.
                artifacts = load_selected_model(selected_model_key)

                # Récupère les colonnes nécessaires au modèle.
                feature_columns = artifacts["feature_columns"]

                # Lit le fichier CSV.
                data = pd.read_csv(uploaded_file, low_memory=False)

                # Supprime les espaces inutiles dans les noms de colonnes.
                data.columns = data.columns.str.strip()

                # Cherche les colonnes manquantes.
                missing_columns = [
                    column
                    for column in feature_columns
                    if column not in data.columns
                ]

                # Vérifie que le fichier possède les colonnes nécessaires.
                if missing_columns:

                    # Affiche une erreur si le CSV est incompatible.
                    error = (
                        f"Le fichier ne contient pas "
                        f"{len(missing_columns)} colonne(s) requise(s). "
                        "Utilisez un CSV CIC-IDS2017."
                    )

                else:

                    # Place les colonnes dans le même ordre que l'entraînement.
                    X = data.reindex(columns=feature_columns)

                    # Convertit les valeurs en données numériques.
                    X = X.apply(pd.to_numeric, errors="coerce")

                    # Remplace les valeurs infinies.
                    X.replace([np.inf, -np.inf], np.nan, inplace=True)

                    # Identifie les lignes valides.
                    valid_rows = X.notna().all(axis=1)

                    # Crée la colonne Prediction.
                    data["Prediction"] = "Données invalides"

                    # Crée la colonne Risk_score.
                    data["Risk_score"] = np.nan

                    # Vérifie qu'il y a des lignes valides.
                    if valid_rows.any():

                        # Lance les prédictions.
                        predictions, risk_scores = predict_traffic(
                            selected_model_key,
                            artifacts,
                            X.loc[valid_rows]
                        )

                        # Ajoute les scores de risque.
                        data.loc[
                            valid_rows,
                            "Risk_score"
                        ] = risk_scores

                        # Ajoute les prédictions normales ou suspectes.
                        data.loc[
                            valid_rows,
                            "Prediction"
                        ] = np.where(
                            predictions == 1,
                            "Suspect",
                            "Normal"
                        )

                    # Vérifie si le CSV possède les labels CIC-IDS2017.
                    if "Label" in data.columns:

                        # Ajoute le nom réel de l'attaque.
                        data["Attack_type"] = data["Label"].apply(
                            get_attack_type
                        )

                    else:

                        # Affiche un type non déterminé pour les fichiers non étiquetés.
                        data["Attack_type"] = np.where(
                            data["Prediction"] == "Normal",
                            "Trafic normal",
                            "Suspect — type inconnu"
                        )

                    # Compte les flux normaux.
                    normal_count = int(
                        (data["Prediction"] == "Normal").sum()
                    )

                    # Compte les flux suspects.
                    suspect_count = int(
                        (data["Prediction"] == "Suspect").sum()
                    )

                    # Compte le nombre total de flux.
                    total_count = len(data)

                    # Calcule les pourcentages si le CSV contient des lignes.
                    if total_count > 0:

                        # Calcule le pourcentage de trafic normal.
                        normal_percent = round(
                            normal_count / total_count * 100,
                            2
                        )

                        # Calcule le pourcentage de trafic suspect.
                        suspect_percent = round(
                            suspect_count / total_count * 100,
                            2
                        )

                    else:

                        # Évite une erreur de division par zéro.
                        normal_percent = 0
                        suspect_percent = 0

                    # Crée le dictionnaire de statistiques.
                    stats = {
                        "total": total_count,
                        "normal": normal_count,
                        "suspect": suspect_count,
                        "normal_percent": normal_percent,
                        "suspect_percent": suspect_percent
                    }

                    # Arrondit les scores pour l'affichage.
                    data["Risk_score"] = data["Risk_score"].round(4)

                    # Sépare les flux suspects.
                    suspect_data = data[
                        data["Prediction"] == "Suspect"
                    ]

                    # Trie les suspects du plus risqué au moins risqué.
                    suspect_data = suspect_data.sort_values(
                        "Risk_score",
                        ascending=False
                    )

                    # Prépare les 500 premiers suspects.
                    suspect_results = suspect_data.head(500).to_dict(
                        orient="records"
                    )

                    # Sépare les flux normaux.
                    normal_data = data[
                        data["Prediction"] == "Normal"
                    ]

                    # Trie les flux normaux.
                    normal_data = normal_data.sort_values(
                        "Risk_score",
                        ascending=True
                    )

                    # Prépare les 500 premiers flux normaux.
                    normal_results = normal_data.head(500).to_dict(
                        orient="records"
                    )

                    # Réunit les résultats normaux et suspects dans un seul tableau.
                    all_results = pd.concat([
                        suspect_data.head(250),
                        normal_data.head(250)
                    ])

                    # Trie le tableau unique par risque.
                    all_results = all_results.sort_values(
                        "Risk_score",
                        ascending=False
                    )

                    # Prépare les résultats pour index.html.
                    results = all_results.to_dict(
                        orient="records"
                    )

                    # Prépare le fichier CSV à télécharger.
                    last_result_csv = data.to_csv(
                        index=False
                    ).encode("utf-8")

            # Affiche l'erreur dans le navigateur.
            except Exception as exception:
                error = f"Erreur lors de l'analyse : {exception}"

    # Récupère les informations du modèle sélectionné.
    selected_model = MODEL_CONFIGS[selected_model_key]

    # Prépare les trois modèles pour le graphe de comparaison.
    comparison_models = [
        MODEL_CONFIGS[key]
        for key in COMPARISON_KEYS
    ]

    # Envoie les données à la page HTML.
    return render_template(
        "index.html",
        error=error,
        selected_model_key=selected_model_key,
        selected_model=selected_model,
        all_models=MODEL_CONFIGS,
        comparison_models=comparison_models,
        suspect_results=suspect_results,
        normal_results=normal_results,
        results=results,
        stats=stats
    )


# Crée une route pour télécharger les résultats.
@app.route("/download")
def download():

    # Vérifie qu'un fichier de résultats existe.
    if last_result_csv is None:
        return "Aucun résultat disponible.", 404

    # Télécharge le CSV de résultats.
    return send_file(
        BytesIO(last_result_csv),
        mimetype="text/csv",
        as_attachment=True,
        download_name="resultats_nids.csv"
    )


# Lance le site Flask.
if __name__ == "__main__":
    app.run(debug=True)