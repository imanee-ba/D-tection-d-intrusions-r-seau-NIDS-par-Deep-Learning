# Détection d'intrusions réseau (NIDS) par Deep Learning

Système de détection d'intrusions réseau (Network Intrusion Detection System) qui classe le trafic réseau en **normal** ou **suspect**, à l'aide de modèles de deep learning et de modèles de machine learning classiques, avec un tableau de bord pour visualiser les résultats.

## Objectifs

- Classifier le trafic réseau normal et malveillant avec un **autoencodeur**.
- Comparer les modèles de deep learning à des approches classiques : **Random Forest**, **XGBoost** et **Isolation Forest**.
- Évaluer les modèles sur plusieurs jeux de données de référence.
- Fournir un **tableau de bord** qui classifie le trafic comme normal ou suspect.

## Datasets

| Dataset | Description |
|---|---|
| CIC-IDS2017 | Trafic réseau réalistes avec attaques (DoS, DDoS, brute force, etc.) |

> Les datasets ne sont pas inclus dans le dépôt en raison de leur taille. Télécharge-les depuis leurs sources officielles et place-les dans le dossier `data/` (voir la structure ci-dessous).

## Modèles

**Deep learning**
- Autoencodeur : détection d'anomalies par erreur de reconstruction, entraîné sur du trafic normal

**Machine learning (modèles de comparaison)**
- Random Forest
- XGBoost
- Isolation Forest

## Méthodologie

1. **Prétraitement** : nettoyage des données, gestion des valeurs manquantes et infinies, encodage des variables catégorielles, normalisation.
2. **Gestion du déséquilibre des classes** : rééquilibrage des données si nécessaire.
3. **Entraînement** des modèles sur chaque dataset.
4. **Évaluation et comparaison** avec les métriques suivantes : accuracy, precision, recall, F1-score, matrice de confusion, courbe ROC.
5. **Déploiement** du meilleur modèle dans un tableau de bord de classification du trafic.

## Résultats

Les résultats seront complétés après l'évaluation des modèles.

| Modèle | Dataset | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| Autoencodeur | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |
| Isolation Forest | | | | | |

## Tableau de bord

Le tableau de bord permet de :

- charger un fichier de trafic réseau,
- classifier chaque flux comme **normal** ou **suspect**,
- visualiser la répartition du trafic et les alertes détectées,
- consulter les performances des modèles.

## Technologies

- Python
- Pandas, NumPy, Matplotlib
- Scikit-learn, XGBoost
- TensorFlow / Keras (ou PyTorch)
- Flask 

## Structure du projet

```
nids-deep-learning/
├── data/               # Datasets (non versionnés)
├── notebooks/          # Exploration et expérimentations
├── src/
│   ├── preprocessing/  # Nettoyage et préparation des données
│   ├── models/         # autoencodeur, modèles ML
│   └── evaluation/     # Métriques et comparaison
├── dashboard/          # Application du tableau de bord
├── requirements.txt
└── README.md



