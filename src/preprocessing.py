from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "MachineLearningCSV"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "nids_clean.csv"

CHUNK_SIZE = 50_000

files = list(DATA_DIR.rglob("*.csv"))

if not files:
    raise FileNotFoundError("Aucun fichier CSV trouvé dans data/MachineLearningCSV/")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Le fichier est recréé à chaque lancement
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

total_rows = 0
class_counts = Counter()
first_write = True

print(f"{len(files)} fichier(s) trouvé(s).\n")

for file in files:
    print(f"Traitement : {file.name}")

    try:
        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False):
            chunk.columns = chunk.columns.str.strip()

            if "Label" not in chunk.columns:
                print("  Ignoré : colonne Label introuvable.")
                break

            # Nettoyage
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.dropna(inplace=True)

            chunk["Label"] = chunk["Label"].astype(str).str.strip()

            # 0 = Normal ; 1 = toutes les attaques
            chunk["Target"] = (chunk["Label"] != "BENIGN").astype(int)

            class_counts.update(chunk["Target"].tolist())
            total_rows += len(chunk)

            # Écriture progressive : évite de charger tout le dataset en mémoire
            chunk.to_csv(
                OUTPUT_FILE,
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )
            first_write = False

    except Exception as error:
        print(f"  Erreur dans {file.name} : {error}")

print("\nPrétraitement terminé.")
print(f"Nombre total de lignes : {total_rows}")
print(f"Normal   : {class_counts[0]}")
print(f"Suspect  : {class_counts[1]}")
print(f"\nFichier créé : {OUTPUT_FILE}")