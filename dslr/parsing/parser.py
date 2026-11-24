import csv
import pickle
import os
import sys

from pathlib import Path


# *****************************************************************************
# *                               parser()                                    *
# *                                                                           *

def _parse_value(val: str):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return val


def load_csv(filepath):
    try:
        with open(filepath, mode='r') as f:
            reader = csv.DictReader(f)
            dataset = []
            for row in reader:
                cleaned_row = {}
                for key, val in row.items():
                    if key != "Index":
                        cleaned_row[key] = _parse_value(val)
                dataset.append(cleaned_row)
            return dataset
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}", file=sys.stderr)
        sys.exit(1)

def parser(filepath):
    try:
        return load_csv(filepath)
    except FileNotFoundError:
        print("Fichier introuvable !" + "\n"
              "Veuillez mettre un fichier .csv avec un jeu de données dans"
              "le dossier data")
        exit(1)


# *****************************************************************************
# *                        save / load dataset pickle                         *
# *                                                                           *

def save_dataset_pickle(data, output_path="data/dataset.pkl") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_dataset_pickle(input_path="data/dataset.pkl"):
    if not os.path.exists(input_path):
        print(f"Fichier {input_path} introuvable.", file=sys.stderr)
        return None
    try:
        with open(input_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Erreur lors de la lecture du pickle {input_path} : {e}", file=sys.stderr)
        return None

# *****************************************************************************
# *                       create_theta_file()                                 *
# *                                                                           *

def create_theta_file(θserdaigle: float, θserpentard: float,
                      θgryffondor: float, θpoufsouffle: float,
                      name: str) -> None:

    dir = Path("data")
    dir.mkdir(exist_ok=True)

    content = {"θserdaigle": θserdaigle, "θserpentard": θserpentard,
               "θgryffondor": θgryffondor, "θpoufsouffle": θpoufsouffle}

    with open(f"data/{name}.pkl", "wb") as f:
        pickle.dump(content, f)


# *****************************************************************************
# *                        take_theta_file()                                  *
# *                                                                           *

def take_theta_file(name: str) -> tuple:
    if not (os.path.exists(f"data/{name}.pkl")):
        print("\n" + "Lancez un logreg_train avant tout !" + "\n"
              "Aucun fichier theta trouvé")
        exit()

    try:
        with open(f"data/{name}.pkl", 'rb') as f:
            c = pickle.load(f)

    except Exception:
        print("Le fichier est corrompu, relancer un logreg_train")
        exit()

    return (c["θserdaigle"], c["θserpentard"],
            c["θgryffondor"], c["θpoufsouffle"])
