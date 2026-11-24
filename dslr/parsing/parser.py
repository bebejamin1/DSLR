import csv
import pickle
import os
import sys

from pathlib import Path


# *****************************************************************************
# *                               parser()                                    *
# *                                                                           *

def _parse_value(val: str):
    if val is None or val == "" or val.lower() == "nan":
        return None
    try:
        return float(val)
    except ValueError:
        return val


def parser(filepath):
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
        print(
            f"Erreur lors de la lecture du pickle {input_path} : {e}",
            file=sys.stderr
        )
        return None


# *****************************************************************************
# *                       create_theta_file()                                 *
# *                                                                           *

def create_theta_file(theta0: float, theta1: float) -> None:

    dir = Path("data")
    dir.mkdir(exist_ok=True)

    content = {"theta0": theta0, "theta1": theta1}

    with open("data/theta.pkl", "wb") as f:
        pickle.dump(content, f)


# *****************************************************************************
# *                        take_theta_file()                                  *
# *                                                                           *

def take_theta_file() -> tuple:
    if not (os.path.exists("data/theta.pkl")):
        print("\n" + "Lancez un learningRate avant tout !" + "\n"
              "Aucun fichier theta trouvé")
        exit()

    try:
        with open("data/theta.pkl", 'rb') as f:
            c = pickle.load(f)

    except Exception:
        print("Le fichier est corrompu, relancer un learningRate")
        exit()

    return (c["theta0"], c["theta1"])
