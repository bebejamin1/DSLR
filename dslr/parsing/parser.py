import csv
import pickle
import os

from pathlib import Path


# *****************************************************************************
# *                               parser()                                    *
# *                                                                           *

def parser(filepath: str) -> list:

    try:
        dir = Path("data")
        dir.mkdir(exist_ok=True)
        dir = Path("img")
        dir.mkdir(exist_ok=True)

        data_list = []
        check = []

        if not (os.path.exists("data/theta.pkl")):
            create_theta_file(0.0, 0.0)

        with open(filepath, 'r') as fcsv:
            f = csv.DictReader(fcsv)

            for row in f:
                if (len(row) > 2 or len(row) < 2):
                    continue

                for k in row.keys():
                    check.append(k)

                if not (row.get("km") or row.get("price")):
                    continue

                km = row.get("km")
                price = row.get("price")

                try:
                    km = int(km)
                    price = int(price)

                except ValueError:
                    continue

                if (km < 0 or price < 0):
                    continue

                data_list.append({"km": km, "price": price})

    except FileNotFoundError:
        print("Fichier introuvable !" + "\n"
              "Veuillez mettre un fichier .csv avec un jeu de données dans"
              "le dossier data")
        exit()

    except Exception:
        print("\n" + "Le data.csv n'est pas conforme, les entêtes doivent être"
              " km et price" + "\n")
        exit()

    if (len(data_list) < 2):
        print("Le data.csv ne contient pas assez de données, minimum 2")
        exit()

    if (check[0] != "km" or check[1] != "price"):
        print("\n" + "Le header du data.csv nest pas conforme"
              f" {check[0], check[1]} -> km, price" + "\n")
        exit()

    return (data_list)


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
