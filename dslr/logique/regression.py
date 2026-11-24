# aLED

# attribuer la bonne maison de Poudlard à chaque élève à partir de ses notes,
# avec une précision minimale de 98 %.

# logreg_predict
# calcules sa probabilité avec chacun des 4 modèles

# Probabilité Gryffondor : 0.12 (12 %)
# Probabilité Poufsouffle : 0.05 (5 %)
# Probabilité Serdaigle : 0.81 (81 %)
# Probabilité Serpentard : 0.02 (2 %)

############################################################

# logreg_train (ONE vs ALL ou ONE vs Rest) - reset theta a 0
# entraîner 4 classifieurs binaires indépendants

# Modèle 1 : Gryffondor (1) vs Toutes les autres maisons (0)
# Modèle 2 : Poufsouffle (1) vs Toutes les autres maisons (0)
# Modèle 3 : Serdaigle (1) vs Toutes les autres maisons (0)
# Modèle 4 : Serpentard (1) vs Toutes les autres maisons (0)
# 4 theta different

# Comment le modèle apprend ? (Coût et Descente de Gradient)
# utilise la Log Loss J(tetha) = ....

# 3 algo
# 1. Batch GD (Obligatoire) Regarde tous les élèves avant de faire un pas.
# 2. Stochastic GD (Bonus) Fait un pas après chaque élève pris individuellement
# 3. Mini-batch GD (Bonus) Fait un pas après chaque petit groupe d'élèves
# (ex: par paquets de 32)

import os
import tqdm

from dslr.parsing.parser import take_theta_file, create_theta_file


# *****************************************************************************
# *                         learning_display()                                *
# *                                                                           *

def logreg_train(data: list) -> None:  # descente stochastique ?? a faire
    # utilise des petit lot choisi aleatoirement a chaque iteration.
    "Permet de lancer les different algo de regression"
    os.system("clear")

    print("\n" + "learningRate()".center(79, " ") + "\n\n")
    create_theta_file(0.0, 0.0)

    for _ in tqdm(range(10000), desc="Training Process"):
        theta0, theta1 = learningRate(data)
        create_theta_file(theta0, theta1)

    print("\nEntraînement terminé! "
          f"Theta0: {theta0:.4f}, Theta1: {theta1:.4f}")
    exit()


# *****************************************************************************
# *                            logreg_predict()                               *
# *                                                                           *

def logreg_predict(value: int) -> int:  # A DETERMINER NE PAS PRENDRE EN COMPTE
    result = 0

    theta0, theta1 = take_theta_file()

    if (theta0 == 0 or theta1 == 0):
        print("\n" + "Lancez un learningRate avant tout!" + "\n"
              "Vos valeurs theta sont à 0")
        exit()

    normalized_value = value / 100000.0
    normalized_result = theta0 + (theta1 * normalized_value)
    result = normalized_result * 10000.0

    return (int(result))


# *****************************************************************************
# *                             learningRate()                                *
# *                                                                           *

LEARNING_RATE = 0.01


def learningRate(data) -> tuple:  # a ne pas prendre en compte pour linstant
    m = len(data)
    somme0 = 0.0
    somme1 = 0.0
    theta0, theta1 = take_theta_file()

    for row in data:
        try:
            km = row['km'] / 100000.0
            price = row['price'] / 10000.0

        except KeyError:
            print("\n" + "Le data csv ne convient pas, veuillez le remplacer")
            exit()

        estimate = theta0 + theta1 * km
        erreur = estimate - price
        somme0 += erreur
        somme1 += erreur * km

    tmp0 = LEARNING_RATE * (1 / m) * somme0
    tmp1 = LEARNING_RATE * (1 / m) * somme1

    theta0 = theta0 - tmp0
    theta1 = theta1 - tmp1

    return (theta0, theta1)
