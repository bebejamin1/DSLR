
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
#    (ex: par paquets de 32)

import os

from tqdm import tqdm
from dslr.parsing.parser import take_theta_file, create_theta_file


# *****************************************************************************
# *                             AlgoDG()                                      *
# *                                                                           *

LEARNING_RATE = 0.01


class AlgoDG:

    def __init__(self, data: list):
        self.data = data
        self.len = len(data)

# ============================== TRAIN =======================================

    def train(self) -> None:
        os.system("clear")

        print("\n" + "TRAINING".center(79, " ") + "\n\n")
        algo_names = ["batch_gd", "mini_batch_gd", "stochastic_gd"]
        colours = ["blue", "white", "red"]

        for algo, c in zip(algo_names, colours):
            create_theta_file(0.0, 0.0, 0.0, 0.0, algo)

            print(f"{algo}()")
            for _ in tqdm(range(10000), desc="Training Process", colour=c):
                func = getattr(self, algo)
                theta1, theta2, theta3, theta4 = func()

                create_theta_file(theta1, theta2, theta3, theta4, algo)

            print(f"θserdaigle: {theta1:.4f} - θserpentard: {theta2:.4f} - "
                  f"θgryffondor: {theta3:.4f} - θpoufsouffle: {theta4:.4f}")
            print("\n\n")

        exit()

# ============================= BATCH GD ======================================

    def batch_gd(self) -> tuple:
        t1, t2, t3, t4 = take_theta_file("batch_gd")

        return (t1, t2, t3, t4)

# ========================== MINI BATCH GD ====================================

    def mini_batch_gd(self) -> tuple:
        t1, t2, t3, t4 = take_theta_file("mini_batch_gd")

        return (t1, t2, t3, t4)

# ========================== STOCHASTIC GD ====================================

    def stochastic_gd(self) -> tuple:
        t1, t2, t3, t4 = take_theta_file("stochastic_gd")

        return (t1, t2, t3, t4)


# *****************************************************************************
# *                            logreg_predict()                               *
# *                                                                           *

def predict(value: int) -> int:  # A DETERMINER NE PAS PRENDRE EN COMPTE
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
