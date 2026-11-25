# attribuer la bonne maison de Poudlard à chaque élève à partir de ses notes,
# avec une précision minimale de 98 %.

# logreg_predict
# calcule sa probabilité avec chacun des 4 modèles

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
# utilise la Log Loss J(theta) = -1/m * sum[y_i * log(h_i) + (1-y_i) * log(1-h_i)]

# 3 algo
# 1. Batch GD (Obligatoire) Regarde tous les élèves avant de faire un pas.
# 2. Stochastic GD (Bonus) Fait un pas après chaque élève pris individuellement
# 3. Mini-batch GD (Bonus) Fait un pas après chaque petit groupe d'élèves

import os

from tqdm import tqdm
from dslr.parsing.parser import take_theta_file, create_theta_file
from dslr.logistique.maths import (
    sigmoid, dot_product, add_bias, 
    vectorized_sigmoid, log_loss, compute_gradient
)
from dslr.logistique.normaliser import normalise


# *****************************************************************************
# *                             Constants                                    *
# *                                                                           *

LEARNING_RATE = 0.1
N_ITERATIONS = 10000

# *****************************************************************************
# *                             AlgoDG()                                      *
# *                                                                           *


class AlgoDG:

    def __init__(self, data: list):
        self.data = data
        self.len = len(data)
        self.X = None
        self.y = None
        self.n_features = None
        self._prepare_data()

    def _prepare_data(self):
        """Prépare les données: extrait features et labels, normalise, ajoute bias."""
        if not self.data:
            raise ValueError("Data is empty")
        
        normalised_data = normalise(self.data)
        
        if not normalised_data:
            raise ValueError("Normalized data is empty")
        
        feature_keys = [k for k in normalised_data[0].keys() 
                       if k not in ['Index', 'Hogwarts House', 'First Name', 
                                   'Last Name', 'Birthday', 'Best Hand']]
        
        X = []
        y = []
        
        for student in normalised_data:
            row = {}
            for key in feature_keys:
                row[key] = student.get(key, 0.0)
            X.append(row)
            y.append(student.get('Hogwarts House', ''))
        
        self.X = add_bias(X)
        self.y = y
        self.n_features = len(self.X[0]) if self.X else 0
        self.feature_keys = feature_keys

    def _create_binary_labels(self, target_house: str) -> list:
        """Crée des labels binaires pour one-vs-all."""
        return [1.0 if house == target_house else 0.0 for house in self.y]

    def _initialize_theta(self, size: int) -> list:
        """Initialise theta à zéro."""
        return [0.0] * size

# ============================== TRAIN =======================================

    def train(self) -> None:
        os.system("clear")

        print("\n" + "TRAINING".center(79, " ") + "\n\n")
        algo_names = ["batch_gd"]
        colours = ["blue"]

        for algo, c in zip(algo_names, colours):
            theta_serdaigle = self._initialize_theta(self.n_features)
            theta_serpentard = self._initialize_theta(self.n_features)
            theta_gryffondor = self._initialize_theta(self.n_features)
            theta_poufsouffle = self._initialize_theta(self.n_features)
            
            create_theta_file(theta_serdaigle, theta_serpentard,
                             theta_gryffondor, theta_poufsouffle, algo)

            for _ in tqdm(range(N_ITERATIONS), desc=f"{algo}()", colour=c):
                func = getattr(self, algo)
                theta_serdaigle, theta_serpentard, theta_gryffondor, theta_poufsouffle = func(
                    theta_serdaigle, theta_serpentard, theta_gryffondor, theta_poufsouffle
                )

                create_theta_file(theta_serdaigle, theta_serpentard,
                                 theta_gryffondor, theta_poufsouffle, algo)

            print(f"θserdaigle: {theta_serdaigle[:5]}... - θserpentard: {theta_serpentard[:5]}...")
            print(f"θgryffondor: {theta_gryffondor[:5]}... - θpoufsouffle: {theta_poufsouffle[:5]}...")
            print("\n\n")

        exit()

# ============================= BATCH GD ======================================

    def batch_gd(self, theta_serdaigle: list, theta_serpentard: list,
                 theta_gryffondor: list, theta_poufsouffle: list) -> tuple:
        """
        Descente de gradient par lots (Batch Gradient Descent).
        Met à jour tous les thetas en utilisant tout le dataset à chaque itération.
        """
        y_serdaigle = self._create_binary_labels("Ravenclaw")
        y_serpentard = self._create_binary_labels("Slytherin")
        y_gryffondor = self._create_binary_labels("Gryffindor")
        y_poufsouffle = self._create_binary_labels("Hufflepuff")
        
        h_serdaigle = vectorized_sigmoid(self.X, theta_serdaigle)
        h_serpentard = vectorized_sigmoid(self.X, theta_serpentard)
        h_gryffondor = vectorized_sigmoid(self.X, theta_gryffondor)
        h_poufsouffle = vectorized_sigmoid(self.X, theta_poufsouffle)
        
        grad_serdaigle = compute_gradient(self.X, y_serdaigle, h_serdaigle)
        grad_serpentard = compute_gradient(self.X, y_serpentard, h_serpentard)
        grad_gryffondor = compute_gradient(self.X, y_gryffondor, h_gryffondor)
        grad_poufsouffle = compute_gradient(self.X, y_poufsouffle, h_poufsouffle)
        
        theta_serdaigle = [theta_serdaigle[i] - LEARNING_RATE * grad_serdaigle[i] 
                         for i in range(len(theta_serdaigle))]
        theta_serpentard = [theta_serpentard[i] - LEARNING_RATE * grad_serpentard[i] 
                          for i in range(len(theta_serpentard))]
        theta_gryffondor = [theta_gryffondor[i] - LEARNING_RATE * grad_gryffondor[i] 
                          for i in range(len(theta_gryffondor))]
        theta_poufsouffle = [theta_poufsouffle[i] - LEARNING_RATE * grad_poufsouffle[i] 
                           for i in range(len(theta_poufsouffle))]
        
        return (theta_serdaigle, theta_serpentard, theta_gryffondor, theta_poufsouffle)

# ========================== MINI BATCH GD ====================================

    def mini_batch_gd(self, theta_serdaigle: list, theta_serpentard: list,
                       theta_gryffondor: list, theta_poufsouffle: list) -> tuple:
        t1, t2, t3, t4 = take_theta_file("mini_batch_gd")

        return (t1, t2, t3, t4)

# ========================== STOCHASTIC GD ====================================

    def stochastic_gd(self, theta_serdaigle: list, theta_serpentard: list,
                       theta_gryffondor: list, theta_poufsouffle: list) -> tuple:
        t1, t2, t3, t4 = take_theta_file("stochastic_gd")

        return (t1, t2, t3, t4)


# *****************************************************************************
# *                            logreg_predict()                               *
# *                                                                           *

def predict(value: int) -> dict:

    algo_result = {"batch_gd": "", "mini_batch_gd": "", "stochastic_gd": ""}

    for algo in algo_result.keys():
        theta1, theta2, theta3, theta4 = take_theta_file(algo)

        algo_result[algo] = "gryffondor"

    return algo_result
