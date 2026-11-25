"""
Fonctions mathématiques utilitaires pour la régression logistique.

Ce module contient les fonctions de base utilisées par l'algorithme :
- Produit scalaire (dot_product)
- Fonction sigmoid
- Ajout du bias
- Fonction de coût (log loss)
- Calcul du gradient
"""

import math


def dot_product(a: list, b: list) -> float:
    """
    Calcule le produit scalaire entre deux vecteurs.

    Args:
        a: Premier vecteur [a_0, a_1, ..., a_n]
        b: Deuxième vecteur [b_0, b_1, ..., b_n]

    Returns:
        Produit scalaire : a_0*b_0 + a_1*b_1 + ... + a_n*b_n

    Raises:
        ValueError: Si les vecteurs n'ont pas la même longueur
    """
    if len(a) != len(b):
        raise ValueError("Vecteurs de tailles différentes: "
                         f"{len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def sigmoid(z: float) -> float:
    """
    Fonction d'activation sigmoid : g(z) = 1 / (1 + e^(-z))

    Cette fonction "écrase" n'importe quelle valeur réelle entre 0 et 1,
    ce qui permet de l'interpréter comme une probabilité.

    Args:
        z: Valeur réelle quelconque

    Returns:
        Valeur entre 0 et 1

    Note:
        - Pour éviter des overflows numériques, on clampe z entre -700 et +700
        - sigmoid(-700) ≈ 0 et sigmoid(+700) ≈ 1
    """
    if z < -700:
        return 0.0
    if z > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def add_bias(X: list[dict]) -> list[list]:
    """
    Ajoute une colonne de bias (1) à chaque échantillon.

    Convertit une liste de dictionnaires (chaque dict = un étudiant)
    en une matrice X où chaque ligne est [1, feature_1, feature_2, ...]

    Args:
        X: Liste de dictionnaires avec les features (ex:
        [{"Arithmancy": 0.5, ...}, ...])

    Returns:
        Matrice X avec bias: [[1, f1, f2, ...], [1, f1, f2, ...], ...]
    """
    if not X:
        return []

    features = list(X[0].keys())
    matrix = []

    for sample in X:
        row = [1.0]  # bias term
        for feature in features:
            row.append(sample.get(feature, 0.0))
        matrix.append(row)

    return matrix


def vectorized_sigmoid(X: list[list], theta: list) -> list[float]:
    """
    Calcule sigmoid(X * theta) pour tous les échantillons.

    Args:
        X: Matrice des features (avec bias) de taille (m, n+1)
        theta: Vecteur des poids de taille (n+1,)

    Returns:
        Liste de probabilités h_theta(x) pour chaque échantillon
    """
    predictions = []
    for x in X:
        z = dot_product(x, theta)
        predictions.append(sigmoid(z))
    return predictions


def log_loss(y: list, h: list) -> float:
    """
    Calcule la fonction de coût (Log Loss / Binary Cross-Entropy).

    J(theta) = -1/m * sum[y_i * log(h_i) + (1-y_i) * log(1-h_i)]

    Args:
        y: Vecteur des labels binaires (0 ou 1) de taille m
        h: Vecteur des prédictions (probabilités entre 0 et 1) de taille m

    Returns:
        Valeur de la log loss

    Note:
        Ajoute une petite valeur epsilon pour éviter log(0)
    """
    m = len(y)
    if m == 0:
        return 0.0

    epsilon = 1e-15
    total = 0.0

    for yi, hi in zip(y, h):
        hi = max(epsilon, min(1 - epsilon, hi))
        term1 = yi * math.log(hi)
        term2 = (1 - yi) * math.log(1 - hi)
        total += (term1 + term2)

    return -total / m


def compute_gradient(X: list[list], y: list, h: list) -> list[float]:
    """
    Calcule le gradient de la fonction de coût pour la régression logistique.

    Gradient formula:
    dJ/dtheta_j = 1/m * sum[(h_i - y_i) * x_j_i] for all i

    Args:
        X: Matrice des features (avec bias) de taille (m, n+1)
        y: Vecteur des labels binaires de taille m
        h: Vecteur des prédictions de taille m

    Returns:
        Vecteur gradient de taille (n+1,)
    """
    m = len(y)
    if m == 0:
        return [0.0] * len(X[0])

    n_features = len(X[0])
    gradient = [0.0] * n_features

    for i in range(m):
        for j in range(n_features):
            gradient[j] += (h[i] - y[i]) * X[i][j]

    for j in range(n_features):
        gradient[j] /= m

    return gradient
