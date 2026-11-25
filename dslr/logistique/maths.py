"""
Fonctions mathématiques utilitaires pour la régression logistique.

Ce module contient les fonctions de base utilisées par l'algorithme :
- Produit scalaire (dot_product)
- Fonction sigmoid
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
        raise ValueError(f"Vecteurs de tailles différentes: {len(a)} vs {len(b)}")
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
