
import math


def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


def sigmoid(z):
    if z < -700:
        return 0.0
    if z > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))
