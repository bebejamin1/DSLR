
import json


def extract_numeric_features(data: list[dict]) -> dict[str, list[float]]:
    """Regroupe les valeurs numériques par colonne."""
    features = {}
    for row in data:
        for col_name, val in row.items():
            if isinstance(val, (int, float)):
                if col_name not in features:
                    features[col_name] = []
                features[col_name].append(float(val))
    return features

def calculate_mean(values: list[float]) -> float:
    """Calcule la moyenne arithmétique d'une liste de nombres."""
    if not values:
        return 0.0
    return sum(values) / len(values)

def calculate_std(values: list[float], mean: float) -> float:
    """Calcule l'écart-type d'échantillon d'une liste par rapport à sa moyenne."""
    n = len(values)
    if n <= 1:
        return 0.0
    squared_differences = 0.0
    for val in values:
        diff = val - mean
        squared_differences += diff * diff
    variance = squared_differences / (n - 1)
    std = variance ** 0.5
    return std

def describe(data: list) -> dict:
    """Calcule les statistiques descriptives pour les données fournies."""
    numeric_features = extract_numeric_features(data)
    described = {}
    for feat_name, values in numeric_features.items():
        values.sort()
        values_len = len(values)

        described[feat_name] = {
            "count": values_len,
            "mean": calculate_mean(values),
            "std": calculate_std(values, calculate_mean(values)),
            "min": values[0],
            "25%": values[int(values_len / 4)],
            "50%": values[int(values_len / 2)],
            "75%": values[int(values_len * 3 / 4)],
            "max": values[values_len - 1]
        }
    print(json.dumps(described, indent=4))
    return described