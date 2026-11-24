import sys
import csv
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

def describe(data: list) -> dict:
    """Calcule les statistiques descriptives pour les données fournies."""
    numeric_features = extract_numeric_features(data)
    described = {}
    for feat_name, values in numeric_features.items():
        described[feat_name] = {
            "count": len(values),
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "25%": 0.0,
            "50%": 0.0,
            "75%": 0.0,
            "max": 0.0
        }
    print(json.dumps(described, indent=4))
    return described