
FEATURE_NAME_WIDTH = 30
VALUE_WIDTH = 11
SEPARATOR_WIDTH = 2
BORDER_COLOUR = '\033[38;2;0;174;139m'
RESET = '\033[0m'
OPEN_PIPE = f"{BORDER_COLOUR}|{RESET} "
CLOSE_PIPE = f" {BORDER_COLOUR}|{RESET}"
HORIZONTAL_LINE = f"{BORDER_COLOUR}-{RESET}"


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
    """Calcule l'écart-type d'échantillon d'une liste par rapport à sa
    moyenne."""
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


def calculate_describe_values(data: list) -> dict:
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
    return described


def print_aligned(value, width: int) -> None:
    """Affiche une valeur précédée d'espaces pour atteindre une largeur totale
    fixe."""
    val_str = str(value)
    spaces = max(0, width - len(val_str))
    print(" " * spaces + val_str, end=CLOSE_PIPE)


def describe(data: list) -> None:
    """Calcule les statistiques descriptives pour les données fournies."""
    described = calculate_describe_values(data)
    first_feature_stats = next(iter(described.values()))
    num_stats = len(first_feature_stats)
    line_len = FEATURE_NAME_WIDTH + (VALUE_WIDTH + SEPARATOR_WIDTH) * num_stats
    border = OPEN_PIPE + HORIZONTAL_LINE * line_len + CLOSE_PIPE

    print(border)

    print(OPEN_PIPE, end="")
    print_aligned("", FEATURE_NAME_WIDTH)
    for stat_name in first_feature_stats.keys():
        print_aligned(stat_name, VALUE_WIDTH)

    print('\n' + border)

    for feat_name, values in described.items():
        print(OPEN_PIPE, end="")
        print_aligned(feat_name, FEATURE_NAME_WIDTH)
        for v in values.values():
            print_aligned(f"{v:.2f}", VALUE_WIDTH)
        print()

    print(border)
