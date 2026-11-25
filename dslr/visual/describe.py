from dslr.maths.statistics import compute_statistics

SUBJECT_WIDTH = 30
STAT_VALUE_WIDTH = 13
SEPARATOR_WIDTH = 2
BORDER_COLOUR = '\033[38;2;0;174;139m'
RESET = '\033[0m'
PIPE = '|'
OPEN_PIPE = f"{BORDER_COLOUR}{PIPE}{RESET} "
CLOSE_PIPE = f" {BORDER_COLOUR}{PIPE}{RESET}"
HORIZONTAL_LINE = f"{BORDER_COLOUR}-{RESET}"


def print_aligned(value, width: int) -> None:
    """Affiche une valeur précédée d'espaces pour atteindre une largeur
    totale fixe."""
    val_str = str(value)
    spaces = max(0, width - len(val_str))
    print(" " * spaces + val_str, end=CLOSE_PIPE)


def describe(students: list[dict]) -> None:
    """Affiche les statistiques des matières de Poudlard sous forme de tableau.
    """
    subjects_statistics = compute_statistics(students)
    first_subject_stats = next(iter(subjects_statistics.values()))
    num_stats = len(first_subject_stats)
    line_length = (SUBJECT_WIDTH + (STAT_VALUE_WIDTH +
                                    SEPARATOR_WIDTH) * num_stats)
    border = OPEN_PIPE + HORIZONTAL_LINE * line_length + CLOSE_PIPE

    print(border)
    print(OPEN_PIPE, end="")
    print_aligned(" ",  (len(subjects_statistics) + 1 - 4) *
                  (STAT_VALUE_WIDTH + SEPARATOR_WIDTH))
    print_aligned("means",  (len(subjects_statistics) - 9) *
                  (STAT_VALUE_WIDTH + SEPARATOR_WIDTH) - SEPARATOR_WIDTH)

    print('\n' + border)

    print(OPEN_PIPE, end="")
    print_aligned("", SUBJECT_WIDTH)
    for stat_name in first_subject_stats.keys():
        print_aligned(stat_name, STAT_VALUE_WIDTH)

    print('\n' + border)

    for subject, stats in subjects_statistics.items():
        print(OPEN_PIPE, end="")
        print_aligned(subject, SUBJECT_WIDTH)
        for stat_value in stats.values():
            print_aligned(f"{stat_value:.2f}", STAT_VALUE_WIDTH)
        print()

    print(border)
