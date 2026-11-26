
import matplotlib.pyplot as plt

from dslr.maths.statistics import (
    compute_house_statistics,
    compute_subject_homogeneity_score,
    extract_subjects,
)
from dslr.visual.utils import HOUSES, HOUSE_COLORS, add_legend


def find_most_homogeneous_subject(
    subjects_marks: dict, subjects_houses: dict
) -> tuple[str, float]:
    """Trouve la matière dont la répartition des notes est la plus homogène
    entre les 4 maisons."""
    best_subject = None
    min_score = float("inf")
    for subject, marks in subjects_marks.items():
        house_means, house_stds = compute_house_statistics(
            subjects_houses, subject
        )
        score = compute_subject_homogeneity_score(
            marks, house_means, house_stds
        )
        if score < min_score:
            min_score = score
            best_subject = subject
    return best_subject, min_score


def _plot_subject_histogram(
    ax: plt.Axes,
    subject: str,
    houses_data: dict
) -> None:
    """Trace l'histogramme des notes des 4 maisons pour une matière donnée."""
    for house in HOUSES:
        marks = houses_data.get(subject, {}).get(house, [])
        ax.hist(
            marks,
            bins=25,
            alpha=0.5,
            label=house,
            color=HOUSE_COLORS.get(house),
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_title(subject, fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)


def histogram(data: dict) -> None:
    """Identifie et affiche l'histogramme du cours ayant une répartition
    homogène."""
    subjects_marks, subjects_houses = extract_subjects(data)
    best_subject, _ = find_most_homogeneous_subject(
        subjects_marks, subjects_houses
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_subject_histogram(ax, best_subject, subjects_houses)
    ax.set_title(
        f"Homogeneous Distribution: {best_subject}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Marks", fontsize=10)
    ax.set_ylabel("Number of Students", fontsize=10)
    add_legend(fig, ax, loc="lower right")
    plt.tight_layout()
    plt.show()
