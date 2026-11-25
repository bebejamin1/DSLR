import math
import matplotlib.pyplot as plt
from dslr.maths.statistics import extract_subjects
from dslr.visual.utils import HOUSES, HOUSE_COLORS, setup_figure_layout


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
    """Affiche une grille d'histogrammes uniforme pour chaque matière."""
    subjects_marks, subjects_houses = extract_subjects(data)
    subjects = list(subjects_marks.keys())
    num_subjects = len(subjects)
    if num_subjects == 0:
        return
    cols = 4
    rows = math.ceil(num_subjects / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()
    for i in range(num_subjects):
        _plot_subject_histogram(axes[i], subjects[i], subjects_houses)
    setup_figure_layout(
        fig,
        axes,
        num_subjects,
        title="Distribution of marks by subject, by house",
        legend_loc="lower right",
    )
    plt.show()
