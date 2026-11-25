import math
import matplotlib.pyplot as plt
from dslr.maths.statistics import extract_subjects

HOUSES = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]

HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Slytherin": "#2a623d",
    "Ravenclaw": "#222f5b",
    "Hufflepuff": "#ecb939",
}


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


def _setup_figure_layout(
    fig: plt.Figure,
    axes: list[plt.Axes],
    num_subjects: int
) -> None:
    """Configure la légende, le titre global et supprime les axes superflus."""
    for i in range(num_subjects, len(axes)):
        fig.delaxes(axes[i])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower right",
        ncol=len(HOUSES),
        title="Maisons",
        fontsize=10,
        title_fontsize=11,
    )
    fig.suptitle(
        "Distribution of marks by subject, by house",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])


def histogram(data: dict) -> None:
    """Affiche une grille d'histogrammes uniforme pour chaque matière."""
    subjects_marks, subjects_houses = extract_subjects(data)
    subjects = list(subjects_marks.keys())
    num_subjects = len(subjects)
    cols = 4
    rows = math.ceil(num_subjects / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()
    for i in range(num_subjects):
        _plot_subject_histogram(axes[i], subjects[i], subjects_houses)
    _setup_figure_layout(fig, axes, num_subjects)
    plt.show()
