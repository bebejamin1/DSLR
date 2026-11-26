
import itertools
import math
import matplotlib.pyplot as plt

from dslr.maths.statistics import extract_subjects
from dslr.visual.utils import (
    setup_figure_layout,
    _extract_pair_data,
    HOUSES,
    HOUSE_COLORS
)


def plot_scatter_points(
    ax: plt.Axes,
    house_points: dict[str, tuple[list[float], list[float]]],
    point_size: float = 4,
    alpha: float = 0.4,
) -> None:
    """Trace les points colorés par maison sur un sous-axe."""
    for house in HOUSES:
        xs, ys = house_points[house]
        ax.scatter(
            xs,
            ys,
            alpha=alpha,
            edgecolors="none",
            s=point_size,
            label=house,
            color=HOUSE_COLORS.get(house),
        )


def scatter_plot(students: list[dict]) -> None:
    """Affiche une grille de nuages de points pour toutes les paires de
    matières."""
    subjects_marks, _ = extract_subjects(students)
    subjects = list(subjects_marks.keys())
    pairs = list(itertools.combinations(subjects, 2))
    num_plots = len(pairs)
    if num_plots == 0:
        return
    cols = 10
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 14))
    axes = axes.flatten()
    for i, (feat_x, feat_y) in enumerate(pairs):
        house_points = _extract_pair_data(students, feat_x, feat_y)
        plot_scatter_points(axes[i], house_points, point_size=4)
        axes[i].set_title(f"{feat_x}\nvs {feat_y}", fontsize=6, pad=2)
        axes[i].tick_params(axis="both", which="both", labelsize=5, length=1)
    setup_figure_layout(
        fig,
        axes,
        num_plots,
        title="Scatter plots - All subject pairs",
        legend_loc="upper right",
    )
    plt.show()
