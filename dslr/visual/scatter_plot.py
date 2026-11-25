import itertools
import math
import matplotlib.pyplot as plt
from dslr.maths.statistics import extract_subjects
from dslr.visual.utils import HOUSES, HOUSE_COLORS, setup_figure_layout


def _extract_pair_data(
    students: list[dict],
    feat_x: str,
    feat_y: str
) -> dict[str, tuple[list[float], list[float]]]:
    """Extrait les coordonnées (x, y) pour chaque maison pour deux matières."""
    house_points = {house: ([], []) for house in HOUSES}
    for student in students:
        house = student.get("Hogwarts House")
        val_x = student.get(feat_x)
        val_y = student.get(feat_y)

        if (
            house in house_points
            and isinstance(val_x, (int, float))
            and isinstance(val_y, (int, float))
        ):
            house_points[house][0].append(float(val_x))
            house_points[house][1].append(float(val_y))
    return house_points


def _plot_sub_scatter(
    ax: plt.Axes,
    house_points: dict[str, tuple[list[float], list[float]]],
    feat_x: str,
    feat_y: str
) -> None:
    """Trace le nuage de points pour une paire de matières sur un axe donné."""
    for house in HOUSES:
        xs, ys = house_points[house]
        ax.scatter(
            xs,
            ys,
            alpha=0.4,
            edgecolors="none",
            s=4,
            label=house,
            color=HOUSE_COLORS.get(house),
        )
    ax.set_title(f"{feat_x}\nvs {feat_y}", fontsize=6, pad=2)
    ax.tick_params(axis="both", which="both", labelsize=5, length=1)


def scatter_plot(students: list[dict]) -> None:
    """Affiche une grille de nuages de points pour toutes les paires de matières."""
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
        _plot_sub_scatter(axes[i], house_points, feat_x, feat_y)
    setup_figure_layout(
        fig,
        axes,
        num_plots,
        title="Scatter plots - All subject pairs",
        legend_loc="upper right",
    )
    plt.show()
