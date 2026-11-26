
import matplotlib.pyplot as plt

from dslr.maths.statistics import extract_subjects
from dslr.visual.histogram import _plot_subject_histogram
from dslr.visual.scatter_plot import plot_scatter_points
from dslr.visual.utils import setup_figure_layout, _extract_pair_data


def _configure_cell_axes(
    ax: plt.Axes,
    label_x: str = None,
    label_y: str = None
) -> None:
    """Configure les étiquettes extérieures et désactive les graduations
    intérieures."""
    if label_x:
        ax.set_xlabel(label_x, fontsize=5, labelpad=1)
    if label_y:
        ax.set_ylabel(label_y, fontsize=5, labelpad=1)
    ax.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False
    )


def pair_plot(students: list[dict]) -> None:
    """Affiche la matrice complète de pair plot pour toutes les matières."""
    subjects_marks, subjects_houses = extract_subjects(students)
    subjects = list(subjects_marks.keys())
    n = len(subjects)
    if n == 0:
        return
    fig, axes = plt.subplots(n, n, figsize=(25, 14))
    for row in range(n):
        for col in range(n):
            ax = axes[row][col]
            if row == col:
                _plot_subject_histogram(ax, subjects[row], subjects_houses)
                ax.set_title("")
            else:
                house_points = _extract_pair_data(
                    students,
                    subjects[col],
                    subjects[row]
                )
                plot_scatter_points(ax, house_points, point_size=2, alpha=0.4)
            label_x = subjects[col] if row == n - 1 else None
            label_y = subjects[row] if col == 0 else None
            _configure_cell_axes(ax, label_x, label_y)
    setup_figure_layout(
        fig,
        axes.flatten(),
        num_plots=n * n,
        title="Pair Plot Matrix - All Subjects by House",
        legend_loc="upper right",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()
