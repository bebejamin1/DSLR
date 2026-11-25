"""Fonctions utilitaires et constantes partagées pour les visualisations DSLR."""

import matplotlib.pyplot as plt

HOUSES = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]

HOUSE_COLORS = {
    "Gryffindor": "#ae0001",
    "Slytherin": "#2a623d",
    "Ravenclaw": "#222f5b",
    "Hufflepuff": "#ecb939",
}


def add_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    loc: str = "upper right",
    fontsize: int = 9,
    title_fontsize: int = 10,
) -> None:
    """Ajoute une légende globale commune aux figures basée sur le premier sous-graphe."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc=loc,
        ncol=len(HOUSES),
        title="Houses",
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )


def setup_figure_layout(
    fig: plt.Figure,
    axes: list[plt.Axes],
    num_plots: int,
    title: str,
    legend_loc: str = "upper right",
) -> None:
    """Supprime les axes inutilisés, ajoute la légende commune, le titre et ajuste les marges."""
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    add_legend(fig, axes[0], loc=legend_loc)
    fig.suptitle(
        title,
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
