"""Plot posterior generation time summaries under the two infectiousness models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from scripts.inputs import get_inputs_generation_time

_MODEL_STYLES = {
    "independent": ("Independent transmission/symptoms", "#0072BD"),
    "mechanistic": ("Mechanistic", "#D95319"),
}
_PANELS = [  # quantity, axis label, upper axis limit
    ("generation_time_mean", "Mean generation time (days)", 10),
    (
        "generation_time_sd",
        "Standard deviation of generation\ntime distribution (days)",
        10,
    ),
    (
        "presymptomatic_proportion",
        "Proportion of transmissions\nbefore symptom onset",
        1,
    ),
    ("beta0", r"Overall infectiousness, $\beta_0$", 2.5),
]


def make_plots() -> None:
    """Violin plots comparing the posteriors of the generation time mean and SD, presymptomatic
    proportion and overall infectiousness between the models.

    Each violin marks the posterior median (white dot), interquartile range (thick line) and
    95% credible interval (thin line).
    """
    inputs = get_inputs_generation_time()
    posteriors = {
        name: pd.read_csv(path)
        for name, path in inputs["results_paths"]["posterior"].items()
    }
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for ax, label, (quantity, ylabel, ylim) in zip(axes.flat, "ABCD", _PANELS):
        for position, (name, posterior) in enumerate(posteriors.items(), start=1):
            _violin(
                ax, posterior[quantity].to_numpy(), position, _MODEL_STYLES[name][1]
            )
        ax.set(
            xlim=(0.4, len(posteriors) + 0.6),
            ylim=(0, ylim),
            xticks=[],
            yticks=np.linspace(0, ylim, 6),
            ylabel=ylabel,
        )
        ax.set_title(f"{label}.", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            Patch(facecolor=colour, edgecolor="0.4", label=label)
            for label, colour in (_MODEL_STYLES[name] for name in posteriors)
        ],
        loc="outside upper center",
        ncols=len(posteriors),
        frameon=False,
    )

    path = inputs["fig_paths"]["posterior_violins"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _violin(ax, values, position, colour):
    parts = ax.violinplot(values, [position], widths=0.7, showextrema=False)
    for body in parts["bodies"]:
        body.set(facecolor=colour, edgecolor="0.4", alpha=1)
    q = np.quantile(values, [0.025, 0.25, 0.5, 0.75, 0.975])
    ax.vlines(position, q[0], q[4], color="0.5", linewidth=1)
    ax.vlines(position, q[1], q[3], color="0.5", linewidth=4)
    ax.scatter(position, q[2], color="white", s=15, zorder=3)


if __name__ == "__main__":
    make_plots()
