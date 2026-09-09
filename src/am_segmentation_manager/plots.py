"""Charts of the breakdowns :mod:`am_segmentation_manager.aggregate` counts.

Two shapes, chosen by the question rather than by the user: one property is
bars, two are a heatmap. The heatmap is the one worth having -- a grouped bar
chart is unreadable past about four columns, and on a grid the *empty* cells
are the finding, being the combinations nothing has been acquired for.

**A deliberately small module, and it should stay small.** It draws the
breakdowns already computed. Anything bespoke belongs in a short script reading
the exported CSV, which is why that export exists.

Figures are built directly rather than through ``pyplot``, so nothing is
registered in a global list, no backend is chosen, and a command drawing a
chart over ssh never looks for a display. The window attaches its own canvas;
:func:`plot_counts` and :func:`plot_cross_tabulation` write a file.
"""
from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path

from .aggregate import totals

logger = logging.getLogger(__name__)

INSTALL_HINT = "Install it with: pip install 'am_segmentation_manager[plots]'"

#: Figure geometry, in inches. Bars grow downwards as groups are added; the
#: floors stop a two-bar chart from being drawn as a stripe.
_WIDTH = 9.0
_HEIGHT_PER_BAR = 0.34
_MIN_HEIGHT = 3.2
_HEIGHT_PER_ROW = 0.4
_WIDTH_PER_COLUMN = 1.15

#: Bar fill, and the line drawn over it for the running share.
_BAR_COLOUR = "#4c78c8"
_LINE_COLOUR = "#c44e52"

#: Fewest groups worth drawing a running share over. Below this the line says
#: nothing the two bars beside it do not.
_MIN_BARS_FOR_SHARE = 3


def has_plot_support() -> bool:
    """Return whether matplotlib is installed."""
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return False
    return True


def require_plot_support() -> None:
    """Raise a clear error if matplotlib is missing.

    Raises:
        RuntimeError: If matplotlib is not installed.
    """
    if not has_plot_support():
        raise RuntimeError(f"Plotting requires matplotlib. {INSTALL_HINT}")


def _new_figure(width: float, height: float):
    """Return a figure and its axes, built without pyplot."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), layout="constrained")
    return figure, figure.add_subplot()


def bar_figure(
    counts: Mapping[Hashable, int],
    title: str,
    *,
    xlabel: str = "",
    ordered: bool = False,
):
    """Draw a one-dimensional breakdown as horizontal bars.

    Horizontal because the groups are named things and the names are long:
    ``Helios 5 Hydra CX`` does not fit under a vertical bar at any font size
    worth reading.

    Args:
        counts: Group to count, in the order to draw them.
        title: Chart title.
        xlabel: What the groups are, named on the axis.
        ordered: Whether *counts* is already in a meaningful order -- area bins
            ascending, say. Ordered groups are a scale and keep their order,
            which is what makes the chart read as a distribution; unordered
            ones are drawn largest first and carry a running share, which is
            what answers how concentrated the collection is.

    Raises:
        RuntimeError: If matplotlib is not installed.
        ValueError: If there is nothing to draw.
    """
    require_plot_support()
    if not counts:
        raise ValueError("Nothing to plot")

    labels = [str(key) for key in counts]
    values = list(counts.values())
    total = sum(values)

    figure, axes = _new_figure(
        _WIDTH, max(_MIN_HEIGHT, len(labels) * _HEIGHT_PER_BAR + 1.6)
    )
    positions = range(len(labels))
    axes.barh(list(positions), values, color=_BAR_COLOUR)
    axes.set_yticks(list(positions))
    axes.set_yticklabels(labels)
    # Bars run top to bottom, because that is the order the table above the
    # chart lists them in and matplotlib's y-axis counts upwards.
    axes.invert_yaxis()

    for position, value in zip(positions, values):
        axes.text(
            value, position, f" {value} ({100 * value / total:.1f}%)",
            va="center", fontsize=8,
        )

    axes.set_title(f"{title}  (n={total})")
    axes.set_xlabel("images")
    if xlabel:
        axes.set_ylabel(xlabel)
    axes.margins(x=0.12)
    axes.spines[["top", "right"]].set_visible(False)

    if not ordered and len(labels) >= _MIN_BARS_FOR_SHARE:
        _add_running_share(axes, values, total)
    return figure


def _add_running_share(axes, values: Sequence[int], total: int) -> None:
    """Draw the cumulative share over bars already sorted largest first.

    The diversity question directly: where the line reaches 90% is how many
    groups the collection actually consists of, whatever the rest of the
    catalogue says it covers.
    """
    running = []
    carried = 0
    for value in values:
        carried += value
        running.append(100 * carried / total)

    share = axes.twiny()
    share.plot(running, list(range(len(values))), color=_LINE_COLOUR, marker="o",
               markersize=3, linewidth=1.2)
    share.set_xlim(0, 105)
    share.set_xlabel("cumulative share (%)", color=_LINE_COLOUR, fontsize=9)
    share.tick_params(axis="x", labelcolor=_LINE_COLOUR, labelsize=8)
    share.spines[["right", "left"]].set_visible(False)


def heatmap_figure(
    table: Mapping[tuple[Hashable, Hashable], int],
    row_title: str,
    column_title: str,
    *,
    title: str | None = None,
    row_order: Sequence[Hashable] = (),
    column_order: Sequence[Hashable] = (),
    normalise: bool = False,
):
    """Draw a two-dimensional breakdown as a grid of counts.

    Args:
        normalise: Colour each cell by its share of its own row rather than by
            the count. That answers a different question -- whether the mix is
            the same on every instrument, rather than where the data is -- and
            it is the one worth asking once one row dwarfs the others.

    Raises:
        RuntimeError: If matplotlib is not installed.
        ValueError: If there is nothing to draw.
    """
    require_plot_support()
    if not table:
        raise ValueError("Nothing to plot")

    import numpy as np


    rows, columns = totals(table, row_order, column_order)
    counts = np.array(
        [[table.get((row, column), 0) for column in columns] for row in rows],
        dtype=float,
    )

    shown = counts
    if normalise:
        row_totals = counts.sum(axis=1, keepdims=True)
        # A row with nothing in it has no shares; leave it at zero rather than
        # dividing by nothing and painting the whole row as not-a-number.
        shown = np.divide(
            100 * counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0
        )

    figure, axes = _new_figure(
        max(_WIDTH, len(columns) * _WIDTH_PER_COLUMN + 4.0),
        max(_MIN_HEIGHT, len(rows) * _HEIGHT_PER_ROW + 2.4),
    )
    # Blues puts white at zero, so a combination nothing was acquired for is a
    # blank cell rather than a dark one -- which is the point of the chart.
    #
    # Counts are shaded on a square root, because one row usually dwarfs the
    # rest -- "no crack" outnumbers every crack size put together -- and on a
    # linear ramp that one cell pushes everything else to white, hiding the
    # pattern the chart was opened to find. Shares already span 0 to 100 and
    # need no such help.
    from matplotlib.colors import Normalize, PowerNorm

    ceiling = float(shown.max()) or 1.0
    norm = (
        Normalize(vmin=0, vmax=ceiling) if normalise
        else PowerNorm(gamma=0.5, vmin=0, vmax=ceiling)
    )
    image = axes.imshow(shown, cmap="Blues", aspect="auto", norm=norm)

    axes.set_xticks(range(len(columns)))
    axes.set_xticklabels([str(column) for column in columns])
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([str(row) for row in rows])
    if max((len(str(column)) for column in columns), default=0) > 6:
        axes.tick_params(axis="x", labelrotation=30)
        for label in axes.get_xticklabels():
            label.set_horizontalalignment("right")

    for row in range(len(rows)):
        for column in range(len(columns)):
            value = shown[row, column]
            text = f"{value:.0f}%" if normalise else str(int(counts[row, column]))
            axes.text(
                column, row, text, ha="center", va="center", fontsize=8,
                # Measured off the shading rather than the value, so the switch
                # to white happens where the cell actually goes dark -- which
                # on a square-root ramp is not where the value is highest.
                color="white" if norm(value) > 0.6 else "#222222",
            )

    axes.set_xlabel(column_title)
    axes.set_ylabel(row_title)
    axes.set_title(title or f"{row_title} by {column_title}  (n={int(counts.sum())})")
    figure.colorbar(
        image, ax=axes, label="share of row (%)" if normalise else "images"
    )
    return figure


def _finish(figure, output: Path) -> Path:
    """Write *figure* to *output*, in whichever format the suffix names."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    logger.info("Wrote %s", output)
    return output


def plot_counts(
    counts: Mapping[Hashable, int],
    title: str,
    output: Path,
    *,
    xlabel: str = "",
    ordered: bool = False,
) -> Path:
    """Draw a one-dimensional breakdown and write it to *output*.

    Args:
        output: File to write. The suffix chooses the format -- ``.png`` for a
            figure to look at, ``.svg`` for one to put in a paper.

    Raises:
        RuntimeError: If matplotlib is not installed.
    """
    return _finish(bar_figure(counts, title, xlabel=xlabel, ordered=ordered), output)


def plot_cross_tabulation(
    table: Mapping[tuple[Hashable, Hashable], int],
    row_title: str,
    column_title: str,
    output: Path,
    *,
    row_order: Sequence[Hashable] = (),
    column_order: Sequence[Hashable] = (),
) -> Path:
    """Draw a two-dimensional breakdown and write it to *output*.

    Raises:
        RuntimeError: If matplotlib is not installed.
    """
    return _finish(
        heatmap_figure(
            table, row_title, column_title,
            row_order=row_order, column_order=column_order,
        ),
        output,
    )
