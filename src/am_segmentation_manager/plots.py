"""Bar charts of the breakdowns :mod:`am_segmentation_manager.aggregate` counts.

**A deliberately small module, and it should stay small.** The package produces
counts; it draws charts only for breakdowns it already computes. Anything
bespoke -- a figure for a paper, an unusual comparison, a distribution rather
than a count -- belongs in a short script reading the exported CSV, which is
why that export exists. If a plotting request cannot be phrased as "bar chart
of a ``--by`` breakdown", it is not a job for this module.

matplotlib is an optional extra. Every statistics command works without it and
declines to plot, so the dependency is never on the path of anything but
drawing.
"""
from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

INSTALL_HINT = "Install it with: pip install 'am_segmentation_manager[plots]'"

#: Figure width per bar, and the floor below which a chart looks stretched.
_WIDTH_PER_BAR = 0.6
_MIN_WIDTH = 6.0
_HEIGHT = 4.5

#: Colours for a grouped chart. A qualitative set rather than a ramp: the
#: columns of a cross-tabulation are categories, not an ordered scale.
_SERIES_COLOURS = (
    "#4c78c8", "#1a7f6e", "#d08c00", "#a78bfa", "#6b6b6b", "#c44e52",
)


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


def _figure(bars: int):
    """Return a figure and axes sized to the number of bars."""
    import matplotlib

    # Chosen before pyplot is imported: a command-line export must not need a
    # display, and on a headless machine the default backend would fail.
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    width = max(_MIN_WIDTH, bars * _WIDTH_PER_BAR + 2.0)
    return plt.subplots(figsize=(width, _HEIGHT), layout="constrained")


def _finish(figure, output: Path) -> Path:
    """Write *figure* to *output*, in whichever format the suffix names."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    import matplotlib.pyplot as plt

    plt.close(figure)
    logger.info("Wrote %s", output)
    return output


def plot_counts(
    counts: Mapping[Hashable, int], title: str, output: Path, *, xlabel: str = ""
) -> Path:
    """Draw a one-dimensional breakdown as a bar chart.

    Args:
        counts: Group to count, as :func:`~am_segmentation_manager.aggregate.count_by`
            returns. Order is preserved, so the largest group is drawn first.
        title: Chart title.
        output: File to write. The suffix chooses the format -- ``.png`` for a
            figure to look at, ``.svg`` for one to put in a paper.
        xlabel: Axis label, usually the breakdown name.

    Raises:
        RuntimeError: If matplotlib is not installed.
    """
    require_plot_support()
    if not counts:
        raise ValueError("Nothing to plot")

    labels = [str(key) for key in counts]
    values = list(counts.values())
    total = sum(values)

    figure, axes = _figure(len(labels))
    bars = axes.bar(labels, values, color=_SERIES_COLOURS[0])
    # The question these charts answer is about balance, which a bare count
    # does not settle on its own.
    axes.bar_label(
        bars,
        labels=[f"{value}\n{100 * value / total:.1f}%" for value in values],
        padding=2, fontsize=8,
    )
    axes.set_title(f"{title}  (n={total})")
    axes.set_ylabel("images")
    if xlabel:
        axes.set_xlabel(xlabel)
    axes.margins(y=0.18)
    axes.spines[["top", "right"]].set_visible(False)
    if max((len(label) for label in labels), default=0) > 8:
        axes.tick_params(axis="x", labelrotation=30)
        for label in axes.get_xticklabels():
            label.set_horizontalalignment("right")
    return _finish(figure, output)


def plot_cross_tabulation(
    table: Mapping[tuple[Hashable, Hashable], int],
    row_title: str,
    column_title: str,
    output: Path,
) -> Path:
    """Draw a two-dimensional breakdown as a grouped bar chart.

    Grouped rather than stacked: the question is usually how the columns
    compare *within* each row, which stacked bars make hard to read past the
    first segment.

    Raises:
        RuntimeError: If matplotlib is not installed.
    """
    require_plot_support()
    if not table:
        raise ValueError("Nothing to plot")

    from .aggregate import totals

    rows, columns = totals(table)
    figure, axes = _figure(len(rows) * max(len(columns), 1))

    width = 0.8 / len(columns)
    positions = range(len(rows))
    for index, column in enumerate(columns):
        offset = (index - (len(columns) - 1) / 2) * width
        values = [table.get((row, column), 0) for row in rows]
        axes.bar(
            [position + offset for position in positions], values, width,
            label=str(column), color=_SERIES_COLOURS[index % len(_SERIES_COLOURS)],
        )

    axes.set_xticks(list(positions))
    axes.set_xticklabels([str(row) for row in rows])
    axes.set_title(f"{row_title} by {column_title}  (n={sum(table.values())})")
    axes.set_ylabel("images")
    axes.set_xlabel(row_title)
    axes.legend(title=column_title, frameon=False)
    axes.margins(y=0.12)
    axes.spines[["top", "right"]].set_visible(False)
    if max((len(str(row)) for row in rows), default=0) > 8:
        axes.tick_params(axis="x", labelrotation=30)
        for label in axes.get_xticklabels():
            label.set_horizontalalignment("right")
    return _finish(figure, output)
