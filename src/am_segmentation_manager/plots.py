"""Charts of the breakdowns :mod:`am_segmentation_manager.aggregate` counts, and of training runs.

One property is bars and two are a heatmap, whose *empty* cells are the finding; a
training run is its curves, epoch by epoch, and several are a table shaded by value.
**A deliberately small module**: it draws
what is already computed or read, and anything bespoke belongs in a script. Figures are
built without ``pyplot``, so nothing registers globally, no backend is chosen, and a
chart drawn over ssh never looks for a display. The window attaches its own canvas; the
``plot_*`` functions write a file, refusing to replace one unless told to.
"""
from __future__ import annotations

import logging
from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .aggregate import totals
from .io.atomic import write_atomically
from .io.tables import refuse_existing
from .training_compare import DEFAULT_COLOUR_RANGE, VALUE_CAPTIONS
from .training_runs import PER_CLASS_METRICS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .training_compare import Comparison
    from .training_runs import TrainingRun

logger = logging.getLogger(__name__)

INSTALL_HINT = "Install it with: pip install 'adaptive-milling-seg[plots]'"

#: Figure geometry, in inches. Bars grow downwards as groups are added; the
#: floors stop a two-bar chart from being drawn as a stripe.
_WIDTH = 9.0
_HEIGHT_PER_BAR = 0.34
_MIN_HEIGHT = 3.2
_HEIGHT_PER_ROW = 0.4
_WIDTH_PER_COLUMN = 1.15

#: Bar fill, and the line drawn over it for the running share.
#: The ramp a comparison of runs is shaded in.
COMPARISON_PALETTE = "YlGn"

_BAR_COLOUR = "#4c78c8"
_CHOSEN_COLOUR = "#7a3fa0"
_LINE_COLOUR = "#c44e52"
#: Behind the epochs a run's encoder was frozen for: pale enough to read a curve over.
_FROZEN_COLOUR = "#e9e9ee"

#: How a grid can be shaded: by the count in each cell, or by the cell's share
#: of its own row or of its own column.
SHADINGS: tuple[str, ...] = ("count", "row", "column")

#: A cell nothing was counted in. Grey rather than the palette's floor, which is
#: white: against white a 1 and a 0 look alike, and the empty cells are the
#: combinations nothing was acquired for -- the finding the grid is drawn for.
ZERO_COLOUR = "#d9dce1"

#: What the colour bar says it measures, for each shading.
_SHADING_LABELS = {
    "count": "images",
    "row": "share of row (%)",
    "column": "share of column (%)",
}

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


def _new_figure(width: float, height: float) -> tuple[Figure, Axes]:
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
) -> Figure:
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


def _add_running_share(axes: Axes, values: Sequence[int], total: int) -> None:
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
    shading: str = "count",
) -> Figure:
    """Draw a two-dimensional breakdown as a grid of counts.

    Args:
        shading: One of :data:`SHADINGS`. ``"row"`` colours each cell by its
            share of its own row, which asks whether the mix is the same on
            every instrument rather than where the data is -- the question worth
            asking once one row dwarfs the others. ``"column"`` asks it the
            other way round.

    Raises:
        RuntimeError: If matplotlib is not installed.
        ValueError: If there is nothing to draw, or *shading* is unknown.
    """
    require_plot_support()
    if shading not in SHADINGS:
        raise ValueError(f"Unknown shading {shading!r}; choose from {', '.join(SHADINGS)}")
    if not table:
        raise ValueError("Nothing to plot")

    import numpy as np

    rows, columns = totals(table, row_order, column_order)
    counts = np.array(
        [[table.get((row, column), 0) for column in columns] for row in rows],
        dtype=float,
    )

    shown = counts
    if shading != "count":
        sums = counts.sum(axis=1 if shading == "row" else 0, keepdims=True)
        # A row or column with nothing in it has no shares; leave it at zero
        # rather than dividing by nothing and painting it as not-a-number.
        shown = np.divide(100 * counts, sums, out=np.zeros_like(counts), where=sums > 0)

    figure, axes = _new_figure(
        max(_WIDTH, len(columns) * _WIDTH_PER_COLUMN + 4.0),
        max(_MIN_HEIGHT, len(rows) * _HEIGHT_PER_ROW + 2.4),
    )
    # Counts are shaded on a square root, because one row usually dwarfs the
    # rest -- "no crack" outnumbers every crack size put together -- and on a
    # linear ramp that one cell pushes everything else to white, hiding the
    # pattern the chart was opened to find. Shares already span 0 to 100 and
    # need no such help.
    from matplotlib import colormaps
    from matplotlib.colors import Normalize, PowerNorm

    ceiling = float(shown.max()) or 1.0
    norm = (
        PowerNorm(gamma=0.5, vmin=0, vmax=ceiling) if shading == "count"
        else Normalize(vmin=0, vmax=ceiling)
    )
    palette = colormaps["Blues"].with_extremes(bad=ZERO_COLOUR)
    image = axes.imshow(
        np.ma.masked_where(counts == 0, shown), cmap=palette, aspect="auto", norm=norm
    )

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
            if not counts[row, column]:
                continue
            value = shown[row, column]
            if shading == "count":
                text = str(int(counts[row, column]))
            else:
                # A cell holding images whose share rounds to nothing would read
                # "0%" beside the grey cells that really hold none.
                text = f"{value:.0f}%" if value >= 0.5 else "<1%"
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
    figure.colorbar(image, ax=axes, label=_SHADING_LABELS[shading])
    return figure


def _finish(figure: Figure, output: Path, *, overwrite: bool = False) -> Path:
    """Write *figure* to *output* whole, in whichever format the suffix names.

    Raises:
        FileExistsError: If *output* exists, or appears while it is being
            written, and *overwrite* is False.
    """
    refuse_existing([output], overwrite=overwrite)
    landed = write_atomically(
        output,
        # The temporary keeps the suffix, which is what savefig reads the format from.
        lambda temporary: figure.savefig(temporary, dpi=200),
        commit_if=None if overwrite else (lambda _temporary: not output.exists()),
    )
    if not landed:
        raise FileExistsError(f"{output} appeared while it was being written; nothing was written")
    logger.info("Wrote %s", output)
    return output


def plot_counts(
    counts: Mapping[Hashable, int],
    title: str,
    output: Path,
    *,
    xlabel: str = "",
    ordered: bool = False,
    overwrite: bool = False,
) -> Path:
    """Draw a one-dimensional breakdown and write it to *output*.

    Args:
        output: File to write. The suffix chooses the format -- ``.png`` for a
            figure to look at, ``.svg`` for one to put in a paper.
        overwrite: Replace *output* if it already exists.

    Raises:
        RuntimeError: If matplotlib is not installed.
        FileExistsError: If *output* exists and *overwrite* is False.
    """
    return _finish(
        bar_figure(counts, title, xlabel=xlabel, ordered=ordered), output, overwrite=overwrite
    )


def plot_cross_tabulation(
    table: Mapping[tuple[Hashable, Hashable], int],
    row_title: str,
    column_title: str,
    output: Path,
    *,
    row_order: Sequence[Hashable] = (),
    column_order: Sequence[Hashable] = (),
    overwrite: bool = False,
) -> Path:
    """Draw a two-dimensional breakdown and write it to *output*.

    Raises:
        RuntimeError: If matplotlib is not installed.
        FileExistsError: If *output* exists and *overwrite* is False.
    """
    return _finish(
        heatmap_figure(
            table, row_title, column_title,
            row_order=row_order, column_order=column_order,
        ),
        output,
        overwrite=overwrite,
    )


#: What each per-class metric is called on an axis.
_METRIC_LABELS = {
    "iou": "IoU", "dice": "Dice", "f1": "F1",
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
}


#: Marks a line as a curve a reader can point at, apart from the lines marking an epoch.
CURVE = "curve"
#: The ``gid`` of the band behind the epochs a run's encoder was frozen for.
FROZEN = "frozen"

#: How near a curve's point the pointer must be to be read as pointing at it, in pixels.
HOVER_PIXELS = 10.0


@dataclass(frozen=True)
class CurvePoint:
    """One point of one curve: what the pointer is on.

    Attributes:
        axes: The panel the curve is drawn in.
        curve: What the curve is called in the legend.
        quantity: What the panel plots: its y axis's label, before any comma.
        x: The point's epoch.
        y: Its value.
    """

    axes: Any
    curve: str
    quantity: str
    x: float
    y: float

    def describe(self) -> str:
        """Return the point in a line: the curve, the epoch, the value."""
        return f"{self.curve} · epoch {self.x:g} · {self.quantity} {self.y:.3f}"


def nearest_curve_point(
    figure: Figure, x: float, y: float, within: float = HOVER_PIXELS
) -> CurvePoint | None:
    """Return the curve point nearest pixel (*x*, *y*) of a drawn *figure*, or None beyond *within*.

    Only lines marked :data:`CURVE` are looked at, and distance is measured on
    the screen, so a logarithmic axis reads as it looks.
    """
    import numpy as np

    nearest: CurvePoint | None = None
    closest = within
    for axes in figure.axes:
        for line in axes.get_lines():
            if line.get_gid() != CURVE or not line.get_visible():
                continue
            data = np.asarray(line.get_xydata(), dtype=float)
            if not len(data):
                continue
            pixels = line.get_transform().transform(data)
            distances = np.hypot(pixels[:, 0] - x, pixels[:, 1] - y)
            index = int(np.nanargmin(distances)) if not np.isnan(distances).all() else None
            if index is not None and distances[index] <= closest:
                closest = float(distances[index])
                nearest = CurvePoint(
                    axes, str(line.get_label()), axes.get_ylabel().split(",")[0],
                    float(data[index, 0]), float(data[index, 1]),
                )
    return nearest


#: A panel's view: its x limits, then its y limits.
Limits = tuple[tuple[float, float], tuple[float, float]]


def view_limits(figure: Figure) -> dict[Axes, Limits]:
    """Return the view each panel of *figure* shows now, for :func:`zoom_about` to stay within."""
    return {axes: (axes.get_xlim(), axes.get_ylim()) for axes in figure.axes}


def zoom_about(
    figure: Figure,
    x: float,
    y: float,
    factor: float,
    within: Mapping[Axes, Limits] | None = None,
) -> bool:
    """Zoom each panel of a drawn *figure* under pixel (*x*, *y*) by *factor*, that pixel staying put.

    Measured on the screen, so a logarithmic axis zooms as it looks. A
    panel that does not navigate, a colour bar's among them, is left alone.

    Args:
        factor: Above 1 to zoom in, below it to zoom out.
        within: The view a panel is never zoomed out beyond, as
            :func:`view_limits` returns it; a panel it leaves out has none.

    Returns:
        Whether any panel was under the pixel, and so changed.

    Raises:
        ValueError: If *factor* is not above 0.
    """
    if factor <= 0:
        raise ValueError(f"A zoom factor must be above 0, not {factor}")
    changed = False
    for axes in figure.axes:
        box = axes.bbox
        if not (axes.get_navigate() and axes.can_zoom() and box.contains(x, y)):
            continue
        x0, x1 = x + (box.x0 - x) / factor, x + (box.x1 - x) / factor
        y0, y1 = y + (box.y0 - y) / factor, y + (box.y1 - y) / factor
        if within is not None and axes in within:
            # Where the outermost view sits on the screen as things are drawn now.
            (low_x, high_x), (low_y, high_y) = within[axes]
            (ax, ay), (bx, by) = axes.transData.transform([(low_x, low_y), (high_x, high_y)])
            x0, x1 = _held_within(x0, x1, min(ax, bx), max(ax, bx))
            y0, y1 = _held_within(y0, y1, min(ay, by), max(ay, by))
        (left, bottom), (right, top) = axes.transData.inverted().transform([(x0, y0), (x1, y1)])
        axes.set_xlim(left, right)
        axes.set_ylim(bottom, top)
        changed = True
    return changed


def _held_within(low: float, high: float, least: float, most: float) -> tuple[float, float]:
    """Return the span *low* to *high* moved, and no wider than needed cut, to lie in *least* to *most*.

    Moved before it is cut, so zooming out at an edge goes on widening the
    other way until the whole is shown.
    """
    span = high - low
    if span >= most - least:
        return least, most
    if low < least:
        return least, least + span
    if high > most:
        return most - span, most
    return low, high


def class_colour(index: int) -> str:
    """Return the colour class *index*'s curve is drawn in, as ``#rrggbb``.

    Raises:
        RuntimeError: If matplotlib is not installed.
    """
    require_plot_support()
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    return to_hex(colormaps["tab10"](index))


def training_figure(
    run: TrainingRun,
    metric: str = "iou",
    classes: Collection[str] | None = None,
    chosen: int | None = None,
    validated_on: str = "",
    best: int | None = None,
    frozen: int | None = None,
) -> Figure:
    """Draw a run's loss, and *metric* for each class, epoch by epoch.

    The classes are drawn from validation, where a model is judged, with the
    weighted average of both sides over them; a run not yet validated draws
    its training classes instead. The epoch the run counts as best is marked,
    and the epochs its encoder was frozen for are shaded, as far as it trained.

    Args:
        metric: One of :data:`~am_segmentation_manager.training_runs.PER_CLASS_METRICS`.
        classes: The classes to draw a curve for; every one where omitted. The
            weighted averages are over every class whichever are drawn.
        chosen: The epoch a person chose as best, marked beside the run's own.
        validated_on: What the validation side is called where it is not the
            run's own list.
        best: The epoch to mark as the run's best in place of the best of the
            validation side drawn: the run chose its best on its own list.
        frozen: For how many epochs the encoder is frozen, where the run has
            not said yet: it says only once it has ended.

    Raises:
        RuntimeError: If matplotlib is not installed.
        ValueError: If the run has no epochs yet, *metric* is unknown, or one of
            *classes* is not a class of the run.
    """
    require_plot_support()
    if metric not in PER_CLASS_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {', '.join(PER_CLASS_METRICS)}")
    unknown = sorted(set(classes or ()) - set(run.label_names))
    if unknown:
        raise ValueError(
            f"Unknown class {', '.join(unknown)}; choose from {', '.join(run.label_names)}"
        )
    if not run.train and not run.val:
        raise ValueError("Nothing to plot: the run has finished no epoch yet")

    from matplotlib import colormaps
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(_WIDTH * 1.25, 4.2), layout="constrained")
    loss_axes, metric_axes = figure.subplots(1, 2)

    validation = validated_on or "validation"
    for epochs, side, style in ((run.train, "training", "--"), (run.val, validation, "-")):
        if epochs:
            loss_axes.plot([e.epoch for e in epochs], [e.loss for e in epochs],
                           style, color=_BAR_COLOUR, label=side, gid=CURVE)
    loss_axes.set_yscale("log")
    loss_axes.set_ylabel("loss")

    by_class = run.val or run.train
    palette = colormaps["tab10"]
    for index, name in enumerate(run.label_names):
        if classes is not None and name not in classes:
            continue
        points = [(e.epoch, e.per_class[metric][index]) for e in by_class
                  if len(e.per_class.get(metric, ())) > index]
        if points:
            metric_axes.plot(*zip(*points), color=palette(index), linewidth=1, label=name,
                             gid=CURVE)
    average = f"weighted_average_{metric}"
    for epochs, side, style in ((run.train, "training", "--"), (run.val, validation, "-")):
        points = [(e.epoch, e.scalars[average]) for e in epochs if average in e.scalars]
        if points:
            metric_axes.plot(*zip(*points), style, color="#222222", linewidth=1.8,
                             label=f"weighted average, {side}", gid=CURVE)
    metric_axes.set_ylabel(f"{_METRIC_LABELS[metric]}, per class from "
                           f"{validation if run.val else 'training'}")
    metric_axes.set_ylim(0, 1)

    if best is None and (ranked := run.best()) is not None:
        best = ranked.epoch
    trained = max((e.epoch for e in (*run.train, *run.val)), default=0)
    if run.frozen_epochs is not None:
        frozen = run.frozen_epochs
    shaded = min(frozen or 0, trained)
    for axes in (loss_axes, metric_axes):
        if shaded:
            # An epoch is a point at its number, so the band runs half an epoch either side.
            axes.axvspan(0.5, shaded + 0.5, color=_FROZEN_COLOUR, linewidth=0, zorder=0,
                         label=f"encoder frozen (1-{frozen})", gid=FROZEN)
        if best is not None:
            axes.axvline(best, color=_LINE_COLOUR, linestyle=":", linewidth=1.2,
                         label=f"best epoch ({best})")
        if chosen is not None:
            axes.axvline(chosen, color=_CHOSEN_COLOUR, linestyle="--", linewidth=1.2,
                         label=f"marked best ({chosen})")
        axes.set_xlabel("epoch")
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes.spines[["top", "right"]].set_visible(False)
    loss_axes.legend(fontsize=7, frameon=False)
    # Outside the axes: every corner of a run's metric panel holds a curve.
    metric_axes.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))

    started = f", started {run.started:%Y-%m-%d %H:%M} UTC" if run.started else ""
    state = "finished" if run.finished else "not finished"
    figure.suptitle(f"{run.run_id or run.directory.name}{started} ({state})")
    return figure


def comparison_ramp(steps: int) -> list[str]:
    """Return the colours a comparison's cells run through, lowest first, as ``#rrggbb``.

    Empty where matplotlib is not installed.
    """
    if not has_plot_support():
        return []
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    palette = colormaps[COMPARISON_PALETTE]
    return [to_hex(palette(step / max(steps - 1, 1))) for step in range(steps)]


def comparison_figure(
    table: Comparison, colour_range: tuple[float, float] = DEFAULT_COLOUR_RANGE
) -> Figure:
    """Draw training runs side by side as a table shaded by value: a row per run, a block per class.

    One ramp over every block, so a class a model does badly at reads as pale
    beside one it does well at, and a bar beside the table saying what the
    colours span. A cell nothing was validated for is grey and empty.

    Args:
        colour_range: The values the ramp runs between; one beyond either takes
            that end's colour.

    Raises:
        RuntimeError: If matplotlib is not installed.
        ValueError: If the table has no rows or no columns, or the range is empty.
    """
    require_plot_support()
    if not table.rows or not table.columns:
        raise ValueError("Nothing to plot: no run, or no class, to compare")
    low, high = colour_range
    if not low < high:
        raise ValueError(f"The colours must run from a lower value to a higher, not {low} to {high}")

    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    classes, sets = table.classes, table.sets
    several = len(sets) > 1
    palette = colormaps[COMPARISON_PALETTE]
    norm = Normalize(low, high, clip=True)
    # A block per class, a gap between blocks; a row of class headings, and
    # one of set names where there is more than one set.
    gap, heading = 0.25, 2 if several else 1
    left = [index * (len(sets) + gap) for index in range(len(classes))]
    width = left[-1] + len(sets)

    figure = Figure(
        figsize=(max(_WIDTH, width * _WIDTH_PER_COLUMN * 0.85 + 4.0),
                 max(2.4, (len(table.rows) + heading) * _HEIGHT_PER_ROW + 1.2)),
        layout="constrained",
    )
    axes = figure.subplots()

    for block, name in enumerate(classes):
        axes.add_patch(Rectangle((left[block], 0), len(sets), 1, edgecolor="#222222",
                                 linewidth=0.8, facecolor=_class_heading(name, block)))
        axes.text(left[block] + len(sets) / 2, 0.5, name,
                  ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        for column, where in enumerate(sets):
            if several:
                axes.add_patch(Rectangle((left[block] + column, 1), 1, 1, facecolor=ZERO_COLOUR,
                                         edgecolor="white", linewidth=0.8))
                axes.text(left[block] + column + 0.5, 1.5, where, ha="center", va="center", fontsize=8)
            for line, row in enumerate(table.rows):
                value = row.values[table.columns.index((name, where))]
                top = heading + line
                shade = None if value is None else palette(float(norm(value)))
                axes.add_patch(Rectangle(
                    (left[block] + column, top), 1, 1, edgecolor="white", linewidth=0.8,
                    facecolor=ZERO_COLOUR if shade is None else shade,
                ))
                if value is not None and shade is not None:
                    red, green, blue = shade[:3]
                    dark = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
                    axes.text(left[block] + column + 0.5, top + 0.5, f"{value:.3f}",
                              ha="center", va="center", fontsize=8, fontweight="bold",
                              color="white" if dark else "#222222")

    # A hair beyond the outermost cells, or their outlines are cut in half.
    axes.set_xlim(-0.03, width + 0.03)
    axes.set_ylim(heading + len(table.rows) + 0.03, -0.03)
    axes.set_xticks([])
    axes.set_yticks([heading + line + 0.5 for line in range(len(table.rows))])
    axes.set_yticklabels(
        [f"{row.name}   {row.epochs}" if row.epochs else row.name for row in table.rows], fontsize=9
    )
    axes.tick_params(axis="y", length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_title(
        f"Validation {_METRIC_LABELS[table.metric]} per class: {VALUE_CAPTIONS[table.value].lower()}. "
        f"Beside each run, the epochs its values are from.",
        fontsize=9, loc="left",
    )
    figure.colorbar(
        ScalarMappable(norm, palette), ax=axes, label=_METRIC_LABELS[table.metric],
        fraction=0.03, pad=0.02,
    )
    return figure


def _class_heading(name: str, block: int) -> str:
    """Return the colour a class's heading is filled with: its curve's, where its place is known."""
    from .training_runs import DEFAULT_LABEL_NAMES

    index = DEFAULT_LABEL_NAMES.index(name) if name in DEFAULT_LABEL_NAMES else block
    return class_colour(index)


def plot_comparison(
    table: Comparison,
    output: Path,
    *,
    overwrite: bool = False,
    colour_range: tuple[float, float] = DEFAULT_COLOUR_RANGE,
) -> Path:
    """Draw *table* as :func:`comparison_figure` does and write it to *output*.

    Raises:
        RuntimeError: If matplotlib is not installed.
        ValueError: If the table has no rows or no columns.
        FileExistsError: If *output* exists and *overwrite* is False.
    """
    return _finish(comparison_figure(table, colour_range), output, overwrite=overwrite)


def plot_training_run(
    run: TrainingRun,
    output: Path,
    *,
    metric: str = "iou",
    classes: Collection[str] | None = None,
    chosen: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Draw a run's curves and write them to *output*.

    Raises:
        RuntimeError: If matplotlib is not installed.
        ValueError: As :func:`training_figure`.
        FileExistsError: If *output* exists and *overwrite* is False.
    """
    return _finish(training_figure(run, metric, classes, chosen), output, overwrite=overwrite)
