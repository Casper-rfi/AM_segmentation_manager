#!/usr/bin/env python3
"""Place lamellae by eye on grid overviews, as ground truth for the projection.

**This is a standalone script, not an ``am-seg`` command.** The fixtures that
``capture_fibsem_reprojection.py`` writes show that ``overview.reproject`` does
fibsemOS's arithmetic; they cannot show that the arithmetic puts a mark on the
lamella. This records where a person says each lamella is, so a test can say
how far the projection lands from it.

For each lamella it opens the project's overview zoomed on a first guess, with
the lamella's own low-magnification SEM reference image laid over it at its
footprint's scale and an opacity slider beside it, so the reference's grid bars
can be lined up with the overview's. Click where its centre belongs, nudge it a
pixel with Shift and the arrow keys, and press Enter; press C where it cannot
be told. The first guess is
the reference image matched into the overview where that match is clear, and
the projection otherwise -- every answer records which, so a guess accepted
unchanged is never mistaken for an independent one. The answer is paired with
the reference image's own stage position, that image being centred on where the
stage was.

Reads the collection without changing it. Writes ``--out`` after every answer,
whole or not at all, and resumes from it: nothing recorded is dropped, and an
answer is replaced only by answering that lamella again.

Usage::

    python scripts/refine_overview_positions.py COLLECTION PROJECT [PROJECT ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from am_segmentation_manager.image_metadata import ImageMetadata, read_image_metadata
from am_segmentation_manager.io.atomic import write_atomically
from am_segmentation_manager.overview import (
    OVERVIEW_MIN_FIELD_UM,
    Overview,
    StagePosition,
    positions_from_headers,
    reproject,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overview_sources import identify  # beside this script, not on the path

REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPOSITORY / "tests" / "data" / "overview_ground_truth.json"

#: The lamella's low-magnification SEM images, most useful first: taken at the
#: lamella's position before milling, like the overview.
REFERENCES = (
    "ref_SetupLamella_final_low_res_eb.tif",
    "ref_MillRough_final_low_res_eb.tif",
    "ref_SetupPolishing_final_low_res_eb.tif",
)

#: How far from the projection the match looks, in overview pixels.
MATCH_RADIUS_PX = 450
#: A match is clear where it correlates at least this well, and this much
#: better than the best match elsewhere in the window -- grid squares repeat,
#: so a match barely ahead of the next is a neighbouring square.
MIN_PEAK = 0.25
MIN_MARGIN = 0.05

#: How opaque the reference opens over the overview, in percent.
DEFAULT_OPACITY = 50

NOTE = (
    "Where a person placed each lamella on a grid overview, in the overview's "
    "own pixels, written by scripts/refine_overview_positions.py. 'first_guess' "
    "says whether the starting point came from matching the lamella's reference "
    "image or from the projection under test, and 'first_guess_at' where it was."
)


# ---------------------------------------------------------------------------
# What is asked about
# ---------------------------------------------------------------------------

@dataclass
class Target:
    """One lamella to place, and everything shown while placing it."""

    name: str
    reference: Path
    position: StagePosition
    projected: tuple[float, float]
    first_guess: tuple[float, float]
    guessed_by: str
    footprint: tuple[float, float]
    overlay: np.ndarray
    turned: bool


@dataclass
class Case:
    """One overview and the lamellae on it."""

    overview: str
    geometry: dict
    display_path: Path
    targets: list[Target]


def geometry_of(overview: Overview, metadata: ImageMetadata) -> dict:
    """Return *overview*'s geometry in the schema the reprojection fixtures use.

    What the header states, and where it states nothing, what the projection
    took instead: an Arctis header the microscope saved states no column
    geometry, and ``tests/test_overview.py`` rebuilds the overview from this.
    """
    def stated(value: float | None, resolved_rad: float) -> float:
        return value if value is not None else math.degrees(resolved_rad)

    centre = overview.centre
    return {
        "width": overview.width,
        "height": overview.height,
        "pixel_size_m": overview.pixel_size_m,
        "scan_rotation_rad": overview.scan_rotation,
        "beam_type": overview.beam,
        "centre": {"x": centre.x, "y": centre.y, "z": centre.z,
                   "r": centre.rotation, "t": centre.tilt},
        "system": {
            "model": metadata.microscope_model,
            "shuttle_pre_tilt_deg": stated(metadata.shuttle_pretilt_deg, overview.shuttle_pretilt),
            "rotation_reference_deg": stated(
                metadata.stage_rotation_reference_deg, overview.rotation_reference
            ),
            "rotation_180_deg": metadata.stage_rotation_180_deg,
            "electron_column_tilt_deg": metadata.electron_column_tilt_deg,
            "ion_column_tilt_deg": stated(metadata.ion_column_tilt_deg, overview.ion_column_tilt),
        },
    }


def stage_position_of(metadata: ImageMetadata, name: str) -> StagePosition | None:
    """Return where the stage was when *metadata*'s image was taken, as the Grid view reads it."""
    found = positions_from_headers({name: [metadata]})
    return found[0] if found else None


def widest_reference(lamella: Path) -> Path | None:
    """Return the lamella's widest SEM image short of an overview's width.

    For a dataset whose images are named none of :data:`REFERENCES`: the one
    showing most of the grid around the lamella is the one to line up.
    """
    best: tuple[float, Path] | None = None
    for path in sorted(lamella.rglob("*.tif")):
        seen = read_image_metadata(path)
        if (
            seen is None
            or seen.beam_type != "ELECTRON"
            or not seen.hfw_um
            or seen.hfw_um >= OVERVIEW_MIN_FIELD_UM
            or stage_position_of(seen, lamella.name) is None
        ):
            continue
        if best is None or seen.hfw_um > best[0]:
            best = (seen.hfw_um, path)
    return best[1] if best is not None else None


def load_case(collection: Path, relative: str) -> Case | None:
    """Return the overview the Grid view opens on for one project, with its lamellae."""
    from am_segmentation_manager.gui.grid_controller import GridController
    from am_segmentation_manager.scanner import Scanner

    project_path = (collection / relative).resolve()
    found = [
        project for project in Scanner(project_path.parent).build_index().all_projects()
        if project.path.resolve() == project_path
    ]
    if not found:
        print(f"{relative}: not a project the scan finds", file=sys.stderr)
        return None
    grid = GridController()
    grid.set_project(found[0])
    choice = grid.current
    if choice is None or choice.placement is None:
        reason = choice.reason if choice is not None else "no overview"
        print(f"{relative}: nothing to place ({reason})", file=sys.stderr)
        return None

    overview = choice.placement.overview
    metadata = read_image_metadata(overview.path)
    picture = grey(choice.overview.display_path)
    targets = []
    for marker in choice.placement.markers:
        target = _target(found[0].path / marker.name, marker.name, overview, metadata, picture)
        if target is not None:
            targets.append(target)
    return Case(
        overview=identify(overview.path.relative_to(collection).as_posix()),
        geometry=geometry_of(overview, metadata),
        display_path=choice.overview.display_path,
        targets=targets,
    )


def _target(lamella: Path, name: str, overview: Overview, metadata: ImageMetadata,
            picture: np.ndarray) -> Target | None:
    reference = next(
        (lamella / r for r in REFERENCES if (lamella / r).exists()), None
    ) or widest_reference(lamella)
    if reference is None:
        print(f"{name}: no SEM image to line up", file=sys.stderr)
        return None
    seen = read_image_metadata(reference)
    position = stage_position_of(seen, name) if seen is not None else None
    if position is None or not seen.pixel_size_nm:
        print(f"{name}: the reference image states no stage position", file=sys.stderr)
        return None
    try:
        projected = reproject(overview, position)
    except ValueError as exc:
        print(f"{name}: {exc}", file=sys.stderr)
        return None

    turned = turned_about(seen.scan_rotation_deg, metadata.scan_rotation_deg)
    template = grey(reference)
    if turned:
        template = template[::-1, ::-1]
    scale_x = seen.pixel_size_nm / metadata.pixel_size_nm
    scale_y = scale_x * _foreshortening(metadata) / _foreshortening(seen, metadata)
    footprint = (template.shape[1] * scale_x, template.shape[0] * scale_y)
    overlay = resize(template, (max(8, round(footprint[1])), max(8, round(footprint[0]))))
    guess, guessed_by = first_guess(picture, overlay, projected)
    return Target(
        name, reference, position, projected, guess, guessed_by, footprint, overlay, turned
    )


def turned_about(reference_scan_deg: float | None, overview_scan_deg: float | None) -> bool:
    """Return whether a reference is written half a turn from the overview.

    A beam at a half-turn of scan rotation writes the image upside down and
    backwards, so a reference taken at the other rotation from its overview has
    to be turned about before it is matched or laid over it.
    """
    difference = ((reference_scan_deg or 0.0) - (overview_scan_deg or 0.0)) % 360.0
    return abs(difference - 180.0) < 1.0


def _foreshortening(seen: ImageMetadata, overview: ImageMetadata | None = None) -> float:
    """Return how much of the grid's length an SEM image at *seen*'s tilt shows."""
    pretilt = (overview or seen).shuttle_pretilt_deg or 0.0
    return math.cos(math.radians((seen.stage_tilt_deg or 0.0) - pretilt))


# ---------------------------------------------------------------------------
# The first guess
# ---------------------------------------------------------------------------

def grey(path: Path) -> np.ndarray:
    """Return the image at *path* as one channel of float32."""
    array = tifffile.imread(str(path))
    if array.ndim == 3:
        array = array[..., 0] if array.shape[-1] in (3, 4) else array[0]
    return array.astype(np.float32)


def resize(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return *array* resampled to *shape*, averaging blocks before interpolating."""
    rows, columns = array.shape
    height, width = shape
    step_y, step_x = max(1, rows // height), max(1, columns // width)
    if step_y > 1 or step_x > 1:
        array = array[: rows - rows % step_y, : columns - columns % step_x]
        array = array.reshape(rows // step_y, step_y, columns // step_x, step_x).mean(axis=(1, 3))
        rows, columns = array.shape
    ys, xs = np.linspace(0, rows - 1, height), np.linspace(0, columns - 1, width)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, rows - 1), np.minimum(x0 + 1, columns - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = array[y0][:, x0] * (1 - wx) + array[y0][:, x1] * wx
    bottom = array[y1][:, x0] * (1 - wx) + array[y1][:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def correlation(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Return the normalised cross-correlation of *template* at every offset in *image*."""
    image = image.astype(np.float64)
    template = template.astype(np.float64)
    rows, columns = image.shape
    height, width = template.shape
    centred = template - template.mean()
    norm = np.sqrt((centred ** 2).sum()) or 1.0
    size = (rows + height - 1, columns + width - 1)
    product = np.fft.irfft2(np.fft.rfft2(image, size) * np.fft.rfft2(centred[::-1, ::-1], size), size)
    product = product[height - 1:rows, width - 1:columns]

    def box(values: np.ndarray) -> np.ndarray:
        summed = np.zeros((rows + 1, columns + 1))
        summed[1:, 1:] = values.cumsum(0).cumsum(1)
        return (summed[height:, width:] - summed[:-height, width:]
                - summed[height:, :-width] + summed[:-height, :-width])

    count = height * width
    local = box(image)
    variance = np.maximum(box(image ** 2) - local ** 2 / count, 1e-9)
    return product / (np.sqrt(variance) * norm)


def first_guess(picture: np.ndarray, template: np.ndarray,
                projected: tuple[float, float]) -> tuple[tuple[float, float], str]:
    """Return where to start, and whether it was ``"matched"`` or ``"projected"``.

    The match looks only near the projection, and is taken only where it is
    clear: the correlation high, and ahead of the best elsewhere in the window.
    """
    height, width = template.shape
    px, py = projected
    top = int(max(0, py - MATCH_RADIUS_PX - height / 2))
    bottom = int(min(picture.shape[0], py + MATCH_RADIUS_PX + height / 2))
    left = int(max(0, px - MATCH_RADIUS_PX - width / 2))
    right = int(min(picture.shape[1], px + MATCH_RADIUS_PX + width / 2))
    window = picture[top:bottom, left:right]
    if window.shape[0] <= height or window.shape[1] <= width:
        return projected, "projected"
    scores = correlation(window, template)
    row, column = np.unravel_index(np.argmax(scores), scores.shape)
    peak = scores[row, column]
    elsewhere = scores.copy()
    elsewhere[max(0, row - height // 2): row + height // 2,
              max(0, column - width // 2): column + width // 2] = -1
    if peak < MIN_PEAK or peak - elsewhere.max() < MIN_MARGIN:
        return projected, "projected"
    return (left + column + width / 2, top + row + height / 2), "matched"


# ---------------------------------------------------------------------------
# What is recorded
# ---------------------------------------------------------------------------

def load_document(path: Path) -> dict:
    """Return what *path* already records, or an empty record."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"note": NOTE, "cases": []}


def answered(document: dict, overview: str, name: str) -> dict | None:
    """Return the answer recorded for lamella *name* on *overview*, if any."""
    for case in document["cases"]:
        if case["overview"] == overview:
            for entry in case["positions"]:
                if entry["name"] == name:
                    return entry
    return None


def record(document: dict, overview: str, geometry: dict, entry: dict) -> dict:
    """Put *entry* in *document*, replacing only that lamella's earlier answer."""
    case = next((case for case in document["cases"] if case["overview"] == overview), None)
    if case is None:
        case = {"overview": overview, **geometry, "positions": []}
        document["cases"].append(case)
    case["positions"] = [kept for kept in case["positions"] if kept["name"] != entry["name"]]
    case["positions"].append(entry)
    return document


def entry_for(target: Target, point: tuple[float, float] | None) -> dict:
    """Return the record of one answer: a point, or None where it could not be told."""
    position = target.position
    entry = {
        "name": target.name,
        "reference": target.reference.name,
        "position": {"x": position.x, "y": position.y, "z": position.z,
                     "r": position.rotation, "t": position.tilt},
        "projected": list(target.projected),
        "first_guess": target.guessed_by,
        "first_guess_at": list(target.first_guess),
        "answer": "placed" if point is not None else "cannot tell",
        "answered_at": datetime.now().isoformat(timespec="seconds"),
    }
    if point is not None:
        entry["x"], entry["y"] = point
    return entry


def save(document: dict, path: Path) -> None:
    """Write *document* to *path*, whole or not at all."""
    write_atomically(
        path, lambda temporary: temporary.write_text(
            json.dumps(document, indent=1) + "\n", encoding="utf-8"),
    )


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

def run(cases: list[Case], out: Path) -> None:
    from qtpy.QtCore import QPointF, QRectF, Qt, Signal
    from qtpy.QtGui import QBrush, QColor, QImage, QKeySequence, QPen, QPixmap
    from qtpy.QtWidgets import (
        QApplication,
        QGraphicsEllipseItem,
        QGraphicsItem,
        QGraphicsPixmapItem,
        QGraphicsRectItem,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QShortcut,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    from am_segmentation_manager.gui.widgets.image_view import ImageView
    from am_segmentation_manager.gui.widgets.theme import active_accent

    def display(array: np.ndarray) -> np.ndarray:
        low, high = np.percentile(array, (0.5, 99.5))
        scaled = (array - low) / max(high - low, 1e-6)
        return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)

    class PlacingView(ImageView):
        """An image view where a click that does not drag places the mark."""

        placed = Signal(float, float)

        def mousePressEvent(self, event):  # noqa: N802 - Qt naming
            self._pressed = event.pos()
            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event):  # noqa: N802 - Qt naming
            pressed = getattr(self, "_pressed", None)
            if pressed is not None and (event.pos() - pressed).manhattanLength() < 4:
                point = self.mapToScene(event.pos())
                self.placed.emit(point.x(), point.y())
            self._pressed = None
            super().mouseReleaseEvent(event)

    targets = [(case, target) for case in cases for target in case.targets]
    document = load_document(out)
    app = QApplication.instance() or QApplication(sys.argv[:1])

    window = QMainWindow()
    window.setWindowTitle("Place lamellae on grid overviews")
    view = PlacingView()
    title = QLabel()
    title.setWordWrap(True)
    reference = QLabel()
    reference.setAlignment(Qt.AlignCenter)
    reference.setMinimumWidth(420)
    help_text = QLabel(
        "The reference is laid over the overview at its own scale. Click where its "
        "centre belongs, Shift and the arrow keys move it a pixel, and the opacity "
        "lets its grid bars be lined up with the overview's; then Enter. C if it "
        "cannot be told; left and right arrows move without answering. Drag to "
        "pan, wheel to zoom. Saved after every answer."
    )
    help_text.setWordWrap(True)
    buttons = QHBoxLayout()
    back, cannot, confirm, onward = (QPushButton(text) for text in
                                     ("Back", "Cannot tell (C)", "Confirm (Enter)", "Next"))
    for button in (back, cannot, confirm, onward):
        buttons.addWidget(button)
    opacity_label = QLabel("Reference overlay opacity")
    opacity = QSlider(Qt.Horizontal)
    opacity.setRange(0, 100)
    opacity.setValue(DEFAULT_OPACITY)
    column = QVBoxLayout()
    for widget in (title, reference, opacity_label, opacity, help_text):
        column.addWidget(widget)
    column.addLayout(buttons)
    column.addStretch(1)
    side = QWidget()
    side.setLayout(column)
    side.setFixedWidth(460)
    central = QWidget()
    row = QHBoxLayout(central)
    row.addWidget(view, stretch=1)
    row.addWidget(side)
    window.setCentralWidget(central)

    pen = QPen(QColor(active_accent().colour), 2)
    pen.setCosmetic(True)
    ring = QGraphicsEllipseItem(QRectF(-8, -8, 16, 16))
    ring.setPen(pen)
    ring.setBrush(QBrush(Qt.NoBrush))
    ring.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
    outline = QGraphicsRectItem()
    outline.setPen(pen)
    laid = QGraphicsPixmapItem()
    laid.setTransformationMode(Qt.SmoothTransformation)
    laid.setOpacity(DEFAULT_OPACITY / 100)
    opacity.valueChanged.connect(lambda value: laid.setOpacity(value / 100))

    def qimage(array: np.ndarray) -> QImage:
        array = np.ascontiguousarray(array)
        return QImage(array.data, array.shape[1], array.shape[0], array.shape[1],
                      QImage.Format_Grayscale8).copy()
    state = {"index": 0, "shown": None, "point": None}

    def mark(x: float, y: float) -> None:
        state["point"] = (x, y)
        target = targets[state["index"]][1]
        width, height = target.footprint
        ring.setPos(x, y)
        outline.setRect(QRectF(x - width / 2, y - height / 2, width, height))
        laid.setPos(x - target.overlay.shape[1] / 2, y - target.overlay.shape[0] / 2)

    def show(index: int) -> None:
        state["index"] = index % len(targets)
        case, target = targets[state["index"]]
        if state["shown"] is not case:
            view.set_array(display(grey(case.display_path)))
            for depth, item in ((5, laid), (10, outline), (10, ring)):
                if item.scene() is not view.scene():
                    view.scene().addItem(item)
                item.setZValue(depth)
            state["shown"] = case
        earlier = answered(document, case.overview, target.name)
        start = (earlier["x"], earlier["y"]) if earlier and "x" in earlier else target.first_guess
        laid.setPixmap(QPixmap.fromImage(qimage(display(target.overlay))))
        mark(*start)
        view.fit()
        wanted = view.viewport().width() / max(target.footprint[0] * 5, 400)
        view.zoom_by(wanted / view.transform().m11())
        view.centerOn(QPointF(*start))
        picture = grey(target.reference)
        if target.turned:
            picture = picture[::-1, ::-1]
        reference.setPixmap(
            QPixmap.fromImage(qimage(display(picture))).scaledToWidth(420, Qt.SmoothTransformation)
        )
        done = sum(1 for c, t in targets if answered(document, c.overview, t.name))
        status = (f"answered before: {earlier['answer']}" if earlier
                  else f"first guess from the {target.guessed_by} position")
        title.setText(
            f"<b>{target.name}</b> — lamella {state['index'] + 1} of {len(targets)}, "
            f"{done} answered<br>{case.overview}<br>{status}"
        )

    def answer(point: tuple[float, float] | None) -> None:
        case, target = targets[state["index"]]
        record(document, case.overview, case.geometry, entry_for(target, point))
        save(document, out)
        unanswered = [i for i, (c, t) in enumerate(targets)
                      if not answered(document, c.overview, t.name)]
        later = [i for i in unanswered if i > state["index"]] or unanswered
        if later:
            show(later[0])
        else:
            show(state["index"])
            title.setText(title.text() + "<br><b>Every lamella is answered.</b>")

    view.placed.connect(mark)

    def nudge(dx: float, dy: float) -> None:
        x, y = state["point"]
        mark(x + dx, y + dy)

    for keys, (dx, dy) in (("Shift+Left", (-1, 0)), ("Shift+Right", (1, 0)),
                           ("Shift+Up", (0, -1)), ("Shift+Down", (0, 1))):
        QShortcut(QKeySequence(keys), window, activated=lambda dx=dx, dy=dy: nudge(dx, dy))
    confirm.clicked.connect(lambda: answer(state["point"]))
    cannot.clicked.connect(lambda: answer(None))
    back.clicked.connect(lambda: show(state["index"] - 1))
    onward.clicked.connect(lambda: show(state["index"] + 1))
    for keys, action in (("Return", lambda: answer(state["point"])), ("Enter", lambda: answer(state["point"])),
                         ("C", lambda: answer(None)), ("Left", lambda: show(state["index"] - 1)),
                         ("Right", lambda: show(state["index"] + 1))):
        QShortcut(QKeySequence(keys), window, activated=action)

    window.resize(1440, 900)
    window.show()
    first = next((i for i, (c, t) in enumerate(targets)
                  if not answered(document, c.overview, t.name)), 0)
    show(first)
    app.exec_() if hasattr(app, "exec_") else app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("collection", type=Path, help="Folder holding the projects")
    parser.add_argument("projects", nargs="+", help="Projects to ask about, relative to it")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Where the answers go and are resumed from (default {DEFAULT_OUT})")
    args = parser.parse_args()
    cases = [case for relative in args.projects
             if (case := load_case(args.collection, relative)) is not None and case.targets]
    if not cases:
        raise SystemExit("Nothing to place.")
    matched = sum(t.guessed_by == "matched" for case in cases for t in case.targets)
    total = sum(len(case.targets) for case in cases)
    print(f"{total} lamellae on {len(cases)} overviews; {matched} first guesses from the image match")
    run(cases, args.out)


if __name__ == "__main__":
    main()
