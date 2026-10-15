#!/usr/bin/env python3
"""Take the user guide's pictures of the window, from a collection on this computer.

**A standalone script, not an ``am-seg`` command.** Run it when the window has
changed enough that a picture in ``guides/images/`` no longer shows it. The
pictures are of the Rosalind Franklin Institute's own Autolamella collection,
which is not published: two of its lamellae (:data:`_WAFFLE`, :data:`_AM01`), the
Export tab's counts over all of it, and the Train tab over two training runs.
Name the collection's folder, and the training directory holding the runs::

    uv run python scripts/guide_screenshots.py <collection> --training-directory <runs>
    uv run python scripts/guide_screenshots.py <collection> --only curate,label

The window is drawn offscreen at :data:`SIZE`, in the default accent, with a
throwaway settings file, and nothing that writes is ever pressed. Run it where
writing to the data is refused all the same, such as under macOS's
``sandbox-exec`` with ``(deny file-write* (subpath "<data>"))``.
A picture is the whole window or the part of it holding the widgets named, and
numbered marks ring the widgets the guide's text refers to by number.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
IMAGES = REPOSITORY / "guides" / "images"

#: The size every picture is taken at, in logical pixels: a laptop's screen.
SIZE = (1440, 900)

#: What a value's chip ends with, after the value.
CHIP_CROSS = "  ✕"

#: The colour of the numbered marks: a red no segmentation class or accent uses.
MARK_COLOUR = "#d6336c"
#: The diameter of a numbered mark.
MARK_SIZE = 26

#: Returns the widgets a picture or a mark covers, from the window.
Widgets = Callable[[object], Sequence[object]]


@dataclass(frozen=True)
class Shot:
    """One picture: which tab, which lamella and frame, what to set first, and what to show.

    *crop* names the widgets the picture holds, with *margin* pixels round
    them, the whole window where it is None; each entry of *marks* names the
    widgets one numbered mark rings. *scanned* False takes the picture before
    the folder is read.
    """

    name: str
    tab: str
    project: str | None = None
    lamella: str | None = None
    frame: int | None = None
    prepare: Callable[[object], None] | None = None
    crop: Widgets | None = None
    marks: tuple[Widgets, ...] = ()
    margin: int = 8
    scanned: bool = True


def settle(application, condition: Callable[[], bool], what: str, timeout: float = 120) -> None:
    """Turn the event loop until *condition* holds.

    Raises:
        TimeoutError: If it does not within *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(f"gave up waiting for {what}")
        application.processEvents()
        time.sleep(0.02)


def open_window(collection: Path, settings_path: Path):
    """Return the window, drawn offscreen at :data:`SIZE` and not yet scanned."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QApplication

    from am_segmentation_manager.config import load_segmentation_config
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.preferences import DEFAULT_ACCENT
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow
    from am_segmentation_manager.gui.widgets.theme import ACCENTS, apply_theme

    application = QApplication.instance() or QApplication([])
    apply_theme(application, ACCENTS[DEFAULT_ACCENT])
    config = load_segmentation_config(start_path=collection)
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config),
        settings=QSettings(str(settings_path), QSettings.Format.IniFormat),
    )
    window.resize(*SIZE)
    window.show()
    return application, window


def tab_widget(window, tab: str):
    """Return the widget of the tab named *tab*, as its label reads."""
    for index in range(window.tabs.count()):
        if window.tabs.tabText(index).lower() == tab:
            return window.tabs.widget(index)
    raise KeyError(f"no tab called {tab!r}")


def go_to(application, window, project: str | None, lamella: str, frame: int | None) -> None:
    """Show *lamella* of *project* on Curate, at *frame* where one is given, and wait.

    Chosen through the lists, as a click would, so the lists show it chosen.
    """
    if project is not None:
        tree = window.project_panel.project_list
        index = [entry.name for entry in window.controller.projects].index(project)
        tree.setCurrentItem(
            next(item for item in tree.project_items() if tree.project_of(item) == index)
        )
    names = [entry.name for entry in window.controller.lamellae]
    window.project_panel.lamella_list.setCurrentRow(names.index(lamella))
    if frame is not None:
        window._seek_frame(frame)
    settle(
        application,
        lambda: not window.image_panel.pair.left._item.pixmap().isNull(),
        f"{lamella} to be drawn",
    )


def pause(application, seconds: float = 1.0) -> None:
    """Let queued work -- a frame loading, a sweep counting -- reach the screen."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.02)


def bounds(window, widgets: Sequence[object]):
    """Return the smallest rectangle in *window* holding every one of *widgets* shown."""
    from qtpy.QtCore import QPoint, QRect

    united = QRect()
    for widget in widgets:
        if not widget.isVisible():
            continue
        united = united.united(QRect(widget.mapTo(window, QPoint(0, 0)), widget.size()))
    return united


def annotate(window, picture, marks: Sequence[Widgets]) -> None:
    """Ring each mark's widgets on *picture*, numbered from 1 at its top-right corner.

    The top right, because a caption or a card's title sits at the top left.
    """
    from qtpy.QtCore import QPointF, QRectF, Qt
    from qtpy.QtGui import QColor, QFont, QPainter, QPen

    colour = QColor(MARK_COLOUR)
    painter = QPainter(picture)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont(painter.font())
    font.setBold(True)
    font.setPixelSize(MARK_SIZE * 6 // 10)
    painter.setFont(font)
    rings = [QRectF(bounds(window, mark(window))).adjusted(-3, -3, 3, 3) for mark in marks]
    # Every ring before any number, so no ring is drawn across a number.
    painter.setPen(QPen(colour, 2.5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for ring in rings:
        painter.drawRoundedRect(ring, 6, 6)
    for number, ring in enumerate(rings, start=1):
        # Centred on the corner, but never so far out that the window's edge cuts it.
        centre = QPointF(
            min(ring.right(), window.width() - MARK_SIZE / 2 - 2),
            max(ring.top(), MARK_SIZE / 2 + 2),
        )
        badge = QRectF(centre.x() - MARK_SIZE / 2, centre.y() - MARK_SIZE / 2, MARK_SIZE, MARK_SIZE)
        painter.setPen(QPen(QColor("white"), 2))
        painter.setBrush(colour)
        painter.drawEllipse(badge)
        painter.setPen(QColor("white"))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(number))
    painter.end()


def take(application, window, shot: Shot) -> Path:
    """Take *shot*, returning where it was saved.

    Every shot starts with editing off, whatever the one before switched on,
    since one window takes them all.
    """
    curate = tab_widget(window, "curate")
    window.tabs.setCurrentWidget(curate)
    window.state_panel.enable_check.setChecked(False)
    if shot.lamella is not None:
        go_to(application, window, shot.project, shot.lamella, shot.frame)
    window.tabs.setCurrentWidget(tab_widget(window, shot.tab))
    pause(application)
    if shot.prepare is not None:
        shot.prepare(window)
    pause(application, 2.0)

    picture = window.grab()
    ratio = picture.devicePixelRatio()
    annotate(window, picture, shot.marks)
    if shot.crop is not None:
        area = bounds(window, shot.crop(window))
        area = area.adjusted(-shot.margin, -shot.margin, shot.margin, shot.margin)
        area = area.intersected(window.rect())
        picture = picture.copy(
            int(area.x() * ratio), int(area.y() * ratio),
            int(area.width() * ratio), int(area.height() * ratio),
        )
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / f"{shot.name}.png"
    # Quality 0 is PNG's strongest compression: lossless, only slower.
    if not picture.save(str(path), "PNG", 0):
        raise OSError(f"could not write {path}")
    return path


def _grid(window) -> None:
    window.image_panel.mode_buttons["grid"].click()


def _editing_on(window) -> None:
    # Unlocks the state controls to show them; nothing is pressed that writes.
    window.state_panel.enable_check.setChecked(True)


def _describing(project: str) -> Callable[[object], None]:
    """Show *project*'s description on Add data, which lists projects apart from Curate."""
    def prepare(window) -> None:
        tree = window.metadata_panel.projects
        index = [entry.name for entry in window.metadata_controller.projects].index(project)
        item = next(item for item in tree.project_items() if tree.project_of(item) == index)
        tree.setCurrentItem(item)
        tree.scrollToItem(item)
        # Unlocks the fields to show them; nothing is saved.
        window.metadata_panel.enable.setChecked(True)
        _first_names_only(window)
    return prepare


def _first_names_only(window) -> None:
    """Show each user but the one recording by first name alone, on the chips only.

    The picture names the people on a project; their full names are theirs to publish.
    """
    from qtpy.QtWidgets import QPushButton

    recording = window._attribution.recorder.by
    for chip in window.metadata_panel.editors["user"].findChildren(QPushButton):
        name = chip.text().removesuffix(CHIP_CROSS)
        if name and name != recording:
            chip.setText(name.split()[0] + CHIP_CROSS)


def _label_frame(position: int) -> Callable[[object], None]:
    def prepare(window) -> None:
        window.label_panel.select_frame(position)
    return prepare


def _specks(window) -> None:
    # This frame holds three specks; fitted, all three rings show. Specks are
    # counted on the label on screen, so only once its frame is read.
    from qtpy.QtWidgets import QApplication

    panel = window.label_panel
    panel.select_frame(17)
    settle(QApplication.instance(), lambda: panel.controller.guide_key is not None,
           "the frame to be read")
    panel.speck_check.setChecked(True)


def _labelling_guide(window) -> None:
    # Ticks are held in memory only; nothing is written.
    from qtpy.QtWidgets import QApplication

    panel = window.label_panel
    panel.select_frame(1)
    panel.guide_check.setChecked(True)
    # A step is ticked against the label on screen, once its frame is read.
    settle(QApplication.instance(), lambda: panel.controller.guide_key is not None,
           "the frame to be read")
    for step in range(3):
        panel.controller.tick_step(step, True)
    panel._refresh_guide()


def _breakdown(view: str) -> Callable[[object], None]:
    """Count every project's SEM Adaptive Polishing images by microscope, then lamella type."""
    def prepare(window) -> None:
        panel = window.export_panel
        panel._tick_all_projects(True)
        panel.modality.setCurrentText("SEM")
        panel.breakdown.setCurrentIndex(panel.breakdown.findData("microscope"))
        panel.cross.setCurrentIndex(panel.cross.findData("lamella-type"))
        (panel.as_chart if view == "chart" else panel.as_table).click()
    return prepare


#: The runs the Train tab is pictured with, in the folder ``--training-directory`` names.
_RUNS = ("cor_only", "cor_sel")
#: The run looked at, and the export a new run is shown set up from.
_SHOWN_RUN = "cor_sel"
_SHOWN_EXPORT = "cor_sel_manifest.yaml"


def _train(view: str, form_room: int | None = None) -> Callable[[object], None]:
    """Show the training directory's runs, both ticked, one looked at, in *view*.

    *form_room*, where given, is the height the New run form is given, so it
    shows whole.
    """
    def prepare(window) -> None:
        panel, controller = window.training_panel, window.training_controller
        controller.training_directory = TRAINING
        controller.manifest = TRAINING / _SHOWN_EXPORT
        panel.refresh()
        panel._tick_all(True)
        names = [run.directory.name for run in panel._listed]
        tree = panel.runs_tree
        tree.setCurrentItem(tree.project_item(names.index(_SHOWN_RUN)))
        panel.view_control.buttons[view].click()
        if form_room is not None:
            total = sum(panel.lists.sizes())
            panel.lists.setSizes([total - form_room, form_room])
        # Named, not placed: where the runs are on this computer is nobody's business.
        panel.directory_label.setText(TRAINING.name)
        panel.manifest_label.setText(_SHOWN_EXPORT)
    return prepare


def _group(widget) -> object:
    """Return the card *widget* sits in, or *widget* where it sits in none."""
    from qtpy.QtWidgets import QGroupBox

    parent = widget
    while parent is not None and not isinstance(parent, QGroupBox):
        parent = parent.parentWidget()
    return parent if parent is not None else widget


#: Where the runs are, from ``--training-directory``.
TRAINING = Path()

#: The two projects, and the lamella of each that is pictured.
_WAFFLE = "AutoLamella-2026-09-24-15-08"
_QUAIL = "02-needed-quail"
_AM01 = "AutoLamella-2025-07-21-15-21_AM01"
_SWINE = "01-wanted-swine"

SHOTS: tuple[Shot, ...] = (
    Shot(
        "start", "curate", scanned=False, margin=24,
        crop=lambda w: [w.image_panel.nothing_read.headline, w.image_panel.nothing_read.scan_button,
                        w.image_panel.nothing_read.hint],
    ),
    Shot(
        "scan-bar", "curate", scanned=False,
        crop=lambda w: [w.scan_bar],
        marks=(lambda w: [w.scan_bar.browse_button], lambda w: [w.scan_bar.scan_button]),
    ),
    Shot("curate", "curate", _WAFFLE, _QUAIL, frame=3),
    Shot(
        "add-data", "add data", _WAFFLE, _QUAIL, prepare=_describing(_WAFFLE),
        marks=(
            lambda w: [w.metadata_panel.data_box],
            lambda w: [w.metadata_panel.enable],
            lambda w: [w.metadata_panel.fields_scroller],
            lambda w: [w.metadata_panel.preview_button, w.metadata_panel.save_button],
        ),
    ),
    Shot(
        "curate-map", "curate", _AM01, _SWINE, frame=19,
        marks=(
            lambda w: [w.project_panel.shown_flow],
            lambda w: [w.project_panel.project_list],
            lambda w: [w.project_panel.lamella_list],
            lambda w: [w.image_panel.mode_control],
            lambda w: [w.image_panel.series_combo, w.image_panel.variant_combo],
            lambda w: [w.timeline],
            lambda w: [w.state_panel.enable_check],
            lambda w: [w.operations_panel.box],
        ),
    ),
    Shot(
        "curate-grid", "curate", _AM01, _SWINE, prepare=_grid, margin=0,
        crop=lambda w: [w.image_panel.stack],
    ),
    Shot(
        "curate-states", "curate", _AM01, _SWINE, frame=19, prepare=_editing_on, margin=6,
        crop=lambda w: [w.side_tabs, w.state_panel.enable_check, w.state_panel.readout],
        marks=(
            lambda w: [w.state_panel.enable_check],
            lambda w: [w.state_panel.applies_to, w.state_panel.reach],
            lambda w: [w.state_panel.target_control],
            lambda w: [w.state_panel.state_box],
        ),
    ),
    Shot(
        "timeline", "curate", _AM01, _SWINE, frame=19, margin=4,
        crop=lambda w: [w.timeline, w.frame_label],
    ),
    Shot(
        "operations", "curate", _AM01, _SWINE, frame=19, margin=4,
        crop=lambda w: [w.operations_panel.box],
    ),
    Shot(
        "label", "label", _WAFFLE, _QUAIL, prepare=_label_frame(1),
        marks=(
            lambda w: list(w.label_panel.paint_buttons.values())
            + list(w.label_panel.lock_buttons.values()),
            lambda w: list(w.label_panel.tool_buttons.values()),
            lambda w: [w.label_panel.speck_check],
            lambda w: [w.label_panel.versions_check],
            lambda w: [w.label_panel.save_button],
        ),
    ),
    Shot(
        "labelling-guide", "label", _WAFFLE, _QUAIL, prepare=_labelling_guide, margin=4,
        crop=lambda w: [w.label_panel.guide_panel],
    ),
    Shot(
        "specks", "label", _AM01, _SWINE, prepare=_specks, margin=0,
        crop=lambda w: [w.label_panel.canvas],
    ),
    Shot(
        "export", "export", prepare=_breakdown("table"),
        marks=(
            lambda w: [w.export_panel.series_block, w.export_panel.filters],
            lambda w: [_group(w.export_panel.breakdown)],
            lambda w: [w.export_panel.measure_button],
            lambda w: [_group(w.export_panel.keep_together)],
            lambda w: [w.export_panel.csv_box],
            lambda w: [w.export_panel.copy_box],
        ),
    ),
    Shot(
        "export-chart", "export", prepare=_breakdown("chart"), margin=0,
        crop=lambda w: [w.export_panel.views],
    ),
    Shot("train", "train", prepare=_train("curves")),
    Shot(
        "train-new-run", "train", prepare=_train("curves", form_room=560), margin=4,
        crop=lambda w: [_group(w.training_panel.name_edit)],
        marks=(
            lambda w: [w.training_panel.last_export_button, w.training_panel.browse_button],
            lambda w: [w.training_panel.mode],
            lambda w: [w.training_panel.name_edit, w.training_panel.device_box],
            lambda w: [w.training_panel.batch],
            lambda w: [w.training_panel.start_button],
        ),
    ),
    Shot(
        "train-table", "train", prepare=_train("table"), margin=0,
        crop=lambda w: [w.training_panel.plots_box],
    ),
)


def main(argv: list[str] | None = None) -> int:
    """Take every picture, or those named with ``--only``."""
    arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    arguments.add_argument("collection", type=Path, help="The folder holding the projects")
    arguments.add_argument(
        "--training-directory", type=Path, default=None,
        help="The folder holding the training runs the Train tab is pictured with",
    )
    arguments.add_argument(
        "--only", default="", help="Take only these pictures, by name, separated by commas"
    )
    options = arguments.parse_args(argv)
    wanted = {name for name in options.only.split(",") if name}
    shots = [shot for shot in SHOTS if not wanted or shot.name in wanted]
    if any(shot.tab == "train" for shot in shots):
        if options.training_directory is None:
            arguments.error("the Train tab's pictures need --training-directory")
        global TRAINING
        TRAINING = options.training_directory

    with tempfile.TemporaryDirectory() as scratch:
        application, window = open_window(options.collection, Path(scratch) / "settings.ini")
        try:
            # The scan bar shows the folder before it is read, as it does at startup.
            window.scan_bar.set_path(str(options.collection))
            for shot in (shot for shot in shots if not shot.scanned):
                print(f"Took {take(application, window, shot).relative_to(REPOSITORY)}")
            window.scan(str(options.collection))
            for shot in (shot for shot in shots if shot.scanned):
                print(f"Took {take(application, window, shot).relative_to(REPOSITORY)}")
        finally:
            window.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
