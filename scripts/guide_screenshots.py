#!/usr/bin/env python3
"""Take the user guide's pictures of the window, from a collection on this computer.

**A standalone script, not an ``am-seg`` command.** Run it when the window has
changed enough that a picture in ``guides/images/`` no longer shows it. The
pictures are of ``AutoLamella-2024-07-05-12-53``, from the Adaptive Milling data
published on Zenodo (doi:10.5281/zenodo.21804785, CC-BY-4.0), which is not in
this repository: name the folder holding it::

    uv run python scripts/guide_screenshots.py <folder>
    uv run python scripts/guide_screenshots.py <folder> --only curate,label

The window is drawn offscreen at :data:`SIZE`, in the default accent, with a
throwaway settings file, and nothing that writes is ever pressed. Point it at a
copy all the same: nothing is meant to write, and a copy makes that certain.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
IMAGES = REPOSITORY / "guides" / "images"

#: The size every picture is taken at, in logical pixels: a laptop's screen.
SIZE = (1440, 900)


@dataclass(frozen=True)
class Shot:
    """One picture: which tab, which lamella and frame, and anything to set first."""

    name: str
    tab: str
    lamella: str | None = None
    frame: int | None = None
    prepare: Callable[[object], None] | None = None


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
    """Return the window, drawn offscreen at :data:`SIZE` with *collection* scanned."""
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
    window.scan(str(collection))
    return application, window


def tab_widget(window, tab: str):
    """Return the widget of the tab named *tab*, as its label reads."""
    for index in range(window.tabs.count()):
        if window.tabs.tabText(index).lower() == tab:
            return window.tabs.widget(index)
    raise KeyError(f"no tab called {tab!r}")


def go_to(application, window, lamella: str, frame: int | None) -> None:
    """Show *lamella* on Curate, at *frame* where one is given, and wait for it.

    Chosen through the list, as a click would, so the list shows it chosen.
    """
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


def take(application, window, shot: Shot) -> Path:
    """Take *shot*, returning where it was saved."""
    curate = tab_widget(window, "curate")
    window.tabs.setCurrentWidget(curate)
    if shot.lamella is not None:
        go_to(application, window, shot.lamella, shot.frame)
    window.tabs.setCurrentWidget(tab_widget(window, shot.tab))
    pause(application)
    if shot.prepare is not None:
        shot.prepare(window)
    pause(application, 2.0)
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / f"{shot.name}.png"
    if not window.grab().save(str(path)):
        raise OSError(f"could not write {path}")
    return path


def _editing_on(window) -> None:
    # Unlocks the state controls to show them; nothing is pressed that writes.
    window.state_panel.enable_check.setChecked(True)


def _label_frame(position: int) -> Callable[[object], None]:
    def prepare(window) -> None:
        window.label_panel.select_frame(position)
    return prepare


def _specks(window) -> None:
    # The first frame holds three specks far apart; fitted, all three rings show.
    panel = window.label_panel
    panel.select_frame(0)
    panel.speck_check.setChecked(True)


def _chart(window) -> None:
    panel = window.export_panel
    panel._tick_all_projects(True)
    panel.breakdown.setCurrentIndex(panel.breakdown.findData("lamella"))
    panel.cross.setCurrentIndex(panel.cross.findData("state"))
    panel.as_table.click()


SHOTS: tuple[Shot, ...] = (
    Shot("add-data", "add data"),
    Shot("curate", "curate", lamella="03-secure-oyster", frame=20),
    Shot("curate-editing", "curate", lamella="11-cosmic-locust", frame=2, prepare=_editing_on),
    Shot("label", "label", lamella="15-brave-tetra", prepare=_label_frame(12)),
    Shot("specks", "label", lamella="16-casual-swan", prepare=_specks),
    Shot("export", "export", prepare=_chart),
)


def main(argv: list[str] | None = None) -> int:
    """Take every picture, or those named with ``--only``."""
    arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    arguments.add_argument("collection", type=Path, help="The folder holding the project")
    arguments.add_argument(
        "--only", default="", help="Take only these pictures, by name, separated by commas"
    )
    options = arguments.parse_args(argv)
    wanted = {name for name in options.only.split(",") if name}
    shots = [shot for shot in SHOTS if not wanted or shot.name in wanted]

    with tempfile.TemporaryDirectory() as scratch:
        application, window = open_window(options.collection, Path(scratch) / "settings.ini")
        try:
            for shot in shots:
                print(f"Took {take(application, window, shot).relative_to(REPOSITORY)}")
        finally:
            window.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
