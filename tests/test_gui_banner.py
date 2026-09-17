"""The panel shown over the window while the collection is being read.

A scan holds the interface thread, so the two things worth pinning are that the
banner is up for the whole of one and down afterwards however it ended, and
that what it says comes from the scan rather than from a guess. Offscreen
nothing is ever visible, so ``isHidden`` is what is asserted -- see
`docs/display.md`.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from conftest import build_dataset, destroy

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qtpy") is None, reason="Qt binding not installed"
)

if importlib.util.find_spec("qtpy") is not None:
    import numpy as np
    import tifffile
    from qtpy.QtCore import QSettings
    from qtpy.QtGui import QColor, QImage
    from qtpy.QtWidgets import QLabel, QWidget

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.cache import FrameCache
    from am_segmentation_manager.gui.controller import CurationController
    from am_segmentation_manager.gui.widgets.branding import PRODUCT_NAME
    from am_segmentation_manager.gui.widgets.main_window import CurationWindow
    from am_segmentation_manager.gui.widgets.scan_banner import ScanBanner


@pytest.fixture
def config():
    return SegmentationConfig(
        class_colors={0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 128, 0)},
        class_labels={0: "background", 1: "GIS", 2: "lamella"},
    )


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    """Three projects under one folder, so a scan has something to count."""
    root = tmp_path / "collection"
    for index in range(3):
        dataset = build_dataset(root, name=f"project_{index}", acquisitions=1)
        for modality in ("SEM", "FIB"):
            tifffile.imwrite(
                str(dataset.image(modality, 0)), np.zeros((16, 16), dtype=np.uint8)
            )
    return root


@pytest.fixture
def window(qt_app, config, tmp_path: Path):
    """A window that has not scanned anything yet."""
    window = CurationWindow(
        CurationController(cache=FrameCache(), config=config),
        settings=QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat),
    )
    yield window
    destroy(window)


# ---------------------------------------------------------------------------
# When it is up
# ---------------------------------------------------------------------------

def test_the_banner_is_down_before_anything_is_scanned(window):
    assert window.banner.isHidden()


def test_the_banner_is_up_for_the_whole_of_a_scan(window, collection):
    """The scan holds the interface thread, so the only place anything can
    observe the banner is from inside the scan itself."""
    seen = []
    window.banner.set_progress = (
        lambda done, total: seen.append((window.banner.isHidden(), done, total))
    )

    window.scan(str(collection))

    assert seen, "the scan reported nothing"
    assert not any(hidden for hidden, _done, _total in seen), (
        "the banner was down while the scan was still running"
    )


def test_the_banner_comes_down_when_the_scan_finishes(window, collection):
    window.scan(str(collection))

    assert window.banner.isHidden()


def test_a_failed_scan_still_takes_the_banner_down(window, collection, monkeypatch):
    """A banner left over a window nobody can use is worse than the failure."""
    def refuse(*_args, **_kwargs):
        raise OSError("the volume went away")

    reported = []
    monkeypatch.setattr(window.controller, "scan", refuse)
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.critical",
        lambda _parent, _title, text: reported.append(text),
    )

    window.scan(str(collection))

    assert reported, "the failure was swallowed"
    assert window.banner.isHidden()


def test_the_banner_covers_the_window(window, collection):
    """Half a banner over a half-drawn window says less than either."""
    covered = []
    window.banner.set_progress = (
        lambda *_: covered.append(window.banner.geometry() == window.rect())
    )

    window.scan(str(collection))

    assert covered and all(covered)


# ---------------------------------------------------------------------------
# What it says
# ---------------------------------------------------------------------------

def test_the_count_comes_from_the_scan(window, collection):
    """Three projects were built, so the scan has three to report."""
    reported = []
    real = window.banner.set_progress
    window.banner.set_progress = (
        lambda done, total: (reported.append((done, total)), real(done, total))[0]
    )

    window.scan(str(collection))

    assert reported[0] == (0, 3), "the total is reported before the first project"
    assert reported[-1] == (3, 3)


def test_the_message_names_what_is_being_read_and_how_far(qt_app):
    host = QWidget()
    banner = ScanBanner(host)

    banner.begin("Reading collection")
    assert banner.message == "Reading collection"

    banner.set_progress(2, 7)
    assert banner.message == "Reading collection  2 of 7"


def test_a_scan_that_cannot_count_yet_says_so_without_a_bar(qt_app):
    """Finding the roots comes first and cannot say how many there will be.

    An indeterminate bar has to animate to mean anything, and animating needs
    the event loop this is painted instead of -- so there is no bar at all
    until there is a number.
    """
    host = QWidget()
    host.resize(400, 300)
    banner = ScanBanner(host)
    banner.begin("Looking")

    assert banner.message == "Looking"
    assert _painted(banner) < _painted_with_progress(banner)


def _painted(banner) -> int:
    """Return how many pixels of *banner* differ from its background."""
    banner.setGeometry(banner.parentWidget().rect())
    image = QImage(banner.size(), QImage.Format_ARGB32)
    image.fill(QColor("white"))
    banner.render(image)
    background = image.pixelColor(2, 2)
    return sum(
        image.pixelColor(x, y) != background
        for x in range(0, image.width(), 2)
        for y in range(0, image.height(), 2)
    )


def _painted_with_progress(banner) -> int:
    banner.set_progress(3, 4)
    return _painted(banner)


def test_each_update_is_put_on_the_screen(qt_app):
    """A repaint reaches a backing store. The frame is only composited when the
    main thread returns to the run loop, which a scan never does -- so without
    this the banner was painted 125 times during one scan of the reference
    collection while the window showed the desktop behind it."""
    from qtpy.QtCore import QTimer

    host = QWidget()
    host.resize(400, 300)
    banner = ScanBanner(host)
    banner.begin("Reading")
    serviced = []
    timer = QTimer()
    timer.setInterval(0)
    timer.timeout.connect(lambda: serviced.append(None))
    timer.start()

    banner.set_progress(1, 4)

    assert serviced, "the run loop was not serviced, so nothing reached the screen"


def test_input_is_let_through_again_once_the_scan_is_over(qt_app):
    """A filter left installed is a window nobody can type into."""
    from qtpy.QtCore import QCoreApplication, QEvent, QPointF, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QPushButton

    host = QWidget()
    host.resize(400, 300)
    banner = ScanBanner(host)
    banner.begin("Reading")
    banner.end()

    pressed = []
    button = QPushButton("Scan", host)
    button.pressed.connect(lambda: pressed.append(None))
    point = QPointF(5, 5)
    QCoreApplication.postEvent(
        button,
        QMouseEvent(
            QEvent.MouseButtonPress, point, point,
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
        ),
    )
    QCoreApplication.processEvents()

    assert pressed, "the banner never let go of the application"


def test_no_user_input_is_delivered_while_the_banner_is_up(qt_app):
    """The one thing that separates this from the QProgressDialog trap: a
    click cannot re-enter the scan or reach the controls being covered.

    ``ExcludeUserInputEvents`` is not enough on its own -- a press already
    posted is delivered through it -- so the banner swallows them itself."""
    from qtpy.QtCore import QCoreApplication, QEvent, QPointF, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QPushButton

    host = QWidget()
    host.resize(400, 300)
    banner = ScanBanner(host)
    banner.begin("Reading")

    pressed = []
    button = QPushButton("Scan", host)
    button.pressed.connect(lambda: pressed.append(None))
    point = QPointF(5, 5)
    QCoreApplication.postEvent(
        button,
        QMouseEvent(
            QEvent.MouseButtonPress, point, point,
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
        ),
    )

    banner.set_progress(1, 4)

    assert not pressed, "a press reached a control the banner is covering"


def test_the_banner_draws_the_mark_and_the_name(qt_app):
    """It is the one moment the window has nothing else to show."""
    host = QWidget()
    host.resize(600, 400)
    banner = ScanBanner(host)
    banner.begin("Reading")
    banner.set_progress(1, 4)

    assert _painted(banner) > 100, "the banner drew almost nothing"


# ---------------------------------------------------------------------------
# The mark beside the tabs
# ---------------------------------------------------------------------------

def test_the_tab_row_carries_the_mark(window):
    from qtpy.QtCore import Qt

    corner = window.tabs.cornerWidget(Qt.TopRightCorner)

    assert corner is not None
    labels = corner.findChildren(QLabel)
    assert any(not label.pixmap().isNull() for label in labels), "no mark"
    assert all(label.text() != PRODUCT_NAME for label in labels), (
        "the title bar and the menu bar already name it"
    )
