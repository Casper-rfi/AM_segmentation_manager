"""The conflict resolver's window: two pages, a tree beside a viewer, and keys.

What is decided is the model's, tested without a display in
`test_gui_conflict_resolution.py`. These check that the window shows it,
offers nothing in advance, and sends each gesture to the right conflict.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from conftest import destroy

# Must be set before any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HAS_QT = importlib.util.find_spec("qtpy") is not None
HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "tifffile")
)

pytestmark = pytest.mark.skipif(
    not (HAS_QT and HAS_IMAGE_STACK), reason="Qt binding or image stack not installed"
)

if HAS_QT and HAS_IMAGE_STACK:
    import numpy as np
    import tifffile
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QKeyEvent
    from qtpy.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

    from am_segmentation_manager.config import SegmentationConfig
    from am_segmentation_manager.gui.report import ConflictRow
    from am_segmentation_manager.gui.widgets.conflict_dialog import ConflictDialog
    from am_segmentation_manager.gui.widgets.conflict_field_page import UNSET
    from am_segmentation_manager.gui.widgets.conflict_tree import SIDE_COLUMNS
    from am_segmentation_manager.gui.widgets.conflict_viewer import ConflictView
    from am_segmentation_manager.models import ImageRecord
    from am_segmentation_manager.operations.merge import OURS, THEIRS

ROOT = Path("/collection")
P = "session_a/Auto-1"
Q = "session_a/Auto-2"


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config():
    return SegmentationConfig(
        class_colors={0: (0, 0, 0), 2: (255, 0, 0)},
        class_labels={0: "background", 2: "crack"},
    )


@pytest.fixture
def opened(qt_app, config):
    """Return a function opening the dialog over some rows, closed afterwards."""
    dialogs = []

    def open_over(rows, **kwargs):
        dialog = ConflictDialog(None, rows, config, **kwargs)
        dialogs.append(dialog)
        return dialog

    yield open_over
    for dialog in dialogs:
        destroy(dialog)


def _group(project, lamella, number, *, corrected=False, **paths) -> ConflictRow:
    """One label group in conflict, with no files behind it unless given."""
    stage = ROOT / project / lamella / "adaptive_polish_2025-09-05-01-35-01PM"
    name = f"{lamella}_AP_img_{number:03d}_SEM.tif"
    record = ImageRecord(
        image_path=stage / "sem" / name, lamella=lamella, modality="SEM",
        stage_path=stage,
    )
    key = f"{project}/{lamella}/{stage.name}/sem_labels/{name}"
    return ConflictRow(
        key=key, title=key, corrected=corrected, project=project,
        lamella=lamella, record=record, **paths,
    )


def _field(project, name, here="a", there="b") -> ConflictRow:
    return ConflictRow(
        key=f"{project}:{name}", title=name, project=project, metadata_field=name,
        details=(f"{here!r} here, {there!r} in the copy",),
        our_value=here, their_value=there,
    )


def _node(nodes, *captions):
    node = None
    for caption in captions:
        node = next(candidate for candidate in nodes if candidate.caption == caption)
        nodes = node.children
    return node


def _labels(tmp_path: Path, ours_shape=(32, 32), theirs_shape=(32, 32)):
    """Write an acquisition and two labels that differ, and return the paths."""
    image, ours, theirs = (tmp_path / f"{n}.tif" for n in ("img", "ours", "theirs"))
    tifffile.imwrite(str(image), np.zeros((32, 32), dtype=np.uint8))
    tifffile.imwrite(str(ours), np.zeros(ours_shape, dtype=np.uint8))
    painted = np.zeros(theirs_shape, dtype=np.uint8)
    painted[4:8, 4:8] = 2
    tifffile.imwrite(str(theirs), painted)
    return {"image": image, "ours": ours, "theirs": theirs}


# ---------------------------------------------------------------------------
# The two pages, and what is offered when
# ---------------------------------------------------------------------------

def test_merge_is_offered_only_once_every_conflict_on_both_pages_is_decided(opened):
    rows = [_field(P, "description"), _group(P, "01-cat", 1), _group(P, "01-cat", 2)]
    dialog = opened(rows)

    assert dialog.pages.currentWidget() is dialog.field_page, "metadata comes first"
    assert not dialog.next.isHidden() and not dialog.next.isEnabled()
    assert dialog.merge.isHidden()

    fields = dialog.field_page
    fields.table.cellClicked.emit(fields.row_of(rows[0].key), SIDE_COLUMNS[THEIRS])
    assert dialog.next.isEnabled()

    dialog.next.click()
    assert dialog.pages.currentWidget() is dialog.label_page
    assert not dialog.back.isHidden()
    assert not dialog.merge.isHidden() and not dialog.merge.isEnabled()

    dialog.label_page.decide_everything(OURS)
    assert dialog.merge.isEnabled()
    assert dialog.resolutions == {
        rows[0].key: THEIRS, rows[1].key: OURS, rows[2].key: OURS
    }


def test_a_page_with_nothing_on_it_is_left_out(opened):
    dialog = opened([_group(P, "01-cat", 1)])

    assert dialog.field_page is None
    assert dialog.pages.count() == 1
    assert dialog.back.isHidden() and dialog.next.isHidden()
    assert not dialog.merge.isHidden()


def test_nothing_is_offered_as_the_answer(opened):
    """Qt makes a dialog's first accepting button its default and draws it in
    the accent; a default here is a rule about which tree wins."""
    dialog = opened([_group(P, "01-cat", 1)])
    page = dialog.label_page

    assert dialog.buttons.button(QDialogButtonBox.Cancel).isDefault()
    for button in (dialog.merge, page.keep, page.take):
        assert not button.isDefault() and not button.autoDefault()
    assert not page.keep.isChecked() and not page.take.isChecked()
    assert dialog.resolutions == {}


def test_cancelling_asks_before_losing_decisions_and_only_then(opened, monkeypatch):
    asked: list = []
    answer = [QMessageBox.No]
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: asked.append(args) or answer[0]
    )

    untouched = opened([_group(P, "01-cat", 1)])
    closed: list = []
    untouched.rejected.connect(lambda: closed.append("untouched"))
    untouched.reject()
    assert asked == [] and closed == ["untouched"], "nothing to lose, nothing asked"

    dialog = opened([_group(P, "01-cat", 1)])
    dialog.rejected.connect(lambda: closed.append("decided"))
    dialog.label_page.decide_everything(OURS)
    dialog.reject()
    assert len(asked) == 1 and "decided" not in closed

    answer[0] = QMessageBox.Yes
    dialog.reject()
    assert "decided" in closed


# ---------------------------------------------------------------------------
# The metadata page
# ---------------------------------------------------------------------------

def _headings(table) -> list[str]:
    """Return the project headings the table lists, in order."""
    return [
        table.item(row, 0).text()
        for row in range(table.rowCount())
        if table.columnSpan(row, 0) > 1
    ]


def test_the_metadata_page_sets_the_two_values_side_by_side(opened):
    rows = [_field(P, "description", "here", "there"), _field(P, "grid_id", None, "G7")]
    page = opened(rows).field_page
    table = page.table

    first = page.row_of(rows[0].key)
    assert table.item(first, SIDE_COLUMNS[OURS]).text() == "here"
    assert table.item(first, SIDE_COLUMNS[THEIRS]).text() == "there"
    assert table.item(page.row_of(rows[1].key), SIDE_COLUMNS[OURS]).text() == UNSET
    assert _headings(table) == [P], "a project heads its fields"


def test_clicking_a_value_chooses_it_and_clicking_it_again_undecides(opened):
    rows = [_field(P, "description")]
    dialog = opened(rows)
    page = dialog.field_page
    row = page.row_of(rows[0].key)

    page.table.cellClicked.emit(row, SIDE_COLUMNS[THEIRS])
    assert dialog.resolutions == {rows[0].key: THEIRS}
    assert page.table.item(row, SIDE_COLUMNS[THEIRS]).font().bold()
    assert not page.table.item(row, SIDE_COLUMNS[OURS]).font().bold()

    page.table.cellClicked.emit(row, SIDE_COLUMNS[THEIRS])
    assert dialog.resolutions == {}

    page.table.cellClicked.emit(row, 0)
    assert dialog.resolutions == {}, "a field's name chooses nothing"


def test_a_folder_lists_every_project_under_it_and_its_mark_decides_them(opened):
    rows = [
        _field(P, "description"),
        _field(Q, "description"),
        _field("session_b/Auto-3", "description"),
    ]
    dialog = opened(rows)
    page = dialog.field_page
    folder = _node(dialog.resolution.fields, "session_a")

    page.tree.setCurrentItem(page.tree.item_of(folder))
    assert _headings(page.table) == [P, Q]

    page.tree.itemClicked.emit(page.tree.item_of(folder), SIDE_COLUMNS[OURS])
    assert dialog.resolutions == {rows[0].key: OURS, rows[1].key: OURS}


def test_a_field_chosen_in_the_tree_lists_its_whole_project(opened):
    rows = [_field(P, "description"), _field(P, "grid_id"), _field(Q, "description")]
    dialog = opened(rows)
    page = dialog.field_page

    page.tree.setCurrentItem(page.tree.item_of(dialog.resolution.conflict(rows[1].key)))

    assert _headings(page.table) == [P]
    assert {page.row_of(rows[0].key), page.row_of(rows[1].key)}
    with pytest.raises(KeyError):
        page.row_of(rows[2].key)


# ---------------------------------------------------------------------------
# The label page
# ---------------------------------------------------------------------------

def test_clicking_a_mark_decides_everything_under_it_and_a_full_one_undecides(opened):
    rows = [_group(P, "01-cat", 1), _group(P, "01-cat", 2), _group(Q, "01-cat", 1)]
    dialog = opened(rows)
    tree = dialog.label_page.tree
    item = tree.item_of(_node(dialog.resolution.labels, "session_a", "Auto-1"))

    tree.itemClicked.emit(item, SIDE_COLUMNS[THEIRS])
    assert dialog.resolutions == {rows[0].key: THEIRS, rows[1].key: THEIRS}

    tree.itemClicked.emit(item, SIDE_COLUMNS[THEIRS])
    assert dialog.resolutions == {}


def test_the_keys_act_on_the_conflict_on_screen_never_a_heading(opened):
    rows = [_group(P, "01-cat", 1), _group(P, "01-cat", 2)]
    dialog = opened(rows)
    page = dialog.label_page
    lamella = _node(dialog.resolution.labels, "session_a", "Auto-1", "01-cat")

    page.tree.setCurrentItem(page.tree.item_of(lamella))
    assert page.current_key == rows[0].key, "a heading shows its first conflict"

    page.decide_current(THEIRS)
    assert dialog.resolutions == {rows[0].key: THEIRS}


def test_moving_on_after_deciding_is_a_switch_that_opens_off(opened):
    rows = [_group(P, "01-cat", n) for n in (1, 2, 3)]
    page = opened(rows).label_page
    assert page.advance.isChecked() is False

    page.show_conflict(rows[0].key)
    page.decide_current(OURS)
    assert page.current_key == rows[0].key

    page.advance.setChecked(True)
    page.decide_current(THEIRS)
    assert page.current_key == rows[1].key


def test_a_setting_can_open_it_moving_on(opened):
    assert opened([_group(P, "01-cat", 1)], advance=True).label_page.advance.isChecked()


def test_the_arrow_keys_step_through_a_lamella_and_across_lamellae(opened):
    rows = [_group(P, "01-cat", 1), _group(P, "01-cat", 2), _group(P, "02-dog", 1)]
    dialog = opened(rows)
    page = dialog.label_page
    page.show_conflict(rows[0].key)

    page.step_within(1)
    assert page.current_key == rows[1].key
    page.step_within(1)
    assert page.current_key == rows[1].key, "a lamella's end is where it stops"

    page.step_across(1)
    assert page.current_key == rows[2].key
    assert page.tree.node_at(page.tree.currentItem()) is dialog.resolution.conflict(
        rows[2].key
    ), "the tree follows"
    page.step_across(-1)
    assert page.current_key == rows[0].key


def test_the_tree_hands_the_page_its_keys(opened):
    """A tree takes the arrows to move between rows and a letter to search,
    both before any parent sees them."""
    tree = opened([_group(P, "01-cat", 1)]).label_page.tree

    for key in (Qt.Key_B, Qt.Key_K, Qt.Key_T, Qt.Key_Left, Qt.Key_Down):
        event = QKeyEvent(QKeyEvent.KeyPress, key, Qt.NoModifier)
        tree.keyPressEvent(event)
        assert not event.isAccepted(), f"the tree kept {key}"


def test_proofread_work_is_marked_where_it_is_decided(opened):
    rows = [_group(P, "01-cat", 1, corrected=True), _group(P, "01-cat", 2)]
    dialog = opened(rows)
    page = dialog.label_page
    resolution = dialog.resolution

    page.show_conflict(rows[0].key)
    assert not page.proofread.isHidden()
    page.show_conflict(rows[1].key)
    assert page.proofread.isHidden()

    conflict = page.tree.item_of(resolution.conflict(rows[0].key))
    lamella = page.tree.item_of(_node(resolution.labels, "session_a", "Auto-1", "01-cat"))
    assert "Proofread" in conflict.toolTip(0)
    assert "1 of the 2" in lamella.toolTip(0)


# ---------------------------------------------------------------------------
# The viewer
# ---------------------------------------------------------------------------

def test_holding_b_shows_the_copys_label_and_says_so(opened, tmp_path: Path):
    """Flicked between in one place, not laid side by side: side by side halves
    the resolution of each, and a crack is a fraction of a percent of a frame."""
    page = opened([_group(P, "01-cat", 1, **_labels(tmp_path))]).label_page

    page.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_B, Qt.NoModifier))
    assert page.viewer.showing_theirs is True
    assert "copy" in page.viewer.badge.text(), "nothing said which is on screen"

    page.keyReleaseEvent(QKeyEvent(QKeyEvent.KeyRelease, Qt.Key_B, Qt.NoModifier))
    assert page.viewer.showing_theirs is False
    assert page.viewer.badge.text() == ""


def test_labels_of_different_sizes_are_still_shown(qt_app, config, tmp_path: Path):
    """Seeing that one is the wrong size is the finding. Only the pixel to
    pixel mask has to go, because there is no such pairing to draw."""
    view = ConflictView(config)
    try:
        view.show_row(_group(P, "01-cat", 1, **_labels(tmp_path, theirs_shape=(16, 16))))
        assert view.differences.isEnabled() is False
        assert "different sizes" in view.differences.toolTip()
        view.set_comparing(True)
    finally:
        destroy(view)


def test_a_state_conflict_shows_its_image_and_misses_no_label(qt_app, config, tmp_path):
    """Two words against two words, both already on the row; there is no pair
    of labels whose absence is worth reporting."""
    image = _labels(tmp_path)["image"]
    row = _group(P, "01-cat", 1, image=image, details=("the state: selected, corrected",))
    view = ConflictView(config)
    try:
        view.show_row(row)
        assert row.is_label is False
        assert view.problems.isHidden(), view.problems.text()
    finally:
        destroy(view)
