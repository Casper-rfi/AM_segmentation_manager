"""The Train tab's Predictions view: a grid of the ticked runs' models on validation images."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from am_segmentation_manager.config import load_segmentation_config
from am_segmentation_manager.gui.preview_controller import PreviewController
from am_segmentation_manager.operations.preview_process import PreviewOutcome
from am_segmentation_manager.training_listing import list_runs
from am_segmentation_manager.training_preview import read_preview_model
from conftest import destroy

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from test_cli_compare_predictions import _export, _run  # noqa: E402 - needs numpy


class Predictor:
    """Stands in for the child: class 1 everywhere, at a resolution of its own, and counts what it is asked."""

    def __init__(self) -> None:
        self.asked: list[tuple[list[Path], list[Path], str]] = []
        self.error = ""
        self.failing: set[Path] = set()

    def __call__(self, images, weights, device="auto", **_) -> PreviewOutcome:
        self.asked.append((list(images), list(weights), device))
        if self.error:
            return PreviewOutcome(error=self.error)
        return PreviewOutcome(
            {
                (model, image): np.ones((16, 24), dtype=np.uint8)
                for model in weights if model not in self.failing for image in images
            },
            {model: "RuntimeError: not a generation-1 checkpoint" for model in weights
             if model in self.failing},
        )


@pytest.fixture
def models(tmp_path: Path):
    manifest = _export(tmp_path)
    _run(tmp_path, manifest, "first")
    _run(tmp_path, manifest, "second")
    return [read_preview_model(run) for run in list_runs(tmp_path / "runs")]


@pytest.fixture
def controller(models) -> PreviewController:
    made = PreviewController(load_segmentation_config(), predict=Predictor())
    made.set_models(models)
    return made


# -- the controller -------------------------------------------------------------


def test_only_what_is_not_already_held_is_predicted(controller: PreviewController, models):
    controller.count = 2
    controller.pick()
    controller.predict()
    assert len(controller._predict.asked) == 1

    controller.predict()
    assert len(controller._predict.asked) == 1, "nothing was missing"

    controller.set_models(models[:1])
    controller.set_models(models)
    controller.predict()
    assert len(controller._predict.asked) == 1, "a run looked at again is already predicted"


def test_a_run_added_predicts_its_own_column_and_no_other(controller: PreviewController, models):
    controller.set_models(models[:1])
    controller.pick()
    controller.predict()

    controller.set_models(models)
    controller.predict()

    _, weights, _ = controller._predict.asked[-1]
    assert [model.name for model in weights] == [models[1].checkpoint.path.name]


def test_a_prediction_is_drawn_at_the_size_of_its_image_whatever_size_it_was_made_at(
    controller: PreviewController, models
):
    controller.pick()
    image = controller.images[0]
    before = controller.picture(models[0], image)

    controller.predict()
    after = controller.picture(models[0], image)

    assert before.shape[:2] == after.shape[:2] == (40, 60)
    assert before.ndim == 2 and after.ndim == 3, "the image alone, then the classes over it"
    assert controller.note(models[0], image) == ""


def test_the_image_s_own_label_is_drawn_in_place_of_a_prediction(controller, models):
    controller.pick()
    controller.predict()
    image = controller.images[0]

    predicted = controller.picture(models[0], image)
    proofread = controller.picture(models[0], image, proofread=True)

    assert not np.array_equal(predicted, proofread)
    assert np.array_equal(proofread[0, 0], [90, 90, 90]), "background is left undrawn"
    assert not np.array_equal(proofread[15, 20], [90, 90, 90]), "the labelled lamella is drawn"


def test_with_the_label_hidden_the_image_alone_is_drawn(controller, models):
    controller.pick()
    controller.predict()
    controller.label_shown = False

    assert controller.picture(models[0], controller.images[0]).ndim == 2


def test_a_model_that_would_not_load_says_why_in_its_tiles_and_is_tried_again_when_asked(
    controller: PreviewController, models
):
    controller._predict.failing = {models[1].checkpoint.path}
    controller.pick()
    controller.predict()

    assert "not a generation-1 checkpoint" in controller.note(models[1], controller.images[0])
    assert controller.missing() == ([], [])

    controller.forget_failures()
    assert controller.missing()[1] == [models[1].checkpoint.path]


def test_what_is_offered_is_said_and_why_nothing_is(controller: PreviewController, models):
    assert controller.describe_offered().startswith("3 validation image(s) of the 2 run(s) ticked")

    controller.set_models([])
    assert controller.describe_offered() == "Tick the runs to compare."


def test_a_new_draw_is_another_draw_and_the_same_one_can_be_shown_again(controller):
    controller.count = 1
    draws = {controller.pick()[0].image for _ in range(12)}

    assert len(draws) > 1
    assert controller.pick(again=False) == controller.pick(again=False)


# -- the panel ------------------------------------------------------------------


@pytest.fixture
def panel(qt_app, controller: PreviewController):
    from am_segmentation_manager.gui.widgets.predictions_panel import PredictionsPanel

    made = PredictionsPanel(controller)
    made.resize(1000, 700)
    made.show()
    qt_app.processEvents()
    yield made
    destroy(made)


def _texts(panel) -> list[str]:
    from qtpy.QtWidgets import QLabel

    return [label.text() for label in panel.grid_host.findChildren(QLabel)]


def test_picking_draws_a_row_for_each_image_and_a_column_for_each_run(panel):
    panel.count_box.setValue(2)

    panel.pick()

    assert set(panel.views) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    texts = _texts(panel)
    assert "first\nepoch 2, run's best" in texts and "second\nepoch 2, run's best" in texts
    assert any(text.endswith(".tif") and text.split("\n")[0] in ("Arctis", "Hydra") for text in texts)
    assert len(panel.controller._predict.asked) == 1


def test_a_row_says_more_of_its_image_when_hovered_than_its_heading_has_room_for(panel):
    from qtpy.QtWidgets import QLabel

    panel.pick()

    hints = [
        label.toolTip() for label in panel.grid_host.findChildren(QLabel)
        if label.text().endswith(".tif")
    ]
    assert hints and all(hint.startswith("Microscope: ") for hint in hints)
    assert all(hint.splitlines()[-1].endswith(".tif") for hint in hints), "where the file is"


def test_each_thing_an_image_is_chosen_by_offers_any_and_the_values_there_are(panel):
    box = panel.filter_boxes["microscope_model"]

    assert [box.itemText(index) for index in range(box.count())] == ["any", "Arctis", "Hydra"]
    assert not panel.filter_boxes["project_lamella_type"].isEnabled(), "the export recorded none"


def test_choosing_a_microscope_draws_only_its_images(panel):
    box = panel.filter_boxes["microscope_model"]
    box.setCurrentIndex(box.findData("Hydra"))
    box.activated.emit(box.currentIndex())
    panel.count_box.setValue(8)

    panel.pick()

    assert {image.image.name for image in panel.controller.images} == {"v2.tif", "v3.tif"}


def test_one_image_drawn_after_several_has_the_height_they_shared(panel, qt_app):
    """A row emptied keeps no share of the height for itself."""
    panel.controller.unseen_only = False
    panel.count_box.setValue(4)
    panel.pick()
    qt_app.processEvents()
    assert len(panel.controller.images) > 2
    among_several = panel.views[0, 0].height()

    panel.count_box.setValue(1)
    panel.pick()
    qt_app.processEvents()

    alone = panel.views[0, 0].height()
    assert alone > 2 * among_several
    assert alone > 0.8 * panel.grid_host.height()


def test_zooming_or_panning_one_picture_does_the_same_to_every_other(panel, qt_app):
    panel.count_box.setValue(2)
    panel.pick()
    qt_app.processEvents()
    first, *others = panel.views.values()
    fitted = first.looking_at()
    assert fitted[1:] == pytest.approx((0.5, 0.5), abs=0.02), "fitted once laid out"

    first.zoom_by(2.0)
    first.horizontalScrollBar().setValue(first.horizontalScrollBar().value() + 7)

    looking = first.looking_at()
    assert looking[0] == pytest.approx(2 * fitted[0], rel=0.01)
    assert looking[1] > fitted[1], "panned to the right"
    for view in others:
        assert view.looking_at() == pytest.approx(looking, abs=0.03)

    panel.fit_button.click()
    for view in panel.views.values():
        assert view.looking_at() == pytest.approx(fitted, abs=0.03)


def test_holding_the_button_shows_the_label_and_letting_go_the_prediction(panel, qt_app):
    panel.pick()
    qt_app.processEvents()
    view = panel.views[0, 0]
    predicted = view.grab().toImage()

    panel.proofread_button.setChecked(True)
    qt_app.processEvents()
    held = view.grab().toImage()
    panel.proofread_button.setChecked(False)
    qt_app.processEvents()

    assert held != predicted
    assert view.grab().toImage() == predicted


def test_the_key_held_over_a_picture_does_what_the_button_does(panel, qt_app):
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest

    panel.pick()
    view = panel.views[0, 0]

    QTest.keyPress(view, Qt.Key.Key_B)
    assert panel.proofread_button.isChecked()
    QTest.keyRelease(view, Qt.Key.Key_B)
    assert not panel.proofread_button.isChecked()


def test_where_a_picture_is_looking_is_kept_when_the_opacity_or_the_label_changes(panel, qt_app):
    panel.pick()
    qt_app.processEvents()
    view = panel.views[0, 0]
    view.zoom_by(3.0)
    before = view.looking_at()

    panel.alpha_slider.setValue(80)
    panel.label_check.setChecked(False)

    assert view.looking_at() == pytest.approx(before, abs=0.02)
    assert panel.controller.alpha == 0.8 and not panel.controller.label_shown


def test_a_child_that_could_not_predict_is_said_and_the_images_still_shown(panel):
    panel.controller._predict.error = "RuntimeError: the ml extra is not installed"

    panel.pick()

    assert "the ml extra is not installed" in panel.status.text()
    assert panel.views and "not predicted yet" in _texts(panel)
    assert panel.predict_button.isEnabled()


def test_with_no_run_ticked_there_is_nothing_to_pick_and_it_says_to_tick(qt_app):
    from am_segmentation_manager.gui.widgets.predictions_panel import PredictionsPanel

    made = PredictionsPanel(PreviewController(load_segmentation_config(), predict=Predictor()))
    try:
        assert made.status.text() == "Tick the runs to compare."
        assert not made.pick_button.isEnabled() and not made.predict_button.isEnabled()
    finally:
        destroy(made)


def test_the_device_predicted_on_is_the_one_the_tab_gives(panel):
    panel.device = lambda: "cuda:1"

    panel.pick()

    assert panel.controller._predict.asked[-1][2] == "cuda:1"


def test_pictures_of_two_shapes_look_at_the_same_part_of_each(qt_app):
    """Two microscopes' images differ in shape, and a scroll position means a different place in each."""
    from am_segmentation_manager.gui.widgets.image_view import ImageView, ViewLink

    wide, tall = ImageView(), ImageView()
    try:
        for view, shape in ((wide, (200, 400)), (tall, (300, 300))):
            view.resize(300, 300)
            view.show()
            view.set_array(np.zeros(shape, dtype=np.uint8))
        qt_app.processEvents()
        link = ViewLink()
        link.add(wide)
        link.add(tall)

        wide.zoom_by(4.0)
        wide.centerOn(100, 50)

        assert tall.looking_at() == pytest.approx(wide.looking_at(), abs=0.03)
        assert wide.looking_at()[1:] == pytest.approx((0.25, 0.25), abs=0.03)
    finally:
        destroy(wide)
        destroy(tall)


def test_a_view_closed_before_its_pictures_were_fitted_raises_nothing_afterwards(
    qt_app, controller, monkeypatch
):
    """A call left waiting for a widget that has gone reaches the window's failure dialog."""
    import sys

    from am_segmentation_manager.gui.widgets.predictions_panel import PredictionsPanel

    raised: list[object] = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: raised.append(error))
    made = PredictionsPanel(controller)
    made.pick()

    destroy(made)
    qt_app.processEvents()

    assert raised == []


# -- images that are not on this machine ------------------------------------------


@pytest.fixture
def gone(tmp_path: Path, models) -> Path:
    """Move the collection the runs name, and return where it was."""
    was = tmp_path / "collection"
    was.rename(tmp_path / "moved")
    return was


def test_images_that_are_not_there_are_not_predicted(controller: PreviewController, gone: Path):
    assert controller.pick() == [] and len(controller.not_found) == 3

    assert controller.missing() == ([], [])
    assert controller.predict() == "" and controller._predict.asked == []
    assert str(gone) in controller.describe_not_found()


def test_an_image_gone_alone_is_left_out_and_the_rest_predicted(
    controller: PreviewController, tmp_path: Path
):
    (tmp_path / "collection" / "v2.tif").unlink()

    drawn = controller.pick()
    controller.predict()

    assert sorted(image.image.name for image in drawn) == ["v1.tif", "v3.tif"]
    assert [sorted(image.name for image in images) for images, _, _ in controller._predict.asked] == [
        ["v1.tif", "v3.tif"]
    ]
    assert controller.describe_not_found().startswith("1 of the 3 image(s) drawn are not")


def test_an_image_that_would_not_open_is_not_predicted_until_asked_again(
    controller: PreviewController, tmp_path: Path
):
    (tmp_path / "collection" / "v2.tif").write_bytes(b"not a tiff")
    controller.pick()
    broken = next(image for image in controller.images if image.image.name == "v2.tif")

    assert isinstance(controller.shown(broken), str)
    assert broken.image not in controller.missing()[0]

    controller.forget_failures()
    assert broken.image in controller.missing()[0]


def test_what_was_not_found_is_forgotten_when_other_runs_are_looked_at(
    controller: PreviewController, models, gone: Path
):
    controller.pick()

    controller.set_models(models[:1])

    assert controller.not_found == [] and controller.describe_not_found() == ""


def test_no_image_found_draws_no_grid_and_says_where_it_looked(panel, gone: Path):
    panel.pick()

    assert not panel.views and _texts(panel) == []
    assert panel.status.text().startswith("None of the 3 image(s) drawn is on this machine.")
    assert str(gone) in panel.status.text() and "Scan the collection" in panel.status.text()
    assert panel.controller._predict.asked == []
    assert panel.pick_button.isEnabled() and not panel.predict_button.isEnabled()


# -- the width the view needs ----------------------------------------------------

#: A path of the length a collection on a shared filesystem has, holding no space to wrap at.
LONG_PATH = "/ceph/groups/structbio/" + "/".join(["adaptive_milling_project"] * 10) + "/img_0018.tif"


def test_a_long_failure_does_not_widen_the_train_tab(panel, qt_app):
    """What went wrong names a file, and a line that cannot shorten is a floor under the window."""
    panel.pick()
    qt_app.processEvents()
    before = panel.minimumSizeHint().width()
    said = f"FileNotFoundError: Image file does not exist: {LONG_PATH}"
    controller = panel.controller
    controller._failures = {
        (model.checkpoint.path, image.image): said
        for model in controller.models for image in controller.images
    }
    controller._masks.clear()

    panel._draw()
    panel._predicted(f"RuntimeError: could not read {LONG_PATH}")
    qt_app.processEvents()

    assert panel.minimumSizeHint().width() <= before
    note = panel.notes[0, 0]
    assert note.full_text() == said and note.toolTip() == said
    assert LONG_PATH in panel.status.text()


def test_a_long_folder_looked_in_does_not_widen_the_train_tab(panel, qt_app, tmp_path: Path):
    before = panel.minimumSizeHint().width()
    controller = panel.controller
    controller.pick()
    controller.models = [
        replace(model, run=replace(model.run, looked_in=Path(LONG_PATH).parent))
        for model in controller.models
    ]
    controller.images, controller.not_found = [], controller.images

    panel._build_grid()
    qt_app.processEvents()

    assert str(Path(LONG_PATH).parent) in panel.status.text()
    assert panel.minimumSizeHint().width() <= before
