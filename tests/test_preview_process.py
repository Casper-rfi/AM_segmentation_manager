"""Predicting to look at: every model on every image, in memory, nothing written."""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.operations import predict_process, preview_process
from am_segmentation_manager.operations.predict_process import ProcessOutcome
from am_segmentation_manager.operations.preview_process import (
    PreviewOutcome,
    predict_previews,
    preview_out_of_process,
)

IMAGES = [Path("a.tif"), Path("b.tif")]
MODELS = [Path("first.pth"), Path("second.pth")]


def _load(model: Path):
    if model.name == "broken.pth":
        raise RuntimeError("not a generation-1 checkpoint")
    return lambda image: f"{model.stem} on {image}"


def _read(image: Path) -> str:
    if image.name == "gone.tif":
        raise FileNotFoundError(image)
    return image.stem


def test_every_model_predicts_every_image():
    outcome = predict_previews(IMAGES, MODELS, _load, _read)

    assert outcome.masks == {
        (model, image): f"{model.stem} on {image.stem}" for model in MODELS for image in IMAGES
    }
    assert not outcome.failures and not outcome.stopped


def test_an_image_is_opened_once_however_many_models_predict_it():
    opened: list[Path] = []

    predict_previews(IMAGES, MODELS, _load, lambda image: opened.append(image) or image.stem)

    assert opened == IMAGES


def test_one_model_is_loaded_at_a_time_and_only_when_its_turn_comes():
    order: list[str] = []

    def load(model: Path):
        order.append(f"load {model.stem}")
        return lambda image: order.append(f"{model.stem} {image}")

    predict_previews(IMAGES, MODELS, load, _read)

    assert order == [
        "load first", "first a", "first b", "load second", "second a", "second b",
    ]


def test_a_model_that_will_not_load_is_said_and_the_others_still_predict():
    broken = Path("broken.pth")

    outcome = predict_previews(IMAGES, [broken, MODELS[0]], _load, _read)

    assert "not a generation-1 checkpoint" in outcome.failures[broken]
    assert set(outcome.masks) == {(MODELS[0], image) for image in IMAGES}


def test_an_image_that_will_not_open_is_said_and_the_others_still_predicted():
    gone = Path("gone.tif")

    outcome = predict_previews([gone, IMAGES[0]], MODELS[:1], _load, _read)

    assert "FileNotFoundError" in outcome.failures[MODELS[0], gone]
    assert set(outcome.masks) == {(MODELS[0], IMAGES[0])}


def test_progress_counts_every_prediction_a_model_that_did_not_load_included():
    seen: list[tuple[int, int]] = []

    predict_previews(IMAGES, [Path("broken.pth"), MODELS[0]], _load, _read,
                     on_progress=lambda done, total: seen.append((done, total)))

    assert seen[-1] == (4, 4)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)


def test_being_stopped_keeps_what_was_predicted():
    def load(model: Path):
        def predict(image):
            if image == "b":
                raise KeyboardInterrupt
            return "mask"
        return predict

    outcome = predict_previews(IMAGES, MODELS, load, _read)

    assert outcome.stopped
    assert set(outcome.masks) == {(MODELS[0], IMAGES[0])}


def test_nothing_to_predict_starts_no_process(monkeypatch):
    monkeypatch.setattr(
        preview_process.multiprocessing, "get_context",
        lambda *_: pytest.fail("a process was started for nothing"),
    )

    assert preview_out_of_process([], MODELS) == PreviewOutcome()
    assert preview_out_of_process(IMAGES, []) == PreviewOutcome()


class _Context:
    """A start method whose process is never run, and whose queue holds what the child would say."""

    def __init__(self, message) -> None:
        self.message = message

    def Queue(self):  # noqa: N802 - multiprocessing's spelling
        return self

    def Process(self, **_):  # noqa: N802 - multiprocessing's spelling
        return self

    def start(self) -> None:
        pass


def test_a_child_that_dies_without_an_outcome_is_reported_as_this_kind(monkeypatch):
    died = ProcessOutcome(error="The prediction process stopped without reporting anything")
    monkeypatch.setattr(preview_process.multiprocessing, "get_context", lambda *_: _Context(None))
    monkeypatch.setattr(predict_process, "_await", lambda *_: died)
    monkeypatch.setattr(predict_process, "_reap", lambda process: None)

    outcome = preview_out_of_process(IMAGES, MODELS)

    assert isinstance(outcome, PreviewOutcome) and "without reporting" in outcome.error
