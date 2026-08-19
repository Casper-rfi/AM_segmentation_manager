"""Tests for reading a checkpoint and the description beside it.

Everything here runs without torch. That is the point of the split: what a
checkpoint states about itself is read from a mapping of tensor shapes, and what
it cannot state is read from a YAML file, so both are checkable without a 170 MB
download or a GPU. Only the load itself needs the inference stack, and that is
covered by the strict-load check in ml/model.py rather than in a test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.ml.model import (
    CheckpointFacts,
    ModelDescription,
    checkpoint_facts,
    description_path,
    read_model_description,
    run_name_of,
)


class FakeTensor:
    """Just enough of a tensor to be asked its shape."""

    def __init__(self, *shape: int) -> None:
        self.shape = shape


def state_dict(classes: int = 5, in_channels: int = 3, padding: int = 0) -> dict:
    """Return a state dict shaped like the real checkpoints."""
    state = {
        "encoder._conv_stem.weight": FakeTensor(56, in_channels, 3, 3),
        "segmentation_head.0.weight": FakeTensor(classes, 128, 1, 1),
        "segmentation_head.0.bias": FakeTensor(classes),
    }
    state.update({f"encoder._blocks.{n}.weight": FakeTensor(8) for n in range(padding)})
    return state


# --- what the weights themselves say ---------------------------------------

def test_the_class_count_is_read_from_the_head():
    assert checkpoint_facts(state_dict(classes=5)).classes == 5
    assert checkpoint_facts(state_dict(classes=3)).classes == 3


def test_three_input_channels_means_the_image_is_expanded_to_rgb():
    """rgb is a property of the weights, not a setting: a FIB model trained on
    one channel has a stem that cannot accept three."""
    assert checkpoint_facts(state_dict(in_channels=3)).rgb is True
    assert checkpoint_facts(state_dict(in_channels=1)).rgb is False


def test_the_tensor_count_is_reported():
    assert checkpoint_facts(state_dict(padding=10)).tensors == 13


def test_a_checkpoint_with_no_head_is_refused():
    state = state_dict()
    del state["segmentation_head.0.weight"]

    with pytest.raises(ValueError, match="Not a segmentation checkpoint"):
        checkpoint_facts(state)


def test_a_checkpoint_with_no_stem_is_refused():
    state = state_dict()
    del state["encoder._conv_stem.weight"]

    with pytest.raises(ValueError, match="input channels"):
        checkpoint_facts(state)


def test_a_run_name_is_carried_through():
    facts = checkpoint_facts(state_dict(), run_name="training_epoch086")

    assert facts.run_name == "training_epoch086"


# --- provenance from the archive -------------------------------------------

def test_the_run_name_comes_from_the_archive_not_the_filename(tmp_path: Path):
    """The filename is chosen by whoever copied it; the archive records the run
    that produced it."""
    import zipfile

    weights = tmp_path / "AM_SEM_All_V01.pth"
    with zipfile.ZipFile(weights, "w") as archive:
        archive.writestr("Franklin_training_epoch086/data.pkl", b"")
        archive.writestr("Franklin_training_epoch086/version", b"3")

    assert run_name_of(weights) == "Franklin_training_epoch086"


def test_a_file_that_is_not_an_archive_has_no_run_name(tmp_path: Path):
    """A legacy checkpoint still loads; it just cannot say where it came from."""
    weights = tmp_path / "old.pth"
    weights.write_bytes(b"not a zip")

    assert run_name_of(weights) is None


# --- the description beside the weights ------------------------------------

def test_the_description_sits_beside_the_weights_keeping_their_extension(tmp_path: Path):
    """Appended rather than substituted, so model.pth and model.pt cannot share
    one description."""
    assert description_path(tmp_path / "model.pth").name == "model.pth.yaml"


def test_no_description_file_means_the_defaults(tmp_path: Path):
    description = read_model_description(tmp_path / "model.pth")

    assert description == ModelDescription()
    assert description.source is None


def test_the_description_says_nothing_about_modality(tmp_path: Path):
    """Which images a model is for is the caller's choice, not a claim stored
    beside the weights: nothing in a checkpoint says, and its filename is
    whatever whoever trained it typed."""
    assert not hasattr(ModelDescription(), "modality")

    weights = tmp_path / "model.pth"
    description_path(weights).write_text("modality: SEM\n", encoding="utf-8")

    with pytest.raises(ValueError, match="modality"):
        read_model_description(weights)


def test_a_description_overrides_only_what_it_names(tmp_path: Path):
    weights = tmp_path / "model.pth"
    description_path(weights).write_text("image_size: 1024\n", encoding="utf-8")

    description = read_model_description(weights)

    assert description.image_size == 1024
    assert description.normalise_first is ModelDescription().normalise_first
    assert description.source == description_path(weights)


def test_an_unknown_setting_is_refused_rather_than_ignored(tmp_path: Path):
    """A silently ignored typo runs the model on settings you did not choose."""
    weights = tmp_path / "model.pth"
    description_path(weights).write_text("image_sze: 1024\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image_sze"):
        read_model_description(weights)


def test_facts_are_frozen_so_a_prediction_cannot_be_relabelled_after_the_fact():
    facts = CheckpointFacts(run_name="r", classes=5, in_channels=3, tensors=1)

    with pytest.raises(Exception):
        facts.classes = 4  # type: ignore[misc]
