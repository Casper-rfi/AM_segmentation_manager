"""Tests for reading a checkpoint and the description beside it.

Everything but the last test runs without torch. That is the point of the split:
what a checkpoint states about itself is read from a mapping of tensor shapes, and
its generation from a YAML file, so both are checkable without a 170 MB download or
a GPU. The load itself needs the inference stack; its fit is covered by the
strict-load check in ml/model.py, and its refusal of anything but tensors here.
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from am_segmentation_manager.ml import environment as ml_env
from am_segmentation_manager.ml.model import (
    DEFAULT_GENERATION,
    GENERATION_1,
    GENERATIONS,
    CheckpointFacts,
    ModelDescription,
    check_generation,
    checkpoint_facts,
    description_path,
    load_model,
    model_input,
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


def test_an_undescribed_checkpoint_is_generation_one(tmp_path: Path):
    """Never the latest generation: a checkpoint left undescribed must not
    change how it is run when one is added."""
    description = read_model_description(tmp_path / "model.pth")

    assert description == ModelDescription()
    assert description.generation is GENERATION_1
    assert DEFAULT_GENERATION == "1.0"
    assert description.source is None


@pytest.mark.parametrize("written", ['"1.0"', "1.0", "'1.0'"])
def test_a_description_names_the_generation_however_yaml_types_it(tmp_path: Path, written):
    weights = tmp_path / "model.pth"
    description_path(weights).write_text(f"generation: {written}\n", encoding="utf-8")

    description = read_model_description(weights)

    assert description.generation is GENERATION_1
    assert description.source == description_path(weights)


def test_the_description_says_nothing_about_modality(tmp_path: Path):
    """Which images a model is for is the caller's choice, not a claim stored
    beside the weights: nothing in a checkpoint says, and its filename is
    whatever whoever trained it typed."""
    assert not hasattr(ModelDescription(), "modality")

    weights = tmp_path / "model.pth"
    description_path(weights).write_text("modality: SEM\n", encoding="utf-8")

    with pytest.raises(ValueError, match="modality"):
        read_model_description(weights)


@pytest.mark.parametrize("setting", ["image_size: 1024", "pad: true", "image_sze: 1024"])
def test_a_description_naming_anything_but_the_generation_is_refused(tmp_path: Path, setting):
    """A generation fixes how a model is run, so a setting it fixes is a request
    that cannot be honoured, and a typo would run it on settings you did not
    choose. Either way ignoring it would be worse than refusing."""
    weights = tmp_path / "model.pth"
    description_path(weights).write_text(f"generation: '1.0'\n{setting}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=setting.split(":")[0]):
        read_model_description(weights)


def test_an_unknown_generation_is_refused_naming_the_known_ones(tmp_path: Path):
    weights = tmp_path / "model.pth"
    description_path(weights).write_text("generation: '1.4fpn'\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"'1\.4fpn'.*'1\.0'"):
        read_model_description(weights)


def test_generation_one_s_settings_are_the_ones_predictions_were_made_with():
    """Written out, so that changing how generation 1 runs is a deliberate edit
    here: every prediction on disk was made with these, and they are what
    reproduce Adaptive Milling's Gen1Model."""
    assert GENERATIONS == {"1.0": GENERATION_1}
    assert (
        GENERATION_1.image_size, GENERATION_1.normalise_first, GENERATION_1.pad,
        GENERATION_1.normalise_version, GENERATION_1.resize_version,
        GENERATION_1.architecture, GENERATION_1.encoder,
        GENERATION_1.classes, GENERATION_1.in_channels,
    ) == (1536, True, False, 2, "pytorch", "FPN", "efficientnet-b6", 5, 3)


@pytest.mark.parametrize(
    "setting, value", [("pad", True), ("normalise_version", 1), ("resize_version", "cv2")]
)
def test_a_generation_asking_for_unimplemented_preprocessing_is_refused(setting, value):
    """Refused before the image is touched, rather than run as generation 1."""
    generation = replace(GENERATION_1, name="2.0", **{setting: value})

    with pytest.raises(ValueError, match=rf"'2\.0'.*{setting}"):
        model_input(None, generation, rgb=True)


def test_a_generation_one_checkpoint_is_accepted():
    check_generation(checkpoint_facts(state_dict()), GENERATION_1)


@pytest.mark.parametrize("classes, in_channels", [(4, 3), (5, 1)])
def test_a_checkpoint_of_another_shape_is_refused_as_not_generation_one(classes, in_channels):
    """Named in both counts, rather than as the state-dict error the strict load
    would give."""
    facts = checkpoint_facts(state_dict(classes=classes, in_channels=in_channels))

    with pytest.raises(RuntimeError, match="Not a generation-1.0 checkpoint"):
        check_generation(facts, GENERATION_1)


def test_facts_are_frozen_so_a_prediction_cannot_be_relabelled_after_the_fact():
    facts = CheckpointFacts(run_name="r", classes=5, in_channels=3, tensors=1)

    with pytest.raises(Exception):
        facts.classes = 4  # type: ignore[misc]


# --- loading -----------------------------------------------------------------


class RunsOnLoad:
    """An object whose unpickling makes the folder *marker*: code a file runs as it loads."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


@pytest.mark.skipif(not ml_env.has_ml_support(), reason="inference stack not installed")
def test_a_checkpoint_holding_more_than_tensors_is_refused_before_it_runs(
    tmp_path: Path, monkeypatch
):
    """Refused even where torch's default is to unpickle anything, as before 2.6."""
    import torch

    marker = tmp_path / "ran"
    weights = tmp_path / "model.pth"
    torch.save({"payload": RunsOnLoad(marker)}, weights)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    with pytest.raises(RuntimeError, match="was not loaded"):
        load_model(weights)

    assert not marker.exists()
