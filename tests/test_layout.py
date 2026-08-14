"""Tests for the filesystem naming conventions in :mod:`layout`.

The classification and resolution functions are pure, so they are tested
exhaustively against every directory-name form seen in the real datasets. The
few functions that touch the filesystem are tested against small trees built in
tmp_path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from am_segmentation_manager.layout import (
    ImageDirectory,
    build_image_sets,
    candidate_labels_directory_names,
    classify_image_directory,
    directory_role,
    image_files_by_name,
    is_labels_directory,
    labels_directory_name,
    resolve_labels_directory,
    state_path,
    subdirectories_by_name,
)


# ---------------------------------------------------------------------------
# Image directory classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("sem", ImageDirectory("SEM", None)),
        ("SEM", ImageDirectory("SEM", None)),
        ("fib", ImageDirectory("FIB", None)),
        ("FIB", ImageDirectory("FIB", None)),
        ("SEM_8", ImageDirectory("SEM", "8")),
        ("SEM_8_fi_1", ImageDirectory("SEM", "8_fi_1")),
        ("SEM_8_fi_16", ImageDirectory("SEM", "8_fi_16")),
        ("sem_5kv", ImageDirectory("SEM", "5kv")),
        ("sem_defocus", ImageDirectory("SEM", "defocus")),
        ("fib_lowmag", ImageDirectory("FIB", "lowmag")),
    ],
)
def test_classify_image_directory_recognises_image_directories(name, expected):
    assert classify_image_directory(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "labels",           # legacy shared SEM labels
        "labels_fib",       # legacy shared FIB labels
        "labels_FIB",       # legacy, as written on disk
        "sem_labels",       # current shared labels
        "fib_labels",
        "sem_fi8_labels",   # private labels for a variant
        "predictions",
        "predictions_fib",
        "sem_predictions",
        "plots",
        "deselected",
        "corrected",
        "Alignment",        # Autolamella images that are out of scope
        "Milling",
        "adaptive_polish",
        "semantic",         # a name that merely starts with the modality
        "fibre",
    ],
)
def test_classify_image_directory_rejects_other_directories(name):
    assert classify_image_directory(name) is None


@pytest.mark.parametrize("variant", ["labels", "predictions", "plots", "deselected", "corrected"])
def test_reserved_variants_are_rejected(variant):
    """A reserved variant would collide with another directory role."""
    assert classify_image_directory(f"sem_{variant}") is None


def test_variant_case_is_preserved():
    """The variant is opaque, so it is reported as written on disk."""
    assert classify_image_directory("SEM_LowMag") == ImageDirectory("SEM", "LowMag")


@pytest.mark.parametrize(
    "name, role",
    [
        ("sem", "image"),
        ("sem_5kv", "image"),
        ("sem_labels", "labels"),
        ("labels_FIB", "labels"),
        ("predictions", "predictions"),
        ("plots", "plots"),
        ("deselected", "state"),
        ("Alignment", "unknown"),
    ],
)
def test_directory_role(name, role):
    assert directory_role(name) == role


# ---------------------------------------------------------------------------
# Label directory resolution
# ---------------------------------------------------------------------------

def test_labels_directory_name_follows_the_image_directory():
    assert labels_directory_name("sem_5kv") == "sem_5kv_labels"


def test_candidates_are_private_then_shared_then_legacy():
    assert candidate_labels_directory_names("sem_5kv", "SEM") == [
        "sem_5kv_labels",
        "sem_labels",
        "labels",
    ]


def test_base_image_directory_collapses_private_and_shared_candidates():
    """For ``sem/`` the private and shared names coincide, which is intended."""
    assert candidate_labels_directory_names("sem", "SEM") == ["sem_labels", "labels"]


def test_private_labels_win_over_shared(tmp_path: Path):
    stage = tmp_path / "stage"
    (stage / "sem_5kv").mkdir(parents=True)
    (stage / "sem_5kv_labels").mkdir()
    (stage / "sem_labels").mkdir()

    directory, exists, legacy = resolve_labels_directory(
        stage / "sem_5kv", "SEM", subdirectories_by_name(stage)
    )
    assert directory == stage / "sem_5kv_labels"
    assert exists is True
    assert legacy is False


def test_shared_labels_used_when_no_private_directory(tmp_path: Path):
    stage = tmp_path / "stage"
    (stage / "sem_5kv").mkdir(parents=True)
    (stage / "sem_labels").mkdir()

    directory, exists, _ = resolve_labels_directory(
        stage / "sem_5kv", "SEM", subdirectories_by_name(stage)
    )
    assert directory == stage / "sem_labels"
    assert exists is True


def test_legacy_label_directory_is_matched_case_insensitively(tmp_path: Path):
    """``labels_FIB`` must resolve on case-sensitive filesystems too."""
    stage = tmp_path / "stage"
    (stage / "FIB").mkdir(parents=True)
    (stage / "labels_FIB").mkdir()

    directory, exists, legacy = resolve_labels_directory(
        stage / "FIB", "FIB", subdirectories_by_name(stage)
    )
    assert directory == stage / "labels_FIB"
    assert exists is True
    assert legacy is True


def test_missing_labels_resolve_to_the_private_name(tmp_path: Path):
    """With nothing on disk, the answer is where a label would be written."""
    stage = tmp_path / "stage"
    (stage / "sem_5kv").mkdir(parents=True)

    directory, exists, legacy = resolve_labels_directory(
        stage / "sem_5kv", "SEM", subdirectories_by_name(stage)
    )
    assert directory == stage / "sem_5kv_labels"
    assert exists is False
    assert legacy is False


# ---------------------------------------------------------------------------
# Image sets
# ---------------------------------------------------------------------------

def _make_image(path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_bytes(b"")


def test_build_image_sets_detects_sharing(tmp_path: Path):
    """The legacy multi-condition layout: five SEM sets sharing one label set."""
    stage = tmp_path / "lam_010"
    for name in ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4", "SEM_8_fi_16"):
        _make_image(stage / name, "img_0004.tif")
    _make_image(stage / "FIB", "img_0004.tif")
    _make_image(stage / "labels", "img_0004.tif")
    _make_image(stage / "labels_FIB", "img_0004.tif")

    image_sets = build_image_sets(stage)
    by_name = {image_set.name: image_set for image_set in image_sets}

    assert set(by_name) == {"SEM_8", "SEM_8_fi_1", "SEM_8_fi_4", "SEM_8_fi_16", "FIB"}
    for name in ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4", "SEM_8_fi_16"):
        assert by_name[name].labels_directory == stage / "labels"
        assert by_name[name].labels_are_shared is True
        assert by_name[name].labels_are_legacy is True
    assert by_name["FIB"].labels_directory == stage / "labels_FIB"
    assert by_name["FIB"].labels_are_shared is False


def test_build_image_sets_reports_private_labels_as_unshared(tmp_path: Path):
    stage = tmp_path / "stage"
    _make_image(stage / "sem", "img_000.tif")
    _make_image(stage / "sem_5kv", "img_000.tif")
    _make_image(stage / "sem_labels", "img_000.tif")
    _make_image(stage / "sem_5kv_labels", "img_000.tif")

    by_name = {image_set.name: image_set for image_set in build_image_sets(stage)}
    assert by_name["sem"].labels_are_shared is False
    assert by_name["sem_5kv"].labels_are_shared is False
    assert by_name["sem_5kv"].labels_directory == stage / "sem_5kv_labels"


def test_build_image_sets_orders_sem_before_fib_and_base_before_variants(tmp_path: Path):
    stage = tmp_path / "stage"
    for name in ("fib", "sem_5kv", "sem", "fib_lowmag"):
        _make_image(stage / name, "img_000.tif")

    assert [image_set.name for image_set in build_image_sets(stage)] == [
        "sem", "sem_5kv", "fib", "fib_lowmag",
    ]


def test_build_image_sets_ignores_label_and_plot_directories(tmp_path: Path):
    stage = tmp_path / "stage"
    _make_image(stage / "sem", "img_000.tif")
    (stage / "plots").mkdir()
    (stage / "predictions" / "clean").mkdir(parents=True)
    _make_image(stage / "sem_labels", "img_000.tif")

    assert [image_set.name for image_set in build_image_sets(stage)] == ["sem"]


def test_build_image_sets_on_a_stage_with_no_image_directories(tmp_path: Path):
    stage = tmp_path / "stage"
    (stage / "plots").mkdir(parents=True)
    assert build_image_sets(stage) == []


# ---------------------------------------------------------------------------
# State locations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "start, state, expected",
    [
        ("sem/img.tif", "deselected", "sem/deselected/img.tif"),
        ("sem/deselected/img.tif", "selected", "sem/img.tif"),
        ("sem/deselected/img.tif", "corrected", "sem/corrected/img.tif"),
        ("sem/corrected/img.tif", "corrected", "sem/corrected/img.tif"),
    ],
)
def test_state_path_moves_between_sibling_state_directories(start, state, expected, tmp_path):
    assert state_path(tmp_path / start, state) == tmp_path / expected


def test_state_path_rejects_an_unknown_state(tmp_path: Path):
    with pytest.raises(ValueError):
        state_path(tmp_path / "sem" / "img.tif", "archived")


def test_image_files_by_name_spans_state_directories(tmp_path: Path):
    directory = tmp_path / "sem"
    _make_image(directory, "a.tif")
    _make_image(directory / "deselected", "b.tif")
    _make_image(directory / "corrected", "c.tif")
    (directory / "notes.txt").write_text("ignored")

    found = image_files_by_name(directory)
    assert set(found) == {"a.tif", "b.tif", "c.tif"}
    assert found["b.tif"] == directory / "deselected" / "b.tif"


def test_subdirectories_by_name_skips_hidden_and_files(tmp_path: Path):
    (tmp_path / "sem").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "file.tif").write_bytes(b"")

    assert set(subdirectories_by_name(tmp_path)) == {"sem"}


def test_subdirectories_by_name_survives_an_unreadable_directory(tmp_path: Path):
    """A directory that cannot be listed must not abort a whole survey."""
    missing = tmp_path / "does-not-exist"
    assert subdirectories_by_name(missing) == {}


def test_is_labels_directory_covers_current_and_legacy_names():
    assert is_labels_directory("sem_labels")
    assert is_labels_directory("labels_FIB")
    assert not is_labels_directory("sem")
