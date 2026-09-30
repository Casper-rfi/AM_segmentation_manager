"""Tests for imaging conditions and shared labels in the curation view model.

Two properties matter here and both are about honesty rather than mechanics:
the variant on screen must never decide what a state change moves, and the
window must be able to say how much will move before it moves.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.config import SegmentationConfig
from am_segmentation_manager.gui.cache import FrameCache, LoadRequest
from am_segmentation_manager.gui.controller import CurationController, StateChangeReach
from am_segmentation_manager.scanner import Scanner

CONFIG = SegmentationConfig(
    class_colors={0: (0, 0, 0), 1: (255, 0, 0)}, class_labels={0: "background", 1: "GIS"}
)

SEM_VARIANTS = ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4")


def make_project(root: Path) -> Path:
    """Build the legacy layout: SEM conditions sharing a label, FIB with its own."""
    project = root / "project"
    lamella = project / "lam_010"
    for directory in (*SEM_VARIANTS, "FIB", "labels", "labels_FIB"):
        (lamella / directory).mkdir(parents=True)
        (lamella / directory / "img_0004.tif").write_bytes(b"")
    (project / "AM_metadata.yaml").write_text(
        "metadata:\n  sample: HeLa\n",
        encoding="utf-8",
    )
    return project


def make_controller(root: Path) -> CurationController:
    """Return a controller over a real scanned project, with a stub cache."""
    cache = FrameCache(load_image=lambda path, label=None: f"image:{path.name}")
    controller = CurationController(
        cache=cache, config=CONFIG,
        scan=lambda path, on_progress=None: Scanner(path).build_index(on_progress),
    )
    controller.scan(root)
    return controller


# ---------------------------------------------------------------------------
# The condition on screen is a view choice
# ---------------------------------------------------------------------------

def test_the_conditions_of_a_frame_are_offered_base_first(tmp_path: Path):
    make_project(tmp_path)
    controller = make_controller(tmp_path)

    assert controller.condition_choices("SEM") == [
        ("8", "SEM_8"), ("8_fi_1", "SEM_8_fi_1"), ("8_fi_4", "SEM_8_fi_4"),
    ]


def test_a_modality_with_one_condition_offers_no_choice(tmp_path: Path):
    make_project(tmp_path)
    controller = make_controller(tmp_path)

    assert controller.condition_choices("FIB") == [("base", "FIB")]


def test_the_viewed_condition_can_be_changed(tmp_path: Path):
    make_project(tmp_path)
    controller = make_controller(tmp_path)

    controller.set_condition("sem_8_FI_1", "SEM")

    assert controller.current_condition("SEM") == "SEM_8_fi_1"
    assert controller.current_condition("FIB") == "FIB"


def test_a_stage_holding_two_families_offers_each_base(tmp_path: Path):
    """Keyed by variant, both bases were None and only one could be reached."""
    lamella = tmp_path / "project" / "lam_010"
    for directory in ("sem", "sem_fi4", "sem-5kV", "FIB"):
        (lamella / directory).mkdir(parents=True)
        (lamella / directory / "img_0004.tif").write_bytes(b"")
    (tmp_path / "project" / "AM_metadata.yaml").write_text("metadata:\n", encoding="utf-8")
    controller = make_controller(tmp_path)

    assert controller.condition_choices("SEM") == [
        ("sem", "sem"), ("sem [fi4]", "sem_fi4"), ("sem-5kV", "sem-5kV"),
    ]
    controller.set_mode("sem")
    controller.set_condition("sem-5kV", "SEM")
    assert controller.current_image_path().parent.name == "sem-5kV"


def test_the_viewed_condition_does_not_change_what_a_state_change_moves(tmp_path: Path):
    """The whole point of the design: viewing is not targeting."""
    make_project(tmp_path)
    controller = make_controller(tmp_path)

    before = controller.reach()
    controller.set_condition("SEM_8_fi_4", "SEM")
    after = controller.reach()

    assert before == after


def test_switching_condition_changes_which_image_is_prepared(tmp_path: Path):
    make_project(tmp_path)
    controller = make_controller(tmp_path)
    controller.set_condition("SEM_8_fi_1", "SEM")

    requests = controller.pending_loads()

    assert LoadRequest(0, "SEM", "SEM_8_fi_1") in requests
    assert LoadRequest(0, "SEM", None) not in requests


def test_both_conditions_stay_cached_when_switching_between_them(tmp_path: Path):
    """Comparing two conditions must not evict either."""
    make_project(tmp_path)
    controller = make_controller(tmp_path)

    controller.cache.image(0, "SEM", "SEM_8")
    controller.cache.image(0, "SEM", "SEM_8_fi_1")

    assert controller.cache.cached_image(0, "SEM", "SEM_8") is not None
    assert controller.cache.cached_image(0, "SEM", "SEM_8_fi_1") is not None


def test_a_frame_missing_a_condition_falls_back_to_its_base_image(tmp_path: Path):
    """Navigating past an acquisition captured under fewer conditions."""
    project = make_project(tmp_path)
    (project / "lam_010" / "SEM_8" / "img_0009.tif").write_bytes(b"")
    (project / "lam_010" / "labels" / "img_0009.tif").write_bytes(b"")
    controller = make_controller(tmp_path)
    controller.set_condition("SEM_8_fi_1", "SEM")

    # Frame 1 is img_0009, which only the base condition captured.
    shown = controller.cache.image(1, "SEM", "SEM_8_fi_1")
    assert shown is not None
    # Held once, under the image shown, however it was asked for.
    assert controller.cache.cached_image(1, "SEM", None) is shown
    assert controller.cache.cached_image(1, "SEM", "SEM_8") is shown


# ---------------------------------------------------------------------------
# The reach line
# ---------------------------------------------------------------------------

def test_reach_counts_the_expanded_set_not_the_scope(tmp_path: Path):
    make_project(tmp_path)
    controller = make_controller(tmp_path)
    controller.set_target("SEM")

    reach = controller.reach()

    assert reach.images == len(SEM_VARIANTS)
    assert reach.labels == 1
    assert reach.shared == len(SEM_VARIANTS) - 1
    assert len(reach.shared_images) == reach.shared
    assert all("/" in name for name in reach.shared_images)


def test_reach_on_an_unshared_modality_reports_no_sharing(tmp_path: Path):
    make_project(tmp_path)
    controller = make_controller(tmp_path)
    controller.set_target("FIB")

    reach = controller.reach()

    assert reach.images == 1
    assert reach.shared == 0
    assert reach.shared_note() == ""


@pytest.mark.parametrize(
    "reach, expected",
    [
        (
            StateChangeReach(images=0, labels=0, shared=0, lamellae=0),
            "Nothing selected",
        ),
        (
            StateChangeReach(images=1, labels=1, shared=0, lamellae=1),
            "Covers 1 image, 1 label, 1 lamella",
        ),
        (
            StateChangeReach(images=4, labels=1, shared=3, lamellae=1),
            "Covers 4 images, 1 label, 1 lamella",
        ),
        (
            StateChangeReach(images=2, labels=2, shared=0, lamellae=2),
            "Covers 2 images, 2 labels, 2 lamellae",
        ),
        (
            StateChangeReach(images=1, labels=0, shared=0, lamellae=1),
            "Covers 1 image, no label yet, 1 lamella",
        ),
        # Split by modality only when both are actually there: "12 SEM + 0 FIB"
        # would report the edit target as though it were data.
        (
            StateChangeReach(
                images=24, labels=12, shared=0, lamellae=3,
                modalities=(("SEM", 12), ("FIB", 12)),
            ),
            "Covers 12 SEM + 12 FIB images, 12 labels, 3 lamellae",
        ),
        (
            StateChangeReach(
                images=12, labels=12, shared=0, lamellae=3,
                modalities=(("SEM", 12), ("FIB", 0)),
            ),
            "Covers 12 images, 12 labels, 3 lamellae",
        ),
    ],
)
def test_reach_summary_wording(reach, expected):
    assert reach.summary() == expected


@pytest.mark.parametrize(
    "shared, expected",
    [
        (0, ""),
        (1, "1 more shares a label with this one"),
        (3, "3 more share a label with this one"),
    ],
)
def test_shared_note_wording(shared, expected):
    reach = StateChangeReach(images=4, labels=1, shared=shared, lamellae=1)
    assert reach.shared_note() == expected


def test_reach_is_always_available_even_with_nothing_selected():
    controller = CurationController(
        cache=FrameCache(load_image=lambda path, label=None: "x"),
        config=CONFIG,
        scan=lambda path, on_progress=None: Scanner(path).build_index(on_progress),
    )
    assert controller.reach().summary() == "Nothing selected"
