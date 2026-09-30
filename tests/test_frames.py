"""Tests for building per-lamella chronological frames.

These tests construct model objects directly rather than scanning a folder
tree: build_lamella_frames is a pure transformation of the scanned model, and
testing it without the filesystem keeps the cases readable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.frames import LamellaFrame, build_lamella_frames
from am_segmentation_manager.models import ImageRecord, Lamella, Stage


def make_record(name: str, modality: str) -> ImageRecord:
    """Build an image record for a file named *name*."""
    folder = "sem" if modality == "SEM" else "fib"
    return ImageRecord(image_path=Path(folder) / name, modality=modality)


def make_stage(
    name: str,
    number: str,
    images: list[ImageRecord] | None = None,
    plots: list[str] | None = None,
) -> Stage:
    return Stage(
        name=name,
        path=Path(name),
        number=number,
        images=images or [],
        plots=[Path("plots") / plot for plot in (plots or [])],
    )


def make_lamella(stages: list[Stage]) -> Lamella:
    return Lamella(name="01-alive-muskox", path=Path("01-alive-muskox"), number="01", stages=stages)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def test_sem_and_fib_of_one_acquisition_share_a_frame():
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("lam_AP_img_000_SEM.tif", "SEM"),
        make_record("lam_AP_img_000_FIB.tif", "FIB"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].record("SEM").image_path.name == "lam_AP_img_000_SEM.tif"
    assert frames[0].record("FIB").image_path.name == "lam_AP_img_000_FIB.tif"
    assert frames[0].image_number == 0


def test_pairing_uses_image_number_not_filename():
    """Older projects name the file inside fib/ with an _SEM suffix too."""
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("09-modern-dove_AP_img_004_SEM.tif", "SEM"),
        make_record("09-modern-dove_AP_img_004_SEM.tif", "FIB"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].record("SEM") is not None
    assert frames[0].record("FIB") is not None


def test_frame_with_only_one_modality():
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("lam_AP_img_000_SEM.tif", "SEM"),
        make_record("lam_AP_img_001_FIB.tif", "FIB"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert [(f.image_number, f.record("SEM") is None, f.record("FIB") is None) for f in frames] == [
        (0, False, True),
        (1, True, False),
    ]


def test_modalities_in_different_states_still_share_a_frame():
    """Curation deselects one modality at a time, so a frame's SEM and FIB
    images are often in different states. Grouping must not split them."""
    stage = make_stage("adaptive_polish_1", "01", [
        ImageRecord(
            image_path=Path("sem/lam_AP_img_000_SEM.tif"),
            modality="SEM", state="deselected",
        ),
        ImageRecord(
            image_path=Path("fib/lam_AP_img_000_FIB.tif"), modality="FIB",
        ),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].record("SEM").state == "deselected"
    assert frames[0].record("FIB").state == "selected"


def test_records_property_lists_present_modalities():
    record = make_record("img_000_SEM.tif", "SEM")
    frame = LamellaFrame(
        lamella="01", stage="s", stage_number="01", image_number=0,
        sem=(record,),
    )

    assert len(frame.records) == 1
    assert frame.record("FIB") is None
    assert frame.record("sem") is record


def test_record_rejects_unknown_modality():
    frame = LamellaFrame(lamella="01", stage="s", stage_number="01", image_number=0)

    with pytest.raises(ValueError):
        frame.record("EDX")


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_frames_ordered_by_image_number_within_a_stage():
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("lam_AP_img_010_SEM.tif", "SEM"),
        make_record("lam_AP_img_002_SEM.tif", "SEM"),
        make_record("lam_AP_img_001_SEM.tif", "SEM"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert [frame.image_number for frame in frames] == [1, 2, 10]


def test_frames_follow_stage_order_across_the_lamella():
    """Stage order comes from the scanner; frames must not re-sort across stages."""
    first = make_stage("adaptive_polish_morning", "01", [
        make_record("lam_AP_img_005_SEM.tif", "SEM"),
    ])
    second = make_stage("adaptive_polish_afternoon", "02", [
        make_record("lam_AP_img_000_SEM.tif", "SEM"),
    ])

    frames = build_lamella_frames(make_lamella([first, second]))

    assert [(f.stage_number, f.image_number) for f in frames] == [("01", 5), ("02", 0)]


def test_unnumbered_images_come_last_ordered_by_name():
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("zzz_reference_SEM.tif", "SEM"),
        make_record("aaa_reference_SEM.tif", "SEM"),
        make_record("lam_AP_img_003_SEM.tif", "SEM"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert [f.image_number for f in frames] == [3, None, None]
    assert [f.record("SEM").image_path.name for f in frames[1:]] == [
        "aaa_reference_SEM.tif",
        "zzz_reference_SEM.tif",
    ]


def test_the_2023_beam_form_pairs_across_its_differing_filenames():
    """``Image_1_000`` and ``Image_2_000`` are one acquisition, two beams.

    The names differ, so the stem fallback would put them in separate frames
    and leave each showing one modality and a blank where the other belongs.
    """
    stage = make_stage("lam_02", "01", [
        make_record("Image_1_000.tif", "SEM"),
        make_record("Image_2_000.tif", "FIB"),
        make_record("Image_1_001.tif", "SEM"),
        make_record("Image_2_001.tif", "FIB"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert [frame.image_number for frame in frames] == [0, 1]
    assert all(frame.record("SEM") and frame.record("FIB") for frame in frames)
    assert frames[0].record("FIB").image_path.name == "Image_2_000.tif"


def test_unnumbered_sem_and_fib_pair_by_filename_stem():
    stage = make_stage("adaptive_polish_1", "01", [
        make_record("centring.tif", "SEM"),
        make_record("centring.tif", "FIB"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].record("SEM") is not None and frames[0].record("FIB") is not None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_plots_matched_to_frames_by_image_number():
    stage = make_stage(
        "adaptive_polish_1", "01",
        [make_record("lam_AP_img_000_SEM.tif", "SEM"),
         make_record("lam_AP_img_001_SEM.tif", "SEM")],
        plots=["lam_AP_img_001_plot.png", "lam_AP_img_000_plot.png"],
    )

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].plot_path.name == "lam_AP_img_000_plot.png"
    assert frames[1].plot_path.name == "lam_AP_img_001_plot.png"


def test_missing_plot_is_normal():
    """Stages routinely hold fewer plots than images."""
    stage = make_stage(
        "adaptive_polish_1", "01",
        [make_record("lam_AP_img_000_SEM.tif", "SEM"),
         make_record("lam_AP_img_001_SEM.tif", "SEM")],
        plots=["lam_AP_img_000_plot.png"],
    )

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].plot_path is not None
    assert frames[1].plot_path is None


def test_plot_without_an_image_is_ignored():
    stage = make_stage(
        "adaptive_polish_1", "01",
        [make_record("lam_AP_img_000_SEM.tif", "SEM")],
        plots=["lam_AP_img_000_plot.png", "lam_AP_img_099_plot.png"],
    )

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def make_variant_record(
    name: str, modality: str, variant: str | None, family: str | None = None
) -> ImageRecord:
    """Build a record in a condition's image directory, ``<family>[_<variant>]``."""
    family_name = family or modality.lower()
    folder = f"{family_name}_{variant}" if variant else family_name
    return ImageRecord(
        image_path=Path(folder) / name, modality=modality, variant=variant, family=family
    )


def test_every_imaging_condition_of_one_acquisition_is_kept():
    """One acquisition captured under several conditions is one frame.

    Earlier this kept only the first record and warned about the rest, which
    lost four of the five SEM images in the legacy multi-condition layout.
    """
    stage = make_stage("adaptive_polish_1", "01", [
        make_variant_record("img_0004.tif", "SEM", "8_fi_1"),
        make_variant_record("img_0004.tif", "SEM", None),
        make_variant_record("img_0004.tif", "SEM", "8_fi_16"),
        make_variant_record("img_0004.tif", "FIB", None),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert len(frames[0].sem) == 3
    assert len(frames[0].records) == 4


def test_the_base_image_comes_first_and_variants_follow_in_name_order():
    stage = make_stage("adaptive_polish_1", "01", [
        make_variant_record("img_0004.tif", "SEM", "fi_16"),
        make_variant_record("img_0004.tif", "SEM", "fi_1"),
        make_variant_record("img_0004.tif", "SEM", None),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].condition_choices("SEM") == [
        ("base", "sem"), ("fi_1", "sem_fi_1"), ("fi_16", "sem_fi_16"),
    ]
    assert frames[0].record("SEM").variant is None


def test_a_condition_is_requested_by_its_directory_in_any_case():
    stage = make_stage("adaptive_polish_1", "01", [
        make_variant_record("img_0004.tif", "SEM", None),
        make_variant_record("img_0004.tif", "SEM", "5kv"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].record("SEM", "SEM_5KV").image_path.parent.name == "sem_5kv"
    assert frames[0].record("SEM", "nonexistent") is None


def test_two_families_of_one_beam_each_offer_their_base():
    """A variant alone cannot tell them apart: both families have a base."""
    stage = make_stage("adaptive_polish_1", "01", [
        make_variant_record("img_0004.tif", "SEM", "fi4", family="sem-5kV"),
        make_variant_record("img_0004.tif", "SEM", None, family="sem-5kV"),
        make_variant_record("img_0004.tif", "SEM", "fi4", family="sem"),
        make_variant_record("img_0004.tif", "SEM", None, family="sem"),
    ])

    frame = build_lamella_frames(make_lamella([stage]))[0]

    assert frame.condition_choices("SEM") == [
        ("sem", "sem"), ("sem [fi4]", "sem_fi4"),
        ("sem-5kV", "sem-5kV"), ("sem-5kV [fi4]", "sem-5kV_fi4"),
    ]
    assert frame.record("SEM", "sem-5kV").family == "sem-5kV"
    assert frame.record("SEM").image_path.parent.name == "sem"


def test_an_image_sitting_in_its_stage_has_no_condition(tmp_path: Path):
    record = ImageRecord(image_path=tmp_path / "ref_SEM.tif", stage_path=tmp_path)

    assert record.condition is None
    assert ImageRecord(image_path=tmp_path / "sem_fi8" / "a.tif").condition == "sem_fi8"


def test_base_records_returns_one_image_per_modality():
    """What a single-image view shows, regardless of how many variants exist."""
    stage = make_stage("adaptive_polish_1", "01", [
        make_variant_record("img_0004.tif", "SEM", None),
        make_variant_record("img_0004.tif", "SEM", "5kv"),
        make_variant_record("img_0004.tif", "FIB", None),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert [record.modality for record in frames[0].base_records] == ["SEM", "FIB"]
    assert all(record.variant is None for record in frames[0].base_records)


def test_a_frame_of_only_variants_still_has_a_primary_image():
    """The legacy layout has no unsuffixed directory: every set is a variant."""
    stage = make_stage("adaptive_polish_1", "01", [
        make_variant_record("img_0004.tif", "SEM", "8_fi_4"),
        make_variant_record("img_0004.tif", "SEM", "8"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].record("SEM").variant == "8"
    assert len(frames[0].base_records) == 1


def test_record_with_unknown_modality_is_skipped(caplog):
    stage = make_stage("adaptive_polish_1", "01", [
        ImageRecord(image_path=Path("other/img_000.tif"), modality=None),
        make_record("lam_AP_img_000_SEM.tif", "SEM"),
    ])

    with caplog.at_level("WARNING"):
        frames = build_lamella_frames(make_lamella([stage]))

    assert len(frames) == 1
    assert frames[0].record("SEM") is not None


def test_lamella_with_no_stages_yields_no_frames():
    assert build_lamella_frames(make_lamella([])) == []


def test_empty_stage_yields_no_frames():
    """A stage folder with no images is normal and must not break the stack."""
    stages = [
        make_stage("adaptive_polish_empty", "01"),
        make_stage("adaptive_polish_2", "02", [make_record("lam_AP_img_000_SEM.tif", "SEM")]),
    ]

    frames = build_lamella_frames(make_lamella(stages))

    assert [f.stage_number for f in frames] == ["02"]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def test_display_name_uses_stage_and_image_number():
    stage = make_stage("adaptive_polish_1", "02", [
        make_record("lam_AP_img_013_SEM.tif", "SEM"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].display_name == "stage 02 - img 013"


def test_display_name_falls_back_to_filename_when_unnumbered():
    stage = make_stage("adaptive_polish_1", "02", [
        make_record("centring.tif", "SEM"),
    ])

    frames = build_lamella_frames(make_lamella([stage]))

    assert frames[0].display_name == "stage 02 - centring"
