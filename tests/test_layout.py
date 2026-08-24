"""Tests for the filesystem naming conventions in :mod:`layout`.

The classification and resolution functions are pure, so they are tested
exhaustively against every directory-name form seen in the real datasets. The
few functions that touch the filesystem are tested against small trees built in
tmp_path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.layout import (
    ImageDirectory,
    LooseImageSets,
    acquisition_stem,
    beam_modality,
    build_image_sets,
    build_loose_image_sets,
    candidate_labels_directory_names,
    classify_image_directory,
    classify_loose_image,
    directory_role,
    image_files_by_name,
    image_files_of,
    is_alignment_directory,
    is_ignored_image_name,
    is_labels_directory,
    labels_directory_name,
    loose_image_set_name,
    resolve_labels_directory,
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

def test_image_files_by_name_does_not_look_in_state_directories(tmp_path: Path):
    """A nested file belongs to an unmigrated dataset. Reading it here would
    report it as selected; inspect-layout says so instead."""
    directory = tmp_path / "sem"
    _make_image(directory, "a.tif")
    _make_image(directory / "deselected", "b.tif")
    _make_image(directory / "corrected", "c.tif")
    (directory / "notes.txt").write_text("ignored")

    found = image_files_by_name(directory)
    assert set(found) == {"a.tif"}


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


def test_is_convertible_prediction_reads_directories_not_the_filename():
    """Only ancestor *directories* disqualify a prediction.

    The check scans path parts rather than building a parent path per level, so
    the filename has to be excluded explicitly -- otherwise a file that happened
    to be called `clean` would be refused on the strength of its own name.
    """
    from am_segmentation_manager.layout import is_convertible_prediction

    assert is_convertible_prediction(Path("/a/predictions/raw/img.tif"))
    assert is_convertible_prediction(Path("/a/predictions/img.tif"))
    assert is_convertible_prediction(Path("/a/predictions/raw/clean"))

    assert not is_convertible_prediction(Path("/a/predictions/clean/img.tif"))
    # Case is not significant, and depth is not either: post-processed output
    # nested any distance below a `clean` directory is still post-processed.
    assert not is_convertible_prediction(Path("/a/predictions/CLEAN/img.tif"))
    assert not is_convertible_prediction(Path("/a/clean/more/img.tif"))


# ---------------------------------------------------------------------------
# Image file name classification
# ---------------------------------------------------------------------------

#: Every loose-image filename form the reference collection holds, transcribed
#: from it. Grouped by the era that writes each form, because a rule that only
#: works on the newest naming would pass a spot check and drop four years of
#: data.
REFERENCE_NAMES = [
    # 2026-07 onwards: the task name is spelled out, spaces and all.
    "ref_Setup Lamella Position_start_eb.tif",
    "ref_Setup Lamella Position_final_res_01_ib.tif",
    "ref_Mill Fiducial_start_ib.tif",
    "ref_Rough Milling_final_res_02_eb.tif",
    "ref_Polishing_start_eb.tif",
    # 2026-07 onwards: taken around a milling task, inside the task directory.
    "Fiducial_pre-task_10-35-26_eb.tif",
    "Polishing_pre-task_12-04-38_ib.tif",
    "Rough-Milling_pre-task_10-40-02_eb.tif",
    "Fiducial-Fiducial_finished_10-38-37_ib.tif",
    "Polishing-Fine-1_finished_11-04-43_eb.tif",
    "Polishing-Localised-Adaptive-Polishing_finished_12-15-12_ib.tif",
    "Rough-Milling-Microexpansion_finished_10-41-52_eb.tif",
    "Rough-Milling-Rough-2_finished_10-54-11_ib.tif",
    # 2025 and 2026-04: the task name runs together, and the resolution is a
    # word rather than a number.
    "ref_MillRough_start_eb.tif",
    "ref_MillRough_final_high_res_ib.tif",
    "ref_SetupLamella_final_low_res_eb.tif",
    "ref_SetupPolishing_final_high_res_ib.tif",
    "ref_MillPolishing_start_eb.tif",
    "ref_milling_Fine-1_finished_1750981637_3822749_eb.tif",
    "ref_milling_Rough-2_finished_1750966808_398733_ib.tif",
    # 2025 and 2026-04, and the one reference image carrying no beam suffix.
    "ref_PositionReady.tif",
]

ALIGNMENT_NAMES = [
    "beam_shift_alignment_1750966320_5954986_ib.tif",
    "ref_alignment_ib.tif",
    "ref_Fiducial_initial_alignment_1750956609_2132776_ib.tif",
    "ref_Fine 1_initial_alignment_1750981636_6501825_ib.tif",
    "ref_Microexpansion_initial_alignment_1775557405_57263_ib.tif",
    "ref_Rough 1_initial_alignment_1750966458_7226627_ib.tif",
    "Polishing_alignment_reference_10-55-27_ib.tif",
    "Rough-Milling_alignment_reference_10-40-00_ib.tif",
]

#: The names an alignment run directory holds. Generic enough to mean anything,
#: so they are only alignment images where the directory vouches for them.
ALIGNMENT_RUN_NAMES = [
    "reference_image.tif",
    "final_image.tif",
    "new_image_00.tif",
    "new_image_02.tif",
]


@pytest.mark.parametrize("name", REFERENCE_NAMES)
def test_reference_images_are_classified_as_reference(name):
    classified = classify_loose_image(name)
    assert classified is not None, name
    assert classified.series == "reference"


@pytest.mark.parametrize("name", ALIGNMENT_NAMES)
def test_alignment_images_are_classified_as_alignment(name):
    classified = classify_loose_image(name)
    assert classified is not None, name
    assert classified.series == "alignment"


@pytest.mark.parametrize("name", ALIGNMENT_NAMES)
def test_alignment_wins_over_the_reference_prefix(name):
    """``ref_`` prefixes alignment images as readily as reference ones.

    The rule order is the whole of what separates them, so it is asserted
    rather than left to the classification tests above to imply.
    """
    assert classify_loose_image(name).series == "alignment"


@pytest.mark.parametrize("name", ALIGNMENT_RUN_NAMES)
def test_alignment_run_names_need_the_directory_to_vouch_for_them(name):
    assert classify_loose_image(name, in_alignment_run=True).series == "alignment"
    assert classify_loose_image(name) is None


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ref_Polishing_start_eb.tif", "SEM"),
        ("ref_Polishing_start_ib.tif", "FIB"),
        ("beam_shift_alignment_1750966320_5954986_ib.tif", "FIB"),
        ("Polishing-Fine-1_finished_11-04-43_eb.tif", "SEM"),
        ("ref_PositionReady.tif", None),
        ("reference_image.tif", None),
        ("new_image_00.tif", None),
        ("ref_Setup Lamella Position_start_EB.tif", "SEM"),
    ],
)
def test_beam_modality_reads_the_suffix(name, expected):
    assert beam_modality(name) == expected


@pytest.mark.parametrize(
    "sem_name, fib_name",
    [
        ("ref_Polishing_start_eb.tif", "ref_Polishing_start_ib.tif"),
        (
            "ref_Setup Lamella Position_final_res_01_eb.tif",
            "ref_Setup Lamella Position_final_res_01_ib.tif",
        ),
        ("ref_MillRough_final_high_res_eb.tif", "ref_MillRough_final_high_res_ib.tif"),
        (
            "ref_milling_Fine-1_finished_1750981637_3822749_eb.tif",
            "ref_milling_Fine-1_finished_1750981637_3822749_ib.tif",
        ),
        (
            "Polishing-Fine-1_finished_11-04-43_eb.tif",
            "Polishing-Fine-1_finished_11-04-43_ib.tif",
        ),
    ],
)
def test_both_beams_of_one_acquisition_share_a_stem(sem_name, fib_name):
    assert acquisition_stem(sem_name) == acquisition_stem(fib_name)


def test_acquisition_stem_leaves_a_name_with_no_beam_suffix_alone():
    assert acquisition_stem("ref_PositionReady.tif") == "ref_PositionReady"
    assert acquisition_stem("new_image_00.tif") == "new_image_00"


def test_acquisition_stems_of_different_acquisitions_differ():
    assert acquisition_stem("ref_Polishing_start_eb.tif") != acquisition_stem(
        "ref_Polishing_final_res_01_eb.tif"
    )


@pytest.mark.parametrize(
    "name",
    [
        "01-ruling-ibex_centring_SEM.tif",
        "08-live-bird_centring_SEM.tif",
    ],
)
def test_centring_images_are_ignored_rather_than_unrecognised(name):
    assert is_ignored_image_name(name)
    assert classify_loose_image(name) is None


@pytest.mark.parametrize(
    "name",
    [
        # Adaptive Polishing images, which fill an image directory instead.
        "01-ruling-ibex_AP_img_013_SEM.tif",
        "img_0013.tif",
        "adapt_mill_img_002_electron_002.tif",
        "Franklin_2024-03-21_lam019_img_0008.tif",
        # Above the lamella level, so no loose-image rule should claim them.
        "overview-image-2026-04-17_11-17-34.tif",
        "tile_0_0_eb.tif",
    ],
)
def test_names_no_rule_recognises_are_reported_rather_than_guessed_at(name):
    assert classify_loose_image(name) is None
    assert not is_ignored_image_name(name)


def test_alignment_directory_is_recognised_whatever_its_case():
    assert is_alignment_directory("Alignment")
    assert is_alignment_directory("alignment")
    assert not is_alignment_directory("Milling")
    assert not is_alignment_directory("01-ruling-ibex - Polishing-11-33-36")


@pytest.mark.parametrize(
    "series, modality, expected",
    [
        ("reference", "SEM", "reference_sem"),
        ("reference", "FIB", "reference_fib"),
        ("alignment", "FIB", "alignment_fib"),
    ],
)
def test_loose_image_set_name_feeds_the_one_label_naming_rule(series, modality, expected):
    assert loose_image_set_name(series, modality) == expected
    assert labels_directory_name(expected) == f"{expected}_labels"


@pytest.mark.parametrize("series", ["reference", "alignment"])
@pytest.mark.parametrize("modality", ["SEM", "FIB"])
def test_a_loose_label_directory_is_never_mistaken_for_an_image_directory(series, modality):
    """The new names must not collide with any role a directory already has."""
    name = labels_directory_name(loose_image_set_name(series, modality))
    assert directory_role(name) == "labels"
    assert classify_image_directory(name) is None


def test_loose_image_set_name_refuses_the_adaptive_series():
    """Adaptive Polishing images fill a directory that already has a name."""
    with pytest.raises(ValueError, match="Not a loose series"):
        loose_image_set_name("adaptive", "SEM")


# ---------------------------------------------------------------------------
# Loose image set assembly
# ---------------------------------------------------------------------------

def make_files(directory: Path, names: list[str]) -> Path:
    """Create *names* as empty files in *directory*, which is created too."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()
    return directory


#: A lamella directory as the 2026-07 layout leaves it: reference images of
#: both beams, one alignment image, and the milling directories below.
LAMELLA_FILES = [
    "ref_Polishing_start_eb.tif",
    "ref_Polishing_start_ib.tif",
    "ref_Rough Milling_final_res_01_eb.tif",
    "ref_Rough Milling_final_res_01_ib.tif",
    "ref_alignment_ib.tif",
]


def test_a_lamella_directory_yields_one_set_per_series_and_beam(tmp_path):
    lamella = make_files(tmp_path / "01-ruling-ibex", LAMELLA_FILES)

    found = build_loose_image_sets(lamella)

    assert [(s.series, s.modality) for s in found.sets] == [
        ("reference", "SEM"),
        ("reference", "FIB"),
        ("alignment", "FIB"),
    ]
    assert found.unrecognised == ()
    assert found.unresolved == ()


def test_each_loose_set_holds_only_its_own_files(tmp_path):
    lamella = make_files(tmp_path / "01-ruling-ibex", LAMELLA_FILES)

    by_key = {(s.series, s.modality): s for s in build_loose_image_sets(lamella).sets}

    assert by_key[("reference", "SEM")].image_names == (
        "ref_Polishing_start_eb.tif",
        "ref_Rough Milling_final_res_01_eb.tif",
    )
    assert by_key[("alignment", "FIB")].image_names == ("ref_alignment_ib.tif",)


def test_a_directory_with_no_loose_images_yields_nothing(tmp_path):
    stage = tmp_path / "adaptive_polish_2026-07-28-11-34-24AM"
    make_files(stage / "sem", ["01-ruling-ibex_AP_img_000_SEM.tif"])

    assert build_loose_image_sets(stage) == LooseImageSets()


def test_adaptive_image_directories_are_left_to_the_other_builder(tmp_path):
    """The two builders must not both claim the same files."""
    stage = tmp_path / "adaptive_polish_2026-07-28-11-34-24AM"
    make_files(stage / "sem", ["01-ruling-ibex_AP_img_000_SEM.tif"])
    make_files(stage / "fib", ["01-ruling-ibex_AP_img_000_FIB.tif"])

    assert build_loose_image_sets(stage).sets == ()
    assert [s.modality for s in build_image_sets(stage)] == ["SEM", "FIB"]


def test_an_adaptive_set_is_the_whole_directory_and_says_so(tmp_path):
    stage = tmp_path / "adaptive_polish_2026-07-28-11-34-24AM"
    make_files(stage / "sem", ["a_SEM.tif", "b_SEM.tif"])

    image_set = build_image_sets(stage)[0]

    assert image_set.series == "adaptive"
    assert image_set.image_names is None
    assert not image_set.is_loose
    assert image_set.name == "sem"
    assert image_set.display_name == "SEM"


def test_a_loose_set_is_named_for_its_series_rather_than_its_directory(tmp_path):
    lamella = make_files(tmp_path / "01-ruling-ibex", LAMELLA_FILES)

    image_set = build_loose_image_sets(lamella).sets[0]

    assert image_set.is_loose
    assert image_set.name == "reference_sem"
    assert image_set.display_name == "SEM (reference)"
    assert image_set.variant is None


def test_centring_images_leave_no_trace(tmp_path):
    """Deliberately omitted, so neither a set nor something to report."""
    stage = make_files(
        tmp_path / "adaptive_polish_2026-07-28-11-34-24AM",
        ["01-ruling-ibex_centring_SEM.tif"],
    )

    assert build_loose_image_sets(stage) == LooseImageSets()


def test_a_name_no_rule_recognises_is_reported(tmp_path):
    lamella = make_files(
        tmp_path / "01-ruling-ibex", [*LAMELLA_FILES, "something_new_eb.tif"]
    )

    found = build_loose_image_sets(lamella)

    assert found.unrecognised == ("something_new_eb.tif",)
    assert len(found.sets) == 3


# ---------------------------------------------------------------------------
# Resolving the beam a filename does not state
# ---------------------------------------------------------------------------

ALIGNMENT_RUN_FILES = ["reference_image.tif", "new_image_00.tif", "final_image.tif"]


def test_an_alignment_run_needs_the_directory_to_vouch_for_its_files(tmp_path):
    run = make_files(tmp_path / "Alignment" / "01-ruling-ibex - Polishing", ALIGNMENT_RUN_FILES)

    assert build_loose_image_sets(run).unrecognised == tuple(sorted(ALIGNMENT_RUN_FILES))
    assert build_loose_image_sets(run, in_alignment_run=True).unrecognised == ()


def test_an_unstated_beam_is_reported_when_nothing_can_resolve_it(tmp_path):
    run = make_files(tmp_path / "Alignment" / "01-ruling-ibex - Polishing", ALIGNMENT_RUN_FILES)

    found = build_loose_image_sets(run, in_alignment_run=True)

    assert found.sets == ()
    assert found.unresolved == tuple(sorted(ALIGNMENT_RUN_FILES))


def test_a_resolver_supplies_the_beam_the_name_does_not(tmp_path):
    run = make_files(tmp_path / "Alignment" / "01-ruling-ibex - Polishing", ALIGNMENT_RUN_FILES)

    found = build_loose_image_sets(
        run, in_alignment_run=True, resolve_modality=lambda path: "FIB"
    )

    assert [(s.series, s.modality) for s in found.sets] == [("alignment", "FIB")]
    assert found.sets[0].image_names == tuple(sorted(ALIGNMENT_RUN_FILES))
    assert found.unresolved == ()


def test_the_resolver_is_asked_only_about_names_that_do_not_state_a_beam(tmp_path):
    """Reading a header is the exception, so it must not become the rule."""
    lamella = make_files(tmp_path / "01-ruling-ibex", [*LAMELLA_FILES, "ref_PositionReady.tif"])
    asked: list[str] = []

    build_loose_image_sets(
        lamella, resolve_modality=lambda path: asked.append(path.name) or "FIB"
    )

    assert asked == ["ref_PositionReady.tif"]


def test_a_resolver_that_cannot_say_leaves_the_image_reported(tmp_path):
    lamella = make_files(tmp_path / "01-ruling-ibex", ["ref_PositionReady.tif"])

    found = build_loose_image_sets(lamella, resolve_modality=lambda path: None)

    assert found.sets == ()
    assert found.unresolved == ("ref_PositionReady.tif",)


# ---------------------------------------------------------------------------
# Where a loose set's labels go
# ---------------------------------------------------------------------------

def test_a_loose_label_directory_is_named_for_the_set(tmp_path):
    lamella = make_files(tmp_path / "01-ruling-ibex", LAMELLA_FILES)

    by_key = {(s.series, s.modality): s for s in build_loose_image_sets(lamella).sets}

    assert by_key[("reference", "SEM")].labels_directory == lamella / "reference_sem_labels"
    assert by_key[("alignment", "FIB")].labels_directory == lamella / "alignment_fib_labels"
    assert not by_key[("reference", "SEM")].labels_exist


def test_an_existing_loose_label_directory_is_found(tmp_path):
    lamella = make_files(tmp_path / "01-ruling-ibex", LAMELLA_FILES)
    (lamella / "reference_sem_labels").mkdir()

    by_key = {(s.series, s.modality): s for s in build_loose_image_sets(lamella).sets}

    assert by_key[("reference", "SEM")].labels_exist
    assert not by_key[("reference", "FIB")].labels_exist


def test_a_loose_set_never_claims_the_adaptive_labels_beside_it(tmp_path):
    """The failure this guards against pairs images with labels of other scenes.

    A dataset that collapses the stage level keeps its Adaptive Polishing
    images and its ``labels/`` in the lamella directory. A reference set in
    that same directory must not fall back to them.
    """
    lamella = make_files(tmp_path / "lam_003", LAMELLA_FILES)
    make_files(lamella / "SEM_8", ["img_0000.tif"])
    make_files(lamella / "labels", ["img_0000.tif"])
    make_files(lamella / "sem_labels", ["img_0000.tif"])

    by_key = {(s.series, s.modality): s for s in build_loose_image_sets(lamella).sets}

    assert by_key[("reference", "SEM")].labels_directory == lamella / "reference_sem_labels"
    assert not by_key[("reference", "SEM")].labels_exist


def test_loose_sets_in_one_directory_never_share_a_label_directory(tmp_path):
    lamella = make_files(tmp_path / "01-ruling-ibex", LAMELLA_FILES)

    sets = build_loose_image_sets(lamella).sets

    assert len({s.labels_directory for s in sets}) == len(sets)
    assert not any(s.labels_are_shared for s in sets)
    assert not any(s.labels_are_legacy for s in sets)


# ---------------------------------------------------------------------------
# Reading a set's files back
# ---------------------------------------------------------------------------

def test_image_files_of_a_loose_set_returns_only_that_set(tmp_path):
    lamella = make_files(tmp_path / "01-ruling-ibex", LAMELLA_FILES)

    by_key = {(s.series, s.modality): s for s in build_loose_image_sets(lamella).sets}

    assert sorted(image_files_of(by_key[("reference", "FIB")])) == [
        "ref_Polishing_start_ib.tif",
        "ref_Rough Milling_final_res_01_ib.tif",
    ]


def test_image_files_of_an_adaptive_set_returns_the_whole_directory(tmp_path):
    stage = tmp_path / "adaptive_polish_2026-07-28-11-34-24AM"
    make_files(stage / "sem", ["a_SEM.tif", "b_SEM.tif"])

    found = image_files_of(build_image_sets(stage)[0])

    assert sorted(found) == ["a_SEM.tif", "b_SEM.tif"]
    assert found["a_SEM.tif"] == stage / "sem" / "a_SEM.tif"
