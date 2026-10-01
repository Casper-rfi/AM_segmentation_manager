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
    NO_CORRECTIONS,
    UNNAMED_STAGE,
    FamilyCorrections,
    ImageDirectory,
    ImageSet,
    LooseImageSets,
    acquisition_stem,
    beam_modality,
    build_image_sets,
    build_loose_image_sets,
    classify_image_directory,
    classify_loose_image,
    condition_of,
    conflicting_label_directories,
    directory_cache,
    directory_role,
    families_claiming_one_directory,
    image_files,
    image_files_by_name,
    image_files_of,
    is_alignment_directory,
    is_ignored_image_name,
    is_labels_directory,
    labels_directory_name,
    loose_image_set_name,
    resolve_labels_directory,
    stage_base_name,
    subdirectories_by_name,
)

# ---------------------------------------------------------------------------
# Image directory classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("sem", ImageDirectory("SEM", None, "sem")),
        ("SEM", ImageDirectory("SEM", None, "SEM")),
        ("fib", ImageDirectory("FIB", None, "fib")),
        ("FIB", ImageDirectory("FIB", None, "FIB")),
        ("SEM_8", ImageDirectory("SEM", "8", "SEM")),
        ("SEM_8_fi_1", ImageDirectory("SEM", "8_fi_1", "SEM")),
        ("SEM_8_fi_16", ImageDirectory("SEM", "8_fi_16", "SEM")),
        ("sem_5kv", ImageDirectory("SEM", "5kv", "sem")),
        ("sem_defocus", ImageDirectory("SEM", "defocus", "sem")),
        ("fib_lowmag", ImageDirectory("FIB", "lowmag", "fib")),
        ("sem5kV", ImageDirectory("SEM", None, "sem5kV")),
        ("sem-5kV", ImageDirectory("SEM", None, "sem-5kV")),
        ("SEM-5kV-mag-12K", ImageDirectory("SEM", None, "SEM-5kV-mag-12K")),
        ("SEM-5kV-mag-12K_fi1", ImageDirectory("SEM", "fi1", "SEM-5kV-mag-12K")),
        ("SEM-5kV-mag-12K_fi1-TLD", ImageDirectory("SEM", "fi1-TLD", "SEM-5kV-mag-12K")),
        ("sem-2.5kV_fi_4", ImageDirectory("SEM", "fi_4", "sem-2.5kV")),
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
    """The variant is reported as written on disk."""
    assert classify_image_directory("SEM_LowMag") == ImageDirectory("SEM", "LowMag", "SEM")


@pytest.mark.parametrize("name", ["fibsem_x", "semi-thin", "semLowMag", "FIBre_4"])
def test_a_modality_followed_by_a_letter_is_not_an_image_directory(name):
    """``sem-LowMag`` names that family; ``semLowMag`` would claim ``fibsem`` too."""
    assert classify_image_directory(name) is None


@pytest.mark.parametrize("name", ["sem5kV", "sem-LowMag", "sem.2", "fib_2"])
def test_a_modality_followed_by_a_digit_or_separator_is_an_image_directory(name):
    assert classify_image_directory(name) is not None


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


def _stage(tmp_path: Path, *names: str) -> Path:
    """Return a stage holding the directories *names*."""
    stage = tmp_path / "stage"
    for name in names:
        (stage / name).mkdir(parents=True)
    return stage


def _labels(stage: Path, name: str) -> Path:
    return resolve_labels_directory(
        stage / name, "SEM", subdirectories_by_name(stage), corrections=NO_CORRECTIONS
    )[0]


def test_a_family_shares_the_directory_named_after_it(tmp_path: Path):
    stage = _stage(tmp_path, "sem-5kV", "sem-5kV_fi1", "sem-5kV_labels")

    assert {_labels(stage, "sem-5kV"), _labels(stage, "sem-5kV_fi1")} == {
        stage / "sem-5kV_labels"
    }


def test_the_family_named_after_its_modality_shares_the_modality_directory(tmp_path: Path):
    stage = _stage(tmp_path, "SEM_8", "SEM_8_fi_1", "sem_labels")

    assert {_labels(stage, "SEM_8"), _labels(stage, "SEM_8_fi_1")} == {stage / "sem_labels"}


def test_two_families_resolve_apart(tmp_path: Path):
    stage = _stage(tmp_path, "sem-2kV", "sem-2kV_fi4", "sem-5kV", "sem-5kV_fi4", "sem-2kV_labels")

    assert _labels(stage, "sem-2kV_fi4") == stage / "sem-2kV_labels"
    assert _labels(stage, "sem-5kV_fi4") == stage / "sem-5kV_labels"


def test_missing_labels_resolve_to_the_family_directory(tmp_path: Path):
    """With nothing on disk, the answer is where a first label would be written."""
    stage = _stage(tmp_path, "SEM_8", "SEM_8_fi_1", "sem-5kV_fi4")

    directory, exists, legacy = resolve_labels_directory(
        stage / "SEM_8_fi_1", "SEM", subdirectories_by_name(stage), corrections=NO_CORRECTIONS
    )
    assert (directory, exists, legacy) == (stage / "sem_labels", False, False)
    assert _labels(stage, "sem-5kV_fi4") == stage / "sem-5kV_labels"


def test_resolution_does_not_change_once_the_first_label_is_written(tmp_path: Path):
    """The answer used to be each condition's private directory, splitting the family."""
    stage = _stage(tmp_path, "sem", "sem_fi1", "sem_fi4")
    before = {name: _labels(stage, name) for name in ("sem", "sem_fi1", "sem_fi4")}

    before["sem_fi4"].mkdir()

    assert {name: _labels(stage, name) for name in before} == before
    assert set(before.values()) == {stage / "sem_labels"}


def test_one_private_directory_stands_in_for_the_family(tmp_path: Path):
    """How a stage labelled before families keeps its conditions together."""
    stage = _stage(tmp_path, "SEM_8", "SEM_8_fi_1", "SEM_8_fi_16", "SEM_8_labels")

    for name in ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_16"):
        assert _labels(stage, name) == stage / "SEM_8_labels"
    assert conflicting_label_directories(
        stage / "SEM_8", "SEM", subdirectories_by_name(stage), corrections=NO_CORRECTIONS
    ) == ()


def test_several_directories_for_one_family_are_a_conflict(tmp_path: Path):
    """Each condition keeps its own until they are merged; one without reads the family's."""
    stage = _stage(
        tmp_path, "sem", "sem_5kv", "sem_fi4", "sem_labels", "sem_5kv_labels"
    )

    assert _labels(stage, "sem") == stage / "sem_labels"
    assert _labels(stage, "sem_5kv") == stage / "sem_5kv_labels"
    assert _labels(stage, "sem_fi4") == stage / "sem_labels"
    assert set(
        conflicting_label_directories(
            stage / "sem_fi4", "SEM", subdirectories_by_name(stage), corrections=NO_CORRECTIONS
        )
    ) == {stage / "sem_labels", stage / "sem_5kv_labels"}


def test_a_family_is_one_whatever_the_case_it_is_spelled_in(tmp_path: Path):
    stage = _stage(tmp_path, "sem-5kV", "SEM-5KV_fi1")

    assert _labels(stage, "SEM-5KV_fi1") == stage / "sem-5kV_labels"
    assert _labels(stage, "sem-5kV") == stage / "sem-5kV_labels"


def test_a_named_family_never_reads_the_modality_legacy_directory(tmp_path: Path):
    stage = _stage(tmp_path, "sem", "sem-5kV", "labels")

    assert _labels(stage, "sem") == stage / "labels"
    assert _labels(stage, "sem-5kV") == stage / "sem-5kV_labels"


def test_the_family_directory_wins_over_a_legacy_one(tmp_path: Path):
    stage = _stage(tmp_path, "sem", "sem_labels", "labels")

    directory, _, legacy = resolve_labels_directory(
        stage / "sem", "SEM", subdirectories_by_name(stage), corrections=NO_CORRECTIONS
    )
    assert (directory, legacy) == (stage / "sem_labels", False)


def test_legacy_label_directory_is_matched_case_insensitively(tmp_path: Path):
    """``labels_FIB`` must resolve on case-sensitive filesystems too."""
    stage = tmp_path / "stage"
    (stage / "FIB").mkdir(parents=True)
    (stage / "labels_FIB").mkdir()

    directory, exists, legacy = resolve_labels_directory(
        stage / "FIB", "FIB", subdirectories_by_name(stage), corrections=NO_CORRECTIONS
    )
    assert directory == stage / "labels_FIB"
    assert exists is True
    assert legacy is True


# ---------------------------------------------------------------------------
# Family corrections
# ---------------------------------------------------------------------------

MAG2_OWN_FAMILY = FamilyCorrections(families={"sem_mag2": "sem-mag2"})


def test_without_corrections_a_condition_is_what_its_name_gives():
    assert condition_of("SEM_8_fi_1", NO_CORRECTIONS) == classify_image_directory("SEM_8_fi_1")
    assert condition_of("sem_labels", NO_CORRECTIONS) is None


def test_a_condition_moved_to_a_family_of_its_own_keeps_its_suffix_as_variant():
    assert condition_of("sem_mag2", MAG2_OWN_FAMILY) == ImageDirectory(
        modality="SEM", variant="mag2", family="sem-mag2"
    )


def test_a_base_condition_moved_into_another_family_is_a_variant_there():
    corrections = FamilyCorrections(families={"sem-2kV": "sem"})

    assert condition_of("SEM-2KV", corrections) == ImageDirectory(
        modality="SEM", variant="SEM-2KV", family="sem"
    )


def test_a_condition_is_base_when_its_name_is_its_family():
    corrections = FamilyCorrections(families={"sem2kvb": "SEM2kVb"})

    assert condition_of("sem2kVb", corrections) == ImageDirectory(
        modality="SEM", variant=None, family="SEM2kVb"
    )


@pytest.mark.parametrize(
    "families, primary, problem",
    [
        ({"sem_x": "fib-a"}, {}, "cannot join the FIB family"),
        ({"sem_x": "sem_y"}, {}, "is not a family name"),
        ({"sem_x": "semi-thin"}, {}, "is not a family name"),
        ({"labels": "sem"}, {}, "is not the name of an image directory"),
        ({"sem_a": "sem-b", "SEM_A": "sem-c"}, {}, "listed twice"),
        ({"sem_mag2": "sem-mag2"}, {"sem": "sem_mag2"}, "belongs to family sem-mag2"),
        ({}, {"sem": "fib_2"}, "belongs to family fib"),
        ({}, {"sem": "sem_8", "SEM": "sem_4"}, "listed twice"),
    ],
)
def test_corrections_refuse_what_resolution_could_misread(families, primary, problem):
    with pytest.raises(ValueError, match=problem):
        FamilyCorrections(families=families, primary=primary)


def test_a_primary_may_name_a_condition_corrected_into_its_family():
    corrections = FamilyCorrections(
        families={"sem_mag2": "sem-mag2"}, primary={"SEM-MAG2": "sem_mag2"}
    )

    assert corrections.primary_of("sem-mag2") == "sem_mag2"
    assert corrections.family_of("SEM_MAG2") == "sem-mag2"


def _corrected_labels(stage: Path, name: str, corrections: FamilyCorrections) -> Path:
    return resolve_labels_directory(
        stage / name, "SEM", subdirectories_by_name(stage), corrections=corrections
    )[0]


def test_a_condition_moved_out_of_a_family_leaves_its_label(tmp_path: Path):
    stage = _stage(tmp_path, "sem", "sem_fi8", "sem_mag2", "sem_labels")

    assert _corrected_labels(stage, "sem_mag2", MAG2_OWN_FAMILY) == stage / "sem-mag2_labels"
    assert _corrected_labels(stage, "sem_fi8", MAG2_OWN_FAMILY) == stage / "sem_labels"
    assert _corrected_labels(stage, "sem", MAG2_OWN_FAMILY) == stage / "sem_labels"


def test_a_condition_moved_into_a_family_shares_its_label(tmp_path: Path):
    stage = _stage(tmp_path, "sem-2kV", "sem2kVb", "sem-2kV_labels")
    corrections = FamilyCorrections(families={"sem2kVb": "sem-2kV"})

    assert _labels(stage, "sem2kVb") == stage / "sem2kVb_labels"
    assert _corrected_labels(stage, "sem2kVb", corrections) == stage / "sem-2kV_labels"


def test_a_private_directory_follows_its_condition_into_its_new_family(tmp_path: Path):
    """Where it stood in for the old family, that family is left without it."""
    stage = _stage(tmp_path, "sem", "sem_fi8", "sem_mag2", "sem_mag2_labels")

    assert _labels(stage, "sem") == stage / "sem_mag2_labels"
    assert _corrected_labels(stage, "sem_mag2", MAG2_OWN_FAMILY) == stage / "sem_mag2_labels"
    assert _corrected_labels(stage, "sem", MAG2_OWN_FAMILY) == stage / "sem_labels"


def test_corrections_decide_which_directories_are_in_conflict(tmp_path: Path):
    stage = _stage(tmp_path, "sem", "sem_mag2", "sem_labels", "sem_mag2_labels")
    siblings = subdirectories_by_name(stage)

    assert conflicting_label_directories(
        stage / "sem", "SEM", siblings, corrections=NO_CORRECTIONS
    )
    assert conflicting_label_directories(
        stage / "sem", "SEM", siblings, corrections=MAG2_OWN_FAMILY
    ) == ()


def test_two_families_never_read_one_label_directory():
    """A base moved out keeps its own name, which is still its old family's."""
    stage = Path("stage")
    directories = [stage / name for name in ("sem", "sem_fi1", "sem-2kV_fi1")]

    assert families_claiming_one_directory(directories, corrections=NO_CORRECTIONS) == ()
    assert families_claiming_one_directory(
        directories, corrections=FamilyCorrections(families={"sem": "sem-2kV"})
    ) == (("sem", ("sem", "sem-2kV")),)


def test_a_variant_moved_out_claims_no_other_familys_directory():
    directories = [Path("stage") / name for name in ("sem", "sem_fi8", "sem_mag2")]

    assert families_claiming_one_directory(directories, corrections=MAG2_OWN_FAMILY) == ()


def test_a_base_moved_out_alone_claims_nothing_twice():
    """With no condition left in its old family, nobody else reads its name."""
    directories = [Path("stage") / name for name in ("sem", "sem-2kV_fi1")]
    corrections = FamilyCorrections(families={"sem": "sem-2kV"})

    assert families_claiming_one_directory(directories, corrections=corrections) == ()


def test_image_sets_carry_the_corrected_family(tmp_path: Path):
    stage = _stage(tmp_path, "sem", "sem_fi8", "sem_mag2", "sem_labels")

    image_sets = build_image_sets(stage, corrections=MAG2_OWN_FAMILY)

    assert [(image_set.name, image_set.family, image_set.variant) for image_set in image_sets] == [
        ("sem", "sem", None),
        ("sem_fi8", "sem", "fi8"),
        ("sem_mag2", "sem-mag2", "mag2"),
    ]
    assert image_sets[2].labels_directory == stage / "sem-mag2_labels"
    assert image_sets[2].labels_are_shared is False
    assert image_sets[2].display_name == "sem-mag2 [mag2]"


# ---------------------------------------------------------------------------
# Image sets
# ---------------------------------------------------------------------------

def _image_sets(stage: Path) -> list[ImageSet]:
    return build_image_sets(stage, corrections=NO_CORRECTIONS)


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

    image_sets = _image_sets(stage)
    by_name = {image_set.name: image_set for image_set in image_sets}

    assert set(by_name) == {"SEM_8", "SEM_8_fi_1", "SEM_8_fi_4", "SEM_8_fi_16", "FIB"}
    for name in ("SEM_8", "SEM_8_fi_1", "SEM_8_fi_4", "SEM_8_fi_16"):
        assert by_name[name].labels_directory == stage / "labels"
        assert by_name[name].labels_are_shared is True
        assert by_name[name].labels_are_legacy is True
    assert by_name["FIB"].labels_directory == stage / "labels_FIB"
    assert by_name["FIB"].labels_are_shared is False


def test_build_image_sets_reports_conditions_in_conflict_as_unshared(tmp_path: Path):
    stage = tmp_path / "stage"
    _make_image(stage / "sem", "img_000.tif")
    _make_image(stage / "sem_5kv", "img_000.tif")
    _make_image(stage / "sem_labels", "img_000.tif")
    _make_image(stage / "sem_5kv_labels", "img_000.tif")

    by_name = {image_set.name: image_set for image_set in _image_sets(stage)}
    assert by_name["sem"].labels_are_shared is False
    assert by_name["sem_5kv"].labels_are_shared is False
    assert by_name["sem_5kv"].labels_directory == stage / "sem_5kv_labels"


def test_build_image_sets_orders_sem_before_fib_and_base_before_variants(tmp_path: Path):
    stage = tmp_path / "stage"
    for name in ("fib", "sem_5kv", "sem", "fib_lowmag"):
        _make_image(stage / name, "img_000.tif")

    assert [image_set.name for image_set in _image_sets(stage)] == [
        "sem", "sem_5kv", "fib", "fib_lowmag",
    ]


def test_build_image_sets_orders_by_family_then_base_first(tmp_path: Path):
    stage = tmp_path / "stage"
    for name in ("sem-5kV_fi4", "sem_fi4", "sem-5kV", "sem"):
        _make_image(stage / name, "img_000.tif")

    image_sets = _image_sets(stage)

    assert [image_set.name for image_set in image_sets] == [
        "sem", "sem_fi4", "sem-5kV", "sem-5kV_fi4",
    ]
    assert [image_set.family for image_set in image_sets] == ["sem", "sem", "sem-5kV", "sem-5kV"]
    assert image_sets[3].display_name == "sem-5kV [fi4]"


def test_build_image_sets_ignores_label_and_plot_directories(tmp_path: Path):
    stage = tmp_path / "stage"
    _make_image(stage / "sem", "img_000.tif")
    (stage / "plots").mkdir()
    (stage / "predictions" / "clean").mkdir(parents=True)
    _make_image(stage / "sem_labels", "img_000.tif")

    assert [image_set.name for image_set in _image_sets(stage)] == ["sem"]


def test_build_image_sets_on_a_stage_with_no_image_directories(tmp_path: Path):
    stage = tmp_path / "stage"
    (stage / "plots").mkdir(parents=True)
    assert _image_sets(stage) == []


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


def test_a_file_where_a_directory_was_expected_lists_as_empty(tmp_path: Path):
    """The other way a listing fails, and the one os.scandir raises for."""
    not_a_directory = tmp_path / "file.tif"
    not_a_directory.write_bytes(b"")

    assert subdirectories_by_name(not_a_directory) == {}
    assert image_files(not_a_directory) == []


# ---------------------------------------------------------------------------
# Holding the listings for one traversal
# ---------------------------------------------------------------------------

def _reads(monkeypatch) -> list[str]:
    """Return a list every directory read is appended to."""
    import os

    seen: list[str] = []
    original = os.scandir

    def watched(path):
        seen.append(str(path))
        return original(path)

    monkeypatch.setattr("am_segmentation_manager.layout.os.scandir", watched)
    return seen


def test_a_directory_is_read_once_inside_the_cache(tmp_path: Path, monkeypatch):
    """The saving: a stage is listed four times over on the way to being
    indexed, and each listing is a read of the whole directory."""
    (tmp_path / "sem").mkdir()
    _make_image(tmp_path, "a.tif")
    reads = _reads(monkeypatch)

    with directory_cache():
        for _ in range(4):
            subdirectories_by_name(tmp_path)
            image_files(tmp_path)

    assert reads.count(str(tmp_path)) == 1


def test_the_cache_reports_what_the_directory_holds(tmp_path: Path):
    """A held listing must answer exactly as an unheld one does."""
    (tmp_path / "sem").mkdir()
    (tmp_path / ".hidden").mkdir()
    _make_image(tmp_path, "a.tif")

    with directory_cache():
        cached = subdirectories_by_name(tmp_path), image_files(tmp_path)

    assert cached == (subdirectories_by_name(tmp_path), image_files(tmp_path))


def test_nothing_is_held_outside_the_cache(tmp_path: Path, monkeypatch):
    """Held any longer than one traversal, a listing could go stale."""
    (tmp_path / "sem").mkdir()

    with directory_cache():
        subdirectories_by_name(tmp_path)
    reads = _reads(monkeypatch)
    subdirectories_by_name(tmp_path)

    assert reads == [str(tmp_path)]


def test_a_directory_created_after_the_block_is_seen(tmp_path: Path):
    """The block is what bounds the staleness, so it has to really end."""
    with directory_cache():
        assert subdirectories_by_name(tmp_path) == {}

    (tmp_path / "sem").mkdir()
    assert set(subdirectories_by_name(tmp_path)) == {"sem"}


def test_nesting_reuses_the_outer_block(tmp_path: Path, monkeypatch):
    """A scan enters the cache and calls a search that enters it again; the
    inner block must not throw away what the outer one holds."""
    (tmp_path / "sem").mkdir()
    reads = _reads(monkeypatch)

    with directory_cache():
        subdirectories_by_name(tmp_path)
        with directory_cache():
            subdirectories_by_name(tmp_path)
        subdirectories_by_name(tmp_path)

    assert reads.count(str(tmp_path)) == 1


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
    # What a milling task aligned against, three to a task before it starts.
    "alignment_target_MillPolishingCut_00_ib.tif",
    "alignment_target_MillRoughCut_02_ib.tif",
]

#: The Adaptive Polishing images the script-based prototype wrote loose in the
#: lamella directory, paired with the modality each spells out.
LOOSE_ADAPTIVE_NAMES = [
    # Runs up to 2024-06 repeated the acquisition number after the beam.
    ("adapt_mill_img_000_electron_000.tif", "SEM"),
    ("adapt_mill_img_005_ion_005.tif", "FIB"),
    ("adapt_mill_img_017_ELECTRON_017.tif", "SEM"),
    # Later ones stopped, and mean exactly the same thing.
    ("adapt_mill_img_000_electron.tif", "SEM"),
    ("adapt_mill_img_012_ion.tif", "FIB"),
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


@pytest.mark.parametrize("name, modality", LOOSE_ADAPTIVE_NAMES)
def test_loose_adaptive_images_state_their_own_beam(name, modality):
    """The one loose name spelling its beam out rather than suffixing it."""
    classified = classify_loose_image(name)
    assert classified is not None, name
    assert classified.series == "adaptive"
    assert classified.modality == modality


@pytest.mark.parametrize(
    "name",
    [
        "adapt_mill_img_000_neutron_000.tif",
        "adapt_mill_img_000_electron_extra.tif",
        "prefixed_adapt_mill_img_000_electron_000.tif",
    ],
)
def test_a_name_near_the_loose_adaptive_form_is_not_guessed_at(name):
    assert classify_loose_image(name) is None


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


@pytest.mark.parametrize(
    "name",
    [
        "ref_Alignment Check_start_eb.tif",
        "ref_Alignment Check_final_res_01_ib.tif",
        "ref_Alignment_start_eb.tif",
    ],
)
def test_a_task_named_after_alignment_is_still_a_reference_image(name):
    """Everything after ``ref_`` is a task name somebody chose.

    ``ref_alignment`` is matched entire for this reason: as a prefix it moved a
    whole task's reference images into the alignment series, where nothing
    predicts or labels them and no count would show them missing.
    """
    assert classify_loose_image(name).series == "reference"


@pytest.mark.parametrize(
    "name",
    [
        "ref_Centring_start_eb.tif",
        "Centring_pre-task_10-35-26_eb.tif",
        "Polishing-Centring_finished_10-00-00_ib.tif",
    ],
)
def test_a_task_named_centring_is_indexed_rather_than_dropped(name):
    """The centring image is ``<lamella>_centring_<modality>`` and nothing else.

    Matched anywhere in a name it swallowed a whole task -- and an image
    dropped as deliberately ignored is not even reported as unrecognised,
    which is the worst way for a naming rule to be wrong.
    """
    assert not is_ignored_image_name(name)
    assert classify_loose_image(name).series == "reference"


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
        # Adaptive Polishing images that fill an image directory instead, so
        # their names carry nothing saying what they are. The prototype's loose
        # ones do, and are classified above.
        "01-ruling-ibex_AP_img_013_SEM.tif",
        "img_0013.tif",
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
        ("adaptive", "SEM", "adaptive_sem"),
        ("adaptive", "FIB", "adaptive_fib"),
    ],
)
def test_loose_image_set_name_feeds_the_one_label_naming_rule(series, modality, expected):
    assert loose_image_set_name(series, modality) == expected
    assert labels_directory_name(expected) == f"{expected}_labels"


@pytest.mark.parametrize("series", ["adaptive", "reference", "alignment"])
@pytest.mark.parametrize("modality", ["SEM", "FIB"])
def test_a_loose_label_directory_is_never_mistaken_for_an_image_directory(series, modality):
    """The new names must not collide with any role a directory already has."""
    name = labels_directory_name(loose_image_set_name(series, modality))
    assert directory_role(name) == "labels"
    assert classify_image_directory(name) is None


def test_loose_image_set_name_refuses_something_that_is_not_a_series():
    """Every series can be loose, so only a non-series is left to refuse."""
    with pytest.raises(ValueError, match="Not a loose series"):
        loose_image_set_name("polishing", "SEM")


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
    assert [s.modality for s in _image_sets(stage)] == ["SEM", "FIB"]


def test_an_adaptive_set_is_the_whole_directory_and_says_so(tmp_path):
    stage = tmp_path / "adaptive_polish_2026-07-28-11-34-24AM"
    make_files(stage / "sem", ["a_SEM.tif", "b_SEM.tif"])

    image_set = _image_sets(stage)[0]

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


def test_a_name_no_rule_recognises_is_reported_and_indexed_as_other(tmp_path):
    """Both: the file becomes something a reviewer can open, and the finding
    that makes an unanticipated layout visible keeps its data."""
    lamella = make_files(
        tmp_path / "01-ruling-ibex", [*LAMELLA_FILES, "something_new_eb.tif"]
    )

    found = build_loose_image_sets(lamella)

    assert found.unrecognised == ("something_new_eb.tif",)
    other = [image_set for image_set in found.sets if image_set.series == "other"]
    assert [(one.modality, one.image_names) for one in other] == [
        ("SEM", ("something_new_eb.tif",))
    ]
    assert other[0].labels_directory is None


def test_an_other_set_takes_its_beam_from_the_header_when_the_name_is_silent(tmp_path):
    lamella = make_files(tmp_path / "01-ruling-ibex", ["something_new.tif"])

    found = build_loose_image_sets(lamella, resolve_modality=lambda path: "FIB")

    assert [(one.series, one.modality) for one in found.sets] == [("other", "FIB")]


def test_an_other_set_indexes_an_image_no_beam_can_be_established_for(tmp_path):
    """The population that would otherwise stay unindexed: a name no rule
    recognises whose header states no beam either."""
    lamella = make_files(tmp_path / "01-ruling-ibex", ["something_new.tif"])

    found = build_loose_image_sets(lamella, resolve_modality=lambda path: None)

    assert [(one.series, one.modality) for one in found.sets] == [("other", None)]
    assert found.unresolved == ()


def test_other_sets_are_split_by_beam_with_the_beamless_one_last(tmp_path):
    lamella = make_files(
        tmp_path / "01-ruling-ibex", ["new_a_eb.tif", "new_b_ib.tif", "new_c.tif"]
    )

    found = build_loose_image_sets(lamella, resolve_modality=lambda path: None)

    assert [one.modality for one in found.sets] == ["SEM", "FIB", None]
    assert [one.name for one in found.sets] == ["other_sem", "other_fib", "other"]


def test_a_deliberately_ignored_name_still_reaches_no_set(tmp_path):
    """`other` is for names nothing recognises, not for names this package
    decided against; see is_ignored_image_name."""
    stage = make_files(tmp_path / "01-ruling-ibex", ["01-ruling-ibex_centring_SEM.tif"])

    assert build_loose_image_sets(stage) == LooseImageSets()


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

    assert [(one.series, one.modality) for one in found.sets] == [("other", None)]
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
    """A set of its own series is keyed by beam and could not take it, so it
    joins the residue -- and the finding keeps what its name did say."""
    lamella = make_files(tmp_path / "01-ruling-ibex", ["ref_PositionReady.tif"])

    found = build_loose_image_sets(lamella, resolve_modality=lambda path: None)

    assert [(one.series, one.modality) for one in found.sets] == [("other", None)]
    assert found.unresolved == ("ref_PositionReady.tif",)
    assert found.unrecognised == ()


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

    found = image_files_of(_image_sets(stage)[0])

    assert sorted(found) == ["a_SEM.tif", "b_SEM.tif"]
    assert found["a_SEM.tif"] == stage / "sem" / "a_SEM.tif"


# ---------------------------------------------------------------------------
# Stage names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "folder, expected",
    [
        ("adaptive_polish_2026-07-28-02-43-32PM", "adaptive_polish"),
        ("adaptive_polish_2025-04-15-02-48-15AM", "adaptive_polish"),
        ("adaptive_polish_2026-08-07-15-30-00", "adaptive_polish"),
        # Already bare: the 2024 layout, and the milling tasks the 2026 one
        # puts reference images in.
        ("adaptive_polish", "adaptive_polish"),
        ("Polishing", "Polishing"),
        ("Rough Milling", "Rough Milling"),
        # Nothing but a timestamp, which names no stage at all.
        ("2026-08-07-15-30-00", UNNAMED_STAGE),
    ],
)
def test_a_stage_name_keeps_what_repeats_and_drops_what_does_not(folder, expected):
    """What is left has to be shared by every lamella that ran the same step,
    or counting by it is counting folders."""
    assert stage_base_name(folder) == expected


def test_a_timestamp_inside_a_name_leaves_no_gap_behind():
    assert stage_base_name("ref_2025-04-15-02-48-15AM_run") == "ref_run"


def test_a_prediction_is_am_segs_only_under_am_seg():
    """Everything else convertible came from the microscope's pipeline."""
    from am_segmentation_manager.layout import prediction_origin

    assert prediction_origin(Path("/a/predictions/am_seg/img.tif")) == "am-seg"
    assert prediction_origin(Path("/a/predictions/raw/img.tif")) == "microscope"
    assert prediction_origin(Path("/a/predictions/img.tif")) == "microscope"


def test_the_preferred_source_ranks_first_and_post_processed_last():
    """Which prediction becomes a label is asked for, never left to listing order."""
    from am_segmentation_manager.layout import prediction_precedence

    am_seg = Path("/a/predictions/am_seg/img.tif")
    raw = Path("/a/predictions/raw/img.tif")
    clean = Path("/a/predictions/clean/img.tif")

    assert sorted([clean, raw, am_seg], key=prediction_precedence) == [am_seg, raw, clean]
    assert sorted(
        [clean, am_seg, raw], key=lambda path: prediction_precedence(path, "microscope")
    ) == [raw, am_seg, clean]
