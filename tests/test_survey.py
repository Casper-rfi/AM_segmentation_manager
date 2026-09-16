"""Tests for the read-only layout survey.

The layout variety is the risk this package has to manage, so the test
structure mirrors it: :func:`make_project` builds each real layout era, and one
shared set of assertions runs against all of them. A new layout found in the
wild becomes a new parameter here, and every existing assertion then covers it.
"""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.reporting import format_survey
from am_segmentation_manager.survey import (
    find_project_roots,
    find_stage_directories,
    survey_collection,
    survey_project,
)
from conftest import build_dataset


def _write_image(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"")
    return path


#: The samples of the pre-Autolamella datasets, which keep their own label names.
LEGACY_LAYOUTS = ("no_stage_level", "no_lamella_level")


def make_project(root: Path, layout: str, name: str = "project") -> Path:
    """Build a project of the given sample layout under *root*.

    A thin adapter over the shared builder in ``conftest``, which is where the
    era definitions live now -- having both know how to build five eras meant
    two places to edit when a sixth turns up.

    ``convention`` label naming is requested explicitly: the survey's findings
    are largely *about* label directory names, so these tests want the current
    convention rather than the pre-convention names the real data of each era
    happened to use.
    """
    data = build_dataset(
        root,
        layout=layout,
        name=name,
        acquisitions=2,
        label_naming=None if layout in LEGACY_LAYOUTS else "convention",
        sem_variants=("SEM_8_fi_1", "SEM_8_fi_4") if layout == "no_stage_level" else (),
        # Autolamella writes alignment and rough-milling images beside the
        # stages; a survey indexes them as `other`, never as a target.
        out_of_scope=(layout == "nested_stage"),
    )
    return data.project


# ---------------------------------------------------------------------------
# Behaviour that must hold for every layout era
# ---------------------------------------------------------------------------

def test_every_era_yields_exactly_one_project(tmp_path: Path, sample_layout: str):
    make_project(tmp_path, sample_layout)
    survey = survey_collection(tmp_path)
    assert len(survey.projects) == 1


def test_every_era_finds_both_modalities(tmp_path: Path, sample_layout: str):
    """Of the sets that can be segmented: images no rule reaches have no modality."""
    make_project(tmp_path, sample_layout)
    survey = survey_collection(tmp_path)
    modalities = {
        entry.image_set.modality
        for entry in survey.projects[0].image_sets
        if entry.image_set.series != "other"
    }
    assert modalities == {"SEM", "FIB"}


def test_a_dataset_lacking_a_modality_does_not_have_one_invented(tmp_path: Path, sample_layout: str):
    """Whether a dataset has FIB is independent of which layout it uses.

    Two of the pre-Autolamella datasets have SEM only, but that is a property of
    those datasets rather than of the collapsed layout -- every lamella of the
    oldest dataset in the current collection has FIB. So this is asserted for
    every era: a survey must report the modality that is absent as absent, and
    never synthesise an image set for it.
    """
    build_dataset(tmp_path, layout=sample_layout, modalities=("SEM",))
    survey = survey_collection(tmp_path)
    modalities = {entry.image_set.modality for entry in survey.projects[0].image_sets}
    assert modalities == {"SEM"}


def test_every_era_pairs_every_image_with_a_label(tmp_path: Path, sample_layout: str):
    """Of the sets that can be segmented: an `other` image has nowhere for a label to go."""
    make_project(tmp_path, sample_layout)
    survey = survey_collection(tmp_path)
    for entry in survey.projects[0].image_sets:
        if entry.image_set.series == "other":
            continue
        assert entry.image_count > 0
        assert entry.images_without_labels == 0
        assert entry.labels_without_images == 0


def test_every_era_reports_a_marker_file(tmp_path: Path, sample_layout: str):
    make_project(tmp_path, sample_layout)
    survey = survey_collection(tmp_path)
    assert survey.projects[0].markers


def test_every_era_formats_without_error(tmp_path: Path, sample_layout: str):
    make_project(tmp_path, sample_layout)
    lines = format_survey(survey_collection(tmp_path), detail=True)
    assert any("Layout survey" in line for line in lines)


# ---------------------------------------------------------------------------
# Hierarchy interpretation
# ---------------------------------------------------------------------------

def test_nested_stage_is_found_below_intermediate_directories(tmp_path: Path):
    project = make_project(tmp_path, "nested_stage")
    stages = find_stage_directories(project)
    assert "adaptive_polish_2026-07-09-11-23-46AM" in [stage.name for stage in stages]


def test_a_directory_of_images_no_rule_recognises_is_still_a_stage(tmp_path: Path):
    """A set cannot be built in a directory the search never returned, so
    reaching every image means reaching every directory holding one."""
    project = make_project(tmp_path, "nested_stage")

    names = [stage.name for stage in find_stage_directories(project)]

    assert "Fine 1-11-01-38" in names
    assert "Rough-Milling" in names


def test_alignment_images_are_indexed_but_never_become_segmentation_targets(tmp_path):
    """Context only, which is now a series rather than an omission: they are
    countable and openable, and there is nowhere for a label of one to go."""
    make_project(tmp_path, "nested_stage")
    sets = [entry.image_set for entry in survey_collection(tmp_path).projects[0].image_sets]

    curated = {one.directory.name for one in sets if one.series != "other"}
    other = {one.directory.name for one in sets if one.series == "other"}

    assert curated == {"sem", "fib"}
    assert other == {"Fine 1-11-01-38", "Rough-Milling"}
    assert all(one.labels_directory is None for one in sets if one.series == "other")


def test_collapsed_layout_synthesises_lamella_and_stage(tmp_path: Path):
    """``project/SEM/`` has neither a lamella nor a stage directory."""
    project = make_project(tmp_path, "no_lamella_level")
    result = survey_project(project, tmp_path)

    assert len(result.lamellae) == 1
    lamella = result.lamellae[0]
    assert lamella.synthetic is True
    assert lamella.path == project
    assert lamella.stages[0].synthetic is True


def test_lamella_without_stage_directory_is_marked_synthetic(tmp_path: Path):
    """``project/lam_010/SEM_8/`` has a lamella but no stage directory."""
    project = make_project(tmp_path, "no_stage_level")
    result = survey_project(project, tmp_path)

    lamella = result.lamellae[0]
    assert lamella.name == "lam_010"
    assert lamella.synthetic is False
    assert lamella.stages[0].synthetic is True


def test_full_hierarchy_is_not_marked_synthetic(tmp_path: Path):
    project = make_project(tmp_path, "timestamped_stage")
    result = survey_project(project, tmp_path)

    lamella = result.lamellae[0]
    assert lamella.synthetic is False
    assert lamella.stages[0].synthetic is False


# ---------------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------------

def test_projects_are_discovered_inside_collection_directories(tmp_path: Path):
    collection = tmp_path / "20250414_adaptive_milling"
    make_project(collection, "timestamped_stage", name="AutoLamella-A")
    make_project(collection, "timestamped_stage", name="AutoLamella-B")

    roots = find_project_roots(tmp_path)
    assert {root.name for root in roots} == {"AutoLamella-A", "AutoLamella-B"}


def test_a_marker_inside_a_project_is_reported_rather_than_ignored(tmp_path: Path):
    """A stray marker splits a project, which must never happen silently.

    Both directories are indexed, because the alternative -- stopping at the
    outer marker -- would let a marker in a collection directory swallow every
    project beneath it. Splitting is visible and recoverable; swallowing is not.
    """
    project = make_project(tmp_path, "timestamped_stage")
    nested = project / "01-full-gannet"
    (nested / "experiment.yaml").write_text("name: nested\n", encoding="utf-8")

    assert set(find_project_roots(tmp_path)) == {project, nested}

    survey = survey_collection(tmp_path)
    nested_findings = [f for f in survey.all_findings if f.category == "nested-project"]
    assert len(nested_findings) == 1
    assert nested_findings[0].severity == "error"
    assert nested_findings[0].path == nested


def test_a_stage_boundary_stops_the_search_for_projects(tmp_path: Path):
    """Nothing below a stage can be a project, so image directories are not walked."""
    project = make_project(tmp_path, "timestamped_stage")
    stage = project / "01-full-gannet" / "adaptive_polish_2025-04-15-02-48-15AM"
    (stage / "sem" / "experiment.yaml").write_text("name: not-a-project\n", encoding="utf-8")

    assert find_project_roots(tmp_path) == [project]


def test_am_metadata_alone_marks_a_project(tmp_path: Path):
    """Pre-Autolamella datasets are made indexable by AM_metadata.yaml."""
    project = make_project(tmp_path, "no_lamella_level")
    assert not (project / "experiment.yaml").exists()
    assert find_project_roots(tmp_path) == [project]


def test_method_is_read_from_experiment_yaml(tmp_path: Path):
    make_project(tmp_path, "untimed_stage")
    survey = survey_collection(tmp_path)
    # The nested `method:` under `positions:` is indented and must not be read.
    assert survey.projects[0].method == "AutoLiftout"


def test_projects_are_identified_by_relative_path(tmp_path: Path):
    make_project(tmp_path, "timestamped_stage", name="AutoLamella-01")
    make_project(tmp_path / "FIB", "timestamped_stage", name="AutoLamella-01")

    survey = survey_collection(tmp_path)
    assert len(survey.projects) == 2
    assert {project.relative_path for project in survey.projects} == {
        "AutoLamella-01", "FIB/AutoLamella-01",
    }


def test_duplicate_project_names_are_reported_as_an_error(tmp_path: Path):
    make_project(tmp_path, "timestamped_stage", name="AutoLamella-01")
    make_project(tmp_path / "FIB", "timestamped_stage", name="AutoLamella-01")

    survey = survey_collection(tmp_path)
    duplicates = [
        finding for finding in survey.all_findings
        if finding.category == "duplicate-project-name"
    ]
    assert len(duplicates) == 2
    assert all(finding.severity == "error" for finding in duplicates)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def _categories(survey) -> set[str]:
    return {finding.category for finding in survey.all_findings}


def test_shared_labels_are_reported(tmp_path: Path):
    make_project(tmp_path, "no_stage_level")
    survey = survey_collection(tmp_path)

    shared = [entry for entry in survey.projects[0].image_sets if entry.image_set.labels_are_shared]
    assert {entry.image_set.name for entry in shared} == {
        "SEM_8", "SEM_8_fi_1", "SEM_8_fi_4",
    }
    assert "shared-labels" in _categories(survey)


def test_legacy_label_directories_are_reported_for_migration(tmp_path: Path):
    make_project(tmp_path, "no_stage_level")
    survey = survey_collection(tmp_path)

    legacy = [
        finding for finding in survey.all_findings
        if finding.category == "legacy-label-directory"
    ]
    assert legacy
    assert "sem_labels/" in legacy[0].message or "fib_labels/" in legacy[0].message


def test_images_without_labels_are_counted_and_reported(tmp_path: Path):
    project = make_project(tmp_path, "timestamped_stage")
    stage = project / "01-full-gannet" / "adaptive_polish_2025-04-15-02-48-15AM"
    _write_image(stage / "sem", "img_002_SEM.tif")

    survey = survey_collection(tmp_path)
    sem = next(
        entry for entry in survey.projects[0].image_sets
        if entry.image_set.name == "sem"
    )
    assert sem.image_count == 3
    assert sem.images_without_labels == 1
    assert "images-without-labels" in _categories(survey)


def test_labels_without_images_are_reported(tmp_path: Path):
    project = make_project(tmp_path, "timestamped_stage")
    stage = project / "01-full-gannet" / "adaptive_polish_2025-04-15-02-48-15AM"
    _write_image(stage / "sem_labels", "orphan.tif")

    survey = survey_collection(tmp_path)
    assert "labels-without-images" in _categories(survey)


def test_a_stage_with_no_labels_yet_is_reported_as_info(tmp_path: Path):
    """The state of every Autolamella project in the collection today."""
    project = make_project(tmp_path, "timestamped_stage")
    stage = project / "01-full-gannet" / "adaptive_polish_2025-04-15-02-48-15AM"
    for labels in ("sem_labels", "fib_labels"):
        for path in (stage / labels).iterdir():
            path.unlink()
        (stage / labels).rmdir()

    survey = survey_collection(tmp_path)
    findings = [f for f in survey.all_findings if f.category == "no-labels-yet"]
    assert len(findings) == 2
    assert all(finding.severity == "info" for finding in findings)
    assert "sem_labels/" in findings[0].message


def test_unexpected_stage_names_are_reported(tmp_path: Path):
    """The adaptive-polish heuristic is an inference, so surprises surface."""
    make_project(tmp_path, "no_stage_level")
    survey = survey_collection(tmp_path)
    assert "unexpected-stage-name" in _categories(survey)


def test_expected_stage_names_are_not_reported(tmp_path: Path):
    make_project(tmp_path, "timestamped_stage")
    survey = survey_collection(tmp_path)
    assert "unexpected-stage-name" not in _categories(survey)


def test_unrecognised_directories_inside_a_stage_are_reported(tmp_path: Path):
    project = make_project(tmp_path, "timestamped_stage")
    stage = project / "01-full-gannet" / "adaptive_polish_2025-04-15-02-48-15AM"
    (stage / "something_new").mkdir()

    survey = survey_collection(tmp_path)
    findings = [f for f in survey.all_findings if f.category == "unrecognised-directory"]
    assert [finding.path.name for finding in findings] == ["something_new"]


def test_a_project_with_no_images_is_reported(tmp_path: Path):
    project = tmp_path / "empty"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: empty\n", encoding="utf-8")

    survey = survey_collection(tmp_path)
    assert "no-images" in _categories(survey)


def test_finding_counts_are_grouped_by_severity(tmp_path: Path):
    make_project(tmp_path, "no_stage_level")
    counts = survey_collection(tmp_path).finding_counts()
    assert set(counts) == {"error", "warning", "info"}
    assert counts["warning"] >= 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_summary_output_omits_the_tree_but_detail_includes_it(tmp_path: Path):
    make_project(tmp_path, "timestamped_stage")
    survey = survey_collection(tmp_path)

    summary = "\n".join(format_survey(survey, detail=False))
    detailed = "\n".join(format_survey(survey, detail=True))

    assert "01-full-gannet" not in summary
    assert "01-full-gannet" in detailed


def test_severity_filter_hides_lesser_findings_but_not_the_counts(tmp_path: Path):
    make_project(tmp_path, "no_stage_level")
    survey = survey_collection(tmp_path)

    errors_only = "\n".join(format_survey(survey, min_severity="error"))
    assert "shared-labels" not in errors_only
    # The counts line still reports every finding that exists.
    assert "info" in errors_only


def test_survey_never_modifies_the_filesystem(tmp_path: Path):
    make_project(tmp_path, "no_stage_level")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    survey_collection(tmp_path)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after


# ---------------------------------------------------------------------------
# Declared workflow state
# ---------------------------------------------------------------------------

def _findings_for(root: Path, category: str) -> list:
    """Return the findings of one category from a survey of *root*."""
    from am_segmentation_manager.survey import survey_collection

    return [
        finding
        for finding in survey_collection(root).all_findings
        if finding.category == category
    ]


def test_a_stage_still_keeping_state_in_folders_is_reported(tmp_path: Path):
    from conftest import build_dataset

    dataset = build_dataset(tmp_path, modalities=("SEM",))
    name = dataset.filenames["SEM"][0]
    nested = dataset.image_dir("SEM") / "deselected"
    nested.mkdir()
    (dataset.image_dir("SEM") / name).rename(nested / name)

    findings = _findings_for(tmp_path, "unmigrated-state")

    assert len(findings) == 1
    assert "migrate_state.py" in findings[0].message


def test_nested_files_after_migration_ask_for_the_move(tmp_path: Path):
    from am_segmentation_manager.state import read_state, write_state
    from conftest import build_dataset

    dataset = build_dataset(tmp_path, modalities=("SEM",))
    name = dataset.filenames["SEM"][0]
    nested = dataset.image_dir("SEM") / "deselected"
    nested.mkdir()
    (dataset.image_dir("SEM") / name).rename(nested / name)
    write_state(
        read_state(dataset.stage),
        {f"{dataset.label_dir('SEM').name}/{name}": "deselected"},
    )

    findings = _findings_for(tmp_path, "nested-files")

    assert len(findings) == 1
    assert "--move-files" in findings[0].message


def test_empty_state_folders_are_noted_as_clutter(tmp_path: Path):
    from am_segmentation_manager.state import read_state, write_state
    from conftest import build_dataset

    dataset = build_dataset(tmp_path, modalities=("SEM",))
    (dataset.image_dir("SEM") / "corrected").mkdir()
    write_state(read_state(dataset.stage), {})

    findings = _findings_for(tmp_path, "empty-state-folders")

    assert len(findings) == 1
    assert findings[0].severity == "info"


def test_an_entry_naming_a_group_that_is_gone_is_reported(tmp_path: Path):
    """A drive that was not mounted must not have its curation deleted for it."""
    from am_segmentation_manager.state import read_state, write_state
    from conftest import build_dataset

    dataset = build_dataset(tmp_path, modalities=("SEM",))
    write_state(
        read_state(dataset.stage),
        {f"{dataset.label_dir('SEM').name}/vanished.tif": "corrected"},
    )

    findings = _findings_for(tmp_path, "orphaned-state")

    assert len(findings) == 1
    assert "vanished.tif" in findings[0].message


def test_a_clean_migrated_stage_reports_nothing_about_state(tmp_path: Path):
    from am_segmentation_manager.state import read_state, write_state
    from conftest import build_dataset

    dataset = build_dataset(tmp_path, modalities=("SEM",))
    name = dataset.filenames["SEM"][0]
    write_state(
        read_state(dataset.stage),
        {f"{dataset.label_dir('SEM').name}/{name}": "deselected"},
    )

    for category in (
        "unmigrated-state", "nested-files", "empty-state-folders",
        "orphaned-state", "unreadable-state",
    ):
        assert _findings_for(tmp_path, category) == [], category


def test_an_unreadable_state_file_is_an_error(tmp_path: Path):
    from am_segmentation_manager.state import state_file_path
    from conftest import build_dataset

    dataset = build_dataset(tmp_path, modalities=("SEM",))
    state_file_path(dataset.stage).write_text("state:\n  - [\n", encoding="utf-8")

    findings = _findings_for(tmp_path, "unreadable-state")

    assert len(findings) == 1
    assert findings[0].severity == "error"


# ---------------------------------------------------------------------------
# Data the collection does not contain yet


def _image(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "img.tif").write_bytes(b"i")


def test_a_survey_reports_data_that_nothing_indexes(tmp_path: Path):
    """The failure this closes: a folder of unindexed datasets surveyed as a
    folder of none, because the survey counted only what was already a project.
    """
    from am_segmentation_manager.survey import survey_collection

    _image(tmp_path / "indexed" / "lam_001" / "sem")
    (tmp_path / "indexed" / "experiment.yaml").write_text("name: p\n")
    _image(tmp_path / "unindexed" / "lam_001" / "sem")

    found = survey_collection(tmp_path, check_labels=False)

    assert [project.name for project in found.projects] == ["indexed"]
    assert [item.path.name for item in found.unindexed] == ["unindexed"]
    reported = [
        finding for finding in found.all_findings
        if finding.category == "unindexed-dataset"
    ]
    assert len(reported) == 1
    assert "1 lamella(e) and 1 image(s)" in reported[0].message


def test_a_survey_names_a_folder_that_gathers_datasets(tmp_path: Path):
    from am_segmentation_manager.survey import survey_collection

    _image(tmp_path / "indexed" / "lam_001" / "sem")
    (tmp_path / "indexed" / "experiment.yaml").write_text("name: p\n")
    _image(tmp_path / "to sort" / "2024_dataset" / "lam_001" / "sem")
    _image(tmp_path / "to sort" / "2026_dataset" / "lam_001" / "stage" / "sem")

    found = survey_collection(tmp_path, check_labels=False)

    gathered = [
        finding for finding in found.all_findings
        if finding.category == "gathered-datasets"
    ]
    assert len(gathered) == 1
    assert "2024_dataset" in gathered[0].message
    assert gathered[0].severity == "warning"


def test_a_collection_with_nothing_unindexed_reports_none(tmp_path: Path):
    from am_segmentation_manager.survey import survey_collection

    _image(tmp_path / "indexed" / "lam_001" / "sem")
    (tmp_path / "indexed" / "experiment.yaml").write_text("name: p\n")

    found = survey_collection(tmp_path, check_labels=False)

    assert found.unindexed == ()
    assert not [
        finding for finding in found.all_findings
        if finding.category in {"unindexed-dataset", "gathered-datasets"}
    ]


def test_the_survey_headline_counts_unindexed_images(tmp_path: Path):
    """On the headline rather than only among the findings: a folder whose data
    is mostly not indexed should say so where the project count is read."""
    from am_segmentation_manager.reporting import format_survey
    from am_segmentation_manager.survey import survey_collection

    _image(tmp_path / "indexed" / "lam_001" / "sem")
    (tmp_path / "indexed" / "experiment.yaml").write_text("name: p\n")
    _image(tmp_path / "unindexed" / "lam_001" / "sem")

    lines = format_survey(survey_collection(tmp_path, check_labels=False))

    assert any("1 directory(ies) hold 1 image(s)" in line for line in lines)


# ---------------------------------------------------------------------------
# Images belonging to no project at all
# ---------------------------------------------------------------------------

def _finding(survey, category: str):
    return next(f for f in survey.all_findings if f.category == category)


def test_an_image_beside_a_project_is_reported_as_stray(tmp_path: Path):
    """The failure neither project-scoped check can see: `unindexed-images`
    walks one project, and a loose file is not a dataset candidate either."""
    make_project(tmp_path, "nested_stage")
    _write_image(tmp_path, "CB015_A2_overview.tif")

    message = _finding(survey_collection(tmp_path), "stray-images").message

    assert "CB015_A2_overview.tif" in message
    assert "1 image file(s)" in message


def test_an_image_below_a_directory_beside_a_project_is_reported(tmp_path: Path):
    """204 of the 215 found in the reference collection sat three levels down,
    under a `log/` directory nothing else descends into."""
    make_project(tmp_path, "nested_stage")
    _write_image(tmp_path / "log" / "data" / "crosscorrelation", "1748512510_xcorr.tif")

    assert "1748512510_xcorr.tif" in _finding(
        survey_collection(tmp_path), "stray-images"
    ).message


def test_a_fully_indexed_collection_reports_no_stray_images(tmp_path: Path):
    """Every image of a project is inside it, so the walk must find nothing."""
    make_project(tmp_path, "nested_stage")

    assert "stray-images" not in _categories(survey_collection(tmp_path))


def test_an_image_inside_a_project_is_never_counted_as_stray(tmp_path: Path):
    """The walk stops at a project root; what is inside is that project's to
    report, and one file may not raise two findings."""
    project = make_project(tmp_path, "nested_stage")
    _write_image(project, "overview-image-2026-07-28_10-31-00.tif")

    assert "stray-images" not in _categories(survey_collection(tmp_path))


def test_a_directory_already_reported_as_a_dataset_is_not_reported_twice(tmp_path: Path):
    """`unindexed-dataset` already names it, and says the more useful thing."""
    make_project(tmp_path, "nested_stage")
    _write_image(tmp_path / "unmarked" / "lam_001" / "sem", "img_000_SEM.tif")

    categories = _categories(survey_collection(tmp_path))

    assert "unindexed-dataset" in categories
    assert "stray-images" not in categories


def test_a_deliberately_ignored_name_is_not_reported_as_stray(tmp_path: Path):
    """Not indexed *and* not reported, wherever it sits."""
    make_project(tmp_path, "nested_stage")
    _write_image(tmp_path, "01-ruling-ibex_centring_SEM.tif")

    assert "stray-images" not in _categories(survey_collection(tmp_path))


def test_a_label_left_beside_a_project_is_not_reported_as_stray(tmp_path: Path):
    """A directory whose role is known is not walked, here as everywhere."""
    make_project(tmp_path, "nested_stage")
    _write_image(tmp_path / "sem_labels", "img_000_SEM.tif")

    assert "stray-images" not in _categories(survey_collection(tmp_path))
