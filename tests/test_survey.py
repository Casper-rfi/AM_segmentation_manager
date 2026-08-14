"""Tests for the read-only layout survey.

The layout variety is the risk this package has to manage, so the test
structure mirrors it: :func:`make_project` builds each real layout era, and one
shared set of assertions runs against all of them. A new layout found in the
wild becomes a new parameter here, and every existing assertion then covers it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.reporting import format_survey
from am_segmentation_manager.survey import (
    find_project_roots,
    find_stage_directories,
    survey_collection,
    survey_project,
)

#: The layout eras seen in the real collection, newest first.
LAYOUT_ERAS = (
    "v2026_nested",
    "v2025_timestamped",
    "v2024_untimed",
    "legacy_variants",
    "legacy_collapsed",
)


def _write_image(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"")
    return path


def _write_experiment(project: Path, name: str, method: str = "AutoLamella-OnGrid") -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / "experiment.yaml").write_text(
        f"created_at: 1753107713.652424\nmethod: {method}\nname: {name}\n"
        "positions:\n  - name: nested\n    method: should-not-be-read\n",
        encoding="utf-8",
    )


def make_project(root: Path, era: str, name: str = "project") -> Path:
    """Build a project of the given layout era under *root*.

    Returns the project root. Every era gets one lamella with one stage holding
    two images per modality, so counts are comparable across eras.
    """
    project = root / name
    if era == "v2026_nested":
        _write_experiment(project, name)
        stage = project / "01-neat-ape" / "Milling" / "Polishing" / "adaptive_polish_2026-07-09-11-23-46AM"
        # Autolamella writes alignment images that are explicitly out of scope.
        _write_image(project / "01-neat-ape" / "Alignment" / "Fine 1-11-01-38", "ref.tif")
        _write_image(project / "01-neat-ape" / "Milling" / "Rough-Milling", "ignored.tif")
    elif era == "v2025_timestamped":
        _write_experiment(project, name)
        stage = project / "01-full-gannet" / "adaptive_polish_2025-04-15-02-48-15AM"
    elif era == "v2024_untimed":
        _write_experiment(project, name, method="AutoLiftout")
        stage = project / "01-real-marmot" / "adaptive_polish"
    elif era == "legacy_variants":
        (project).mkdir(parents=True, exist_ok=True)
        (project / "AM_metadata.yaml").write_text("metadata:\n  sample: HeLa\n", encoding="utf-8")
        stage = project / "lam_010"
        for variant in ("SEM_8_fi_1", "SEM_8_fi_4"):
            for index in (0, 1):
                _write_image(stage / variant, f"img_000{index}.tif")
        for index in (0, 1):
            _write_image(stage / "labels", f"img_000{index}.tif")
            _write_image(stage / "labels_FIB", f"img_000{index}.tif")
        for index in (0, 1):
            _write_image(stage / "SEM_8", f"img_000{index}.tif")
            _write_image(stage / "FIB", f"img_000{index}.tif")
        return project
    elif era == "legacy_collapsed":
        project.mkdir(parents=True, exist_ok=True)
        (project / "AM_metadata.yaml").write_text("metadata:\n  sample: HeLa\n", encoding="utf-8")
        for index in (0, 1):
            _write_image(project / "SEM", f"SEM_{index}.tif")
            _write_image(project / "labels", f"SEM_{index}.tif")
        return project
    else:  # pragma: no cover - guard against a typo in a parameter list
        raise ValueError(f"Unknown layout era: {era}")

    for index in (0, 1):
        _write_image(stage / "sem", f"img_00{index}_SEM.tif")
        _write_image(stage / "fib", f"img_00{index}_FIB.tif")
        _write_image(stage / "sem_labels", f"img_00{index}_SEM.tif")
        _write_image(stage / "fib_labels", f"img_00{index}_FIB.tif")
    _write_image(stage / "plots", "img_000.png")
    (stage / "predictions" / "clean").mkdir(parents=True)
    return project


@pytest.fixture(params=LAYOUT_ERAS)
def era(request) -> str:
    """Run a test against every known layout era."""
    return request.param


# ---------------------------------------------------------------------------
# Behaviour that must hold for every layout era
# ---------------------------------------------------------------------------

def test_every_era_yields_exactly_one_project(tmp_path: Path, era: str):
    make_project(tmp_path, era)
    survey = survey_collection(tmp_path)
    assert len(survey.projects) == 1


def test_every_era_finds_both_modalities(tmp_path: Path, era: str):
    if era == "legacy_collapsed":
        pytest.skip("this era has no FIB images")
    make_project(tmp_path, era)
    survey = survey_collection(tmp_path)
    modalities = {entry.image_set.modality for entry in survey.projects[0].image_sets}
    assert modalities == {"SEM", "FIB"}


def test_every_era_pairs_every_image_with_a_label(tmp_path: Path, era: str):
    make_project(tmp_path, era)
    survey = survey_collection(tmp_path)
    for entry in survey.projects[0].image_sets:
        assert entry.image_count > 0
        assert entry.images_without_labels == 0
        assert entry.labels_without_images == 0


def test_every_era_reports_a_marker_file(tmp_path: Path, era: str):
    make_project(tmp_path, era)
    survey = survey_collection(tmp_path)
    assert survey.projects[0].markers


def test_every_era_formats_without_error(tmp_path: Path, era: str):
    make_project(tmp_path, era)
    lines = format_survey(survey_collection(tmp_path), detail=True)
    assert any("Layout survey" in line for line in lines)


# ---------------------------------------------------------------------------
# Hierarchy interpretation
# ---------------------------------------------------------------------------

def test_nested_stage_is_found_below_intermediate_directories(tmp_path: Path):
    project = make_project(tmp_path, "v2026_nested")
    stages = find_stage_directories(project)
    assert [stage.name for stage in stages] == ["adaptive_polish_2026-07-09-11-23-46AM"]


def test_alignment_directories_are_not_indexed(tmp_path: Path):
    """Alignment images are context only and out of scope."""
    make_project(tmp_path, "v2026_nested")
    survey = survey_collection(tmp_path)
    indexed = {entry.image_set.directory.name for entry in survey.projects[0].image_sets}
    assert indexed == {"sem", "fib"}


def test_collapsed_layout_synthesises_lamella_and_stage(tmp_path: Path):
    """``project/SEM/`` has neither a lamella nor a stage directory."""
    project = make_project(tmp_path, "legacy_collapsed")
    result = survey_project(project, tmp_path)

    assert len(result.lamellae) == 1
    lamella = result.lamellae[0]
    assert lamella.synthetic is True
    assert lamella.path == project
    assert lamella.stages[0].synthetic is True


def test_lamella_without_stage_directory_is_marked_synthetic(tmp_path: Path):
    """``project/lam_010/SEM_8/`` has a lamella but no stage directory."""
    project = make_project(tmp_path, "legacy_variants")
    result = survey_project(project, tmp_path)

    lamella = result.lamellae[0]
    assert lamella.name == "lam_010"
    assert lamella.synthetic is False
    assert lamella.stages[0].synthetic is True


def test_full_hierarchy_is_not_marked_synthetic(tmp_path: Path):
    project = make_project(tmp_path, "v2025_timestamped")
    result = survey_project(project, tmp_path)

    lamella = result.lamellae[0]
    assert lamella.synthetic is False
    assert lamella.stages[0].synthetic is False


# ---------------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------------

def test_projects_are_discovered_inside_collection_directories(tmp_path: Path):
    collection = tmp_path / "20250414_adaptive_milling"
    make_project(collection, "v2025_timestamped", name="AutoLamella-A")
    make_project(collection, "v2025_timestamped", name="AutoLamella-B")

    roots = find_project_roots(tmp_path)
    assert {root.name for root in roots} == {"AutoLamella-A", "AutoLamella-B"}


def test_a_marker_inside_a_project_is_reported_rather_than_ignored(tmp_path: Path):
    """A stray marker splits a project, which must never happen silently.

    Both directories are indexed, because the alternative -- stopping at the
    outer marker -- would let a marker in a collection directory swallow every
    project beneath it. Splitting is visible and recoverable; swallowing is not.
    """
    project = make_project(tmp_path, "v2025_timestamped")
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
    project = make_project(tmp_path, "v2025_timestamped")
    stage = project / "01-full-gannet" / "adaptive_polish_2025-04-15-02-48-15AM"
    (stage / "sem" / "experiment.yaml").write_text("name: not-a-project\n", encoding="utf-8")

    assert find_project_roots(tmp_path) == [project]


def test_am_metadata_alone_marks_a_project(tmp_path: Path):
    """Pre-Autolamella datasets are made indexable by AM_metadata.yaml."""
    project = make_project(tmp_path, "legacy_collapsed")
    assert not (project / "experiment.yaml").exists()
    assert find_project_roots(tmp_path) == [project]


def test_method_is_read_from_experiment_yaml(tmp_path: Path):
    make_project(tmp_path, "v2024_untimed")
    survey = survey_collection(tmp_path)
    # The nested `method:` under `positions:` is indented and must not be read.
    assert survey.projects[0].method == "AutoLiftout"


def test_projects_are_identified_by_relative_path(tmp_path: Path):
    make_project(tmp_path, "v2025_timestamped", name="AutoLamella-01")
    make_project(tmp_path / "FIB", "v2025_timestamped", name="AutoLamella-01")

    survey = survey_collection(tmp_path)
    assert len(survey.projects) == 2
    assert {project.relative_path for project in survey.projects} == {
        "AutoLamella-01", str(Path("FIB") / "AutoLamella-01"),
    }


def test_duplicate_project_names_are_reported_as_an_error(tmp_path: Path):
    make_project(tmp_path, "v2025_timestamped", name="AutoLamella-01")
    make_project(tmp_path / "FIB", "v2025_timestamped", name="AutoLamella-01")

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
    make_project(tmp_path, "legacy_variants")
    survey = survey_collection(tmp_path)

    shared = [entry for entry in survey.projects[0].image_sets if entry.image_set.labels_are_shared]
    assert {entry.image_set.name for entry in shared} == {
        "SEM_8", "SEM_8_fi_1", "SEM_8_fi_4",
    }
    assert "shared-labels" in _categories(survey)


def test_legacy_label_directories_are_reported_for_migration(tmp_path: Path):
    make_project(tmp_path, "legacy_variants")
    survey = survey_collection(tmp_path)

    legacy = [
        finding for finding in survey.all_findings
        if finding.category == "legacy-label-directory"
    ]
    assert legacy
    assert "sem_labels/" in legacy[0].message or "fib_labels/" in legacy[0].message


def test_images_without_labels_are_counted_and_reported(tmp_path: Path):
    project = make_project(tmp_path, "v2025_timestamped")
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
    project = make_project(tmp_path, "v2025_timestamped")
    stage = project / "01-full-gannet" / "adaptive_polish_2025-04-15-02-48-15AM"
    _write_image(stage / "sem_labels", "orphan.tif")

    survey = survey_collection(tmp_path)
    assert "labels-without-images" in _categories(survey)


def test_a_stage_with_no_labels_yet_is_reported_as_info(tmp_path: Path):
    """The state of every Autolamella project in the collection today."""
    project = make_project(tmp_path, "v2025_timestamped")
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
    make_project(tmp_path, "legacy_variants")
    survey = survey_collection(tmp_path)
    assert "unexpected-stage-name" in _categories(survey)


def test_expected_stage_names_are_not_reported(tmp_path: Path):
    make_project(tmp_path, "v2025_timestamped")
    survey = survey_collection(tmp_path)
    assert "unexpected-stage-name" not in _categories(survey)


def test_unrecognised_directories_inside_a_stage_are_reported(tmp_path: Path):
    project = make_project(tmp_path, "v2025_timestamped")
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
    make_project(tmp_path, "legacy_variants")
    counts = survey_collection(tmp_path).finding_counts()
    assert set(counts) == {"error", "warning", "info"}
    assert counts["warning"] >= 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_summary_output_omits_the_tree_but_detail_includes_it(tmp_path: Path):
    make_project(tmp_path, "v2025_timestamped")
    survey = survey_collection(tmp_path)

    summary = "\n".join(format_survey(survey, detail=False))
    detailed = "\n".join(format_survey(survey, detail=True))

    assert "01-full-gannet" not in summary
    assert "01-full-gannet" in detailed


def test_severity_filter_hides_lesser_findings_but_not_the_counts(tmp_path: Path):
    make_project(tmp_path, "legacy_variants")
    survey = survey_collection(tmp_path)

    errors_only = "\n".join(format_survey(survey, min_severity="error"))
    assert "shared-labels" not in errors_only
    # The counts line still reports every finding that exists.
    assert "info" in errors_only


def test_survey_never_modifies_the_filesystem(tmp_path: Path):
    make_project(tmp_path, "legacy_variants")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    survey_collection(tmp_path)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
