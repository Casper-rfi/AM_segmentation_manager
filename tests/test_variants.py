"""Selecting images by imaging-condition variant.

A variant is the suffix of an image directory name (``sem_5kv`` -> ``5kv``) and
is opaque to this package: it is matched exactly, never parsed, so arbitrary
names like ``5kv``, ``defocus`` or ``lowmag`` work without code changes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.filtering import iter_matching_records, matches_variant
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.scanner import Scanner


def _record(variant: str | None) -> ImageRecord:
    folder = f"sem_{variant}" if variant else "sem"
    return ImageRecord(image_path=Path(folder) / "img.tif", modality="SEM", variant=variant)


@pytest.mark.parametrize(
    "variant, selectors, expected",
    [
        ("5kv", ["5kv"], True),
        ("5kv", ["lowmag"], False),
        (None, ["base"], True),
        (None, ["5kv"], False),
        ("5kv", ["base"], False),
        ("5kv", ["all"], True),
        (None, ["all"], True),
        ("5kv", None, True),        # no selector means no filtering
        ("5kv", [], True),
        ("5kv", ["base,5kv"], True),  # comma-separated, like the other selectors
        ("defocus", ["defocus"], True),
        ("8_fi_16", ["8_fi_16"], True),
    ],
)
def test_variant_matching(variant, selectors, expected):
    assert matches_variant(_record(variant), selectors) is expected


def test_variants_are_matched_exactly_not_by_prefix():
    """``5kv`` must not match ``5kv_extra``: the string is opaque."""
    assert not matches_variant(_record("5kv_extra"), ["5kv"])


def _build_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"
    for directory in ("sem", "sem_5kv", "sem_lowmag", "fib"):
        (stage / directory).mkdir(parents=True)
        (stage / directory / "img_000.tif").write_bytes(b"")
    (project / "experiment.yaml").write_text("name: project\n", encoding="utf-8")
    return project


def _select(tmp_path: Path, **kwargs) -> list[str]:
    index = Scanner(tmp_path).build_index()
    records = iter_matching_records(
        index, project_selectors=["all"], lamella_selectors=["all"],
        stage_selectors=["all"], image_selectors=["all"], **kwargs,
    )
    return sorted(record.image_path.parent.name for record in records)


def test_every_variant_is_selected_by_default(tmp_path: Path):
    _build_project(tmp_path)
    assert _select(tmp_path) == ["fib", "sem", "sem_5kv", "sem_lowmag"]


def test_a_single_variant_can_be_selected(tmp_path: Path):
    _build_project(tmp_path)
    assert _select(tmp_path, variant_selectors=["5kv"]) == ["sem_5kv"]


def test_base_selects_the_unsuffixed_directories(tmp_path: Path):
    _build_project(tmp_path)
    assert _select(tmp_path, variant_selectors=["base"]) == ["fib", "sem"]


def test_variant_combines_with_modality(tmp_path: Path):
    _build_project(tmp_path)
    assert _select(tmp_path, variant_selectors=["base"], modality="SEM") == ["sem"]


def test_several_variants_can_be_selected_at_once(tmp_path: Path):
    _build_project(tmp_path)
    assert _select(tmp_path, variant_selectors=["5kv", "lowmag"]) == ["sem_5kv", "sem_lowmag"]


def test_cli_accepts_a_variant_selector(tmp_path: Path):
    """The flag reaches the record resolution used by every scoped command."""
    from am_segmentation_manager.cli import main

    project = _build_project(tmp_path)
    stage = project / "01-lamella" / "adaptive_polish_2025-04-15-02-48-15AM"

    result = main([
        "create-training-csv", "--start-path", str(tmp_path), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all",
        "--variant", "5kv", "--state", "all", "--modality", "SEM",
        "-o", str(tmp_path / "out"), "--split", "1.0",
    ])

    assert result == 0
    # No labels exist, so nothing is exported; the point is that the flag parses
    # and narrows the selection without error.
    assert (tmp_path / "out" / "dataset_train.csv").exists()
    assert stage.exists()
