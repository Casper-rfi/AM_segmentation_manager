"""Tests for how often a command scans to resolve its scope.

A scan of the collection is the slowest thing the package does, so a command
takes its projects from the scope it resolved rather than scanning again.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.scanner import Scanner
from conftest import build_dataset

SCOPE = [
    "--project", "all", "--lamella", "all", "--stage", "all", "--image", "all",
    "--modality", "all",
]


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    """One annotated project, in a folder of its own so outputs land beside it."""
    pytest.importorskip("cv2")
    root = tmp_path / "collection"
    build_dataset(root, acquisitions=2, labels="annotated")
    return root


@pytest.fixture
def scans(monkeypatch) -> list[Scanner]:
    """Record every index built, from the moment the fixture is requested."""
    built: list[Scanner] = []
    real = Scanner.build_index

    def counted(self, *args, **kwargs):
        built.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Scanner, "build_index", counted)
    return built


@pytest.mark.parametrize(
    "command",
    [
        lambda root, out: ["stats", str(root), *SCOPE, "--by", "autolamella-project"],
        lambda root, out: [
            "export-metadata-csv", str(root), *SCOPE, "--no-headers", "-o", str(out),
        ],
        lambda root, out: [
            "quantify-labels", "--start-path", str(root), *SCOPE, "--state", "all",
        ],
        lambda root, out: ["revert", str(root), *SCOPE, "--to", "no-such-savepoint"],
    ],
    ids=["stats", "export-metadata-csv", "quantify-labels", "revert --to"],
)
def test_a_command_scans_the_collection_once(command, collection, tmp_path, scans):
    main(command(collection, tmp_path / "out.csv"))

    assert len(scans) == 1


def test_explicit_image_paths_still_find_their_project(collection, capsys, scans):
    """Explicit paths resolve without a scan, so the project lookup needs one."""
    image = Scanner(collection).build_index().all_projects()[0].image_records[0]
    scans.clear()

    assert main([
        "stats", str(image.image_path), "--start-path", str(collection),
        "--by", "autolamella-project",
    ]) == 0

    assert "(not recorded)" not in capsys.readouterr().out
    assert len(scans) == 1
