"""Opt-in smoke test against a real dataset collection.

Every other test builds its tree under ``tmp_path``, which means every other
test asserts against layouts we thought to write down. This one runs against
actual data, and exists for the failure those cannot catch: **the scanner
quietly stops seeing a whole class of data.** A synthetic suite stays green
through that, because the layout it stopped seeing is the one nobody built a
fixture for.

Opt in by pointing an environment variable at a collection::

    AM_SEG_REFERENCE_COLLECTION=~/Data/20260807_lamella_segmentations_new pytest -q

Unset, every test here skips, so the suite runs anywhere.

**It never writes.** The collection is real data; a test that modified it would
be worse than no test. :func:`test_scanning_never_writes_to_the_collection`
checks that by comparing a snapshot of every file before and after.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from am_segmentation_manager.scanner import Scanner
from am_segmentation_manager.survey import survey_collection

ENV_VAR = "AM_SEG_REFERENCE_COLLECTION"

#: State subdirectories, which sit inside an image directory rather than
#: alongside it. Spelled out rather than imported: see `image_set_directory`.
STATE_DIRECTORIES = {"deselected", "corrected"}

#: Directory names that are never image sets.
NON_IMAGE_DIRECTORIES = {"predictions", "plots", "raw", "clean"}

#: Finding categories that the collection is *expected* to raise as errors.
#: Two projects there share a name and differ only by path -- a real condition
#: the survey is supposed to report, not a regression. Anything else appearing
#: at error severity should fail this test and be looked at.
EXPECTED_ERROR_CATEGORIES = {"duplicate-project-name"}


def collection_root() -> Path:
    """Return the collection to test against, or skip."""
    value = os.environ.get(ENV_VAR)
    if not value:
        pytest.skip(f"set {ENV_VAR} to a dataset collection to run this")
    root = Path(value).expanduser()
    if not root.is_dir():
        pytest.skip(f"{ENV_VAR} is not a directory: {root}")
    return root


@pytest.fixture(scope="module")
def root() -> Path:
    return collection_root()


@pytest.fixture(scope="module")
def index(root: Path):
    """Scan once; these tests are read-only so they can share the result."""
    return Scanner(root).build_index()


def records_of(index) -> list:
    return [
        record
        for project in index.all_projects()
        for lamella in project.lamellae
        for stage in lamella.stages
        for record in stage.images
    ]


def image_set_directory(image: Path) -> Path | None:
    """Return the image-set directory *image* sits in, or None.

    **Deliberately does not use** :mod:`am_segmentation_manager.layout`. The
    whole value of this test is that it counts images by a rule derived
    independently of the code under test; asking ``layout.py`` what counts as an
    image directory would make the comparison circular, and a bug that made the
    scanner blind to a layout would make this blind to it in the same way.

    The rule: an image lives in a directory whose own name starts with a
    modality and is not a label directory, or in a state subdirectory of one.
    Images sitting directly in a lamella directory -- alignment and reference
    shots -- are context, never segmentation targets, and are excluded.
    """
    parent = image.parent
    if parent.name.lower() in STATE_DIRECTORIES:
        parent = parent.parent
    name = parent.name.lower()
    if "label" in name or name in NON_IMAGE_DIRECTORIES:
        return None
    return parent if name.startswith(("sem", "fib")) else None


def images_on_disk(root: Path) -> set[Path]:
    """Every image file that ought to be indexed, found independently."""
    return {
        path.resolve()
        for path in root.rglob("*.tif")
        if image_set_directory(path) is not None
    }


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Return ``{path: (size, mtime_ns)}`` for every file under *root*."""
    return {
        str(path): (stat.st_size, stat.st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file() and (stat := path.stat())
    }


# ---------------------------------------------------------------------------
# The test this file exists for
# ---------------------------------------------------------------------------

def test_every_image_on_disk_is_indexed(root: Path, index):
    """No image that should be indexed is missing from the index.

    This is the one that catches a whole layout going invisible. A synthetic
    suite cannot: it only ever builds the layouts someone remembered.
    """
    indexed = {record.image_path.resolve() for record in records_of(index)}
    on_disk = images_on_disk(root)

    missing = on_disk - indexed
    assert not missing, (
        f"{len(missing)} image(s) on disk were not indexed. If a layout has "
        f"changed, add it to tests/conftest.py rather than only fixing this. "
        f"First few: {sorted(str(p.relative_to(root)) for p in missing)[:5]}"
    )


def test_nothing_is_indexed_that_is_not_on_disk(root: Path, index):
    """The converse: no invented records, and no stale paths."""
    indexed = {record.image_path.resolve() for record in records_of(index)}
    unexpected = indexed - images_on_disk(root)

    assert not unexpected, (
        f"{len(unexpected)} indexed image(s) are not where the index says. "
        f"First few: {sorted(str(p) for p in unexpected)[:5]}"
    )


def test_scanning_never_writes_to_the_collection(root: Path):
    """Scanning and surveying are read-only, on real data.

    Asserted rather than assumed, because this test suite is pointed at data
    that is expensive to lose.
    """
    before = snapshot(root)
    survey_collection(Scanner(root).start_path)
    Scanner(root).build_index()
    after = snapshot(root)

    assert after == before, "scanning modified the collection"


# ---------------------------------------------------------------------------
# Invariants that should hold of any collection
# ---------------------------------------------------------------------------

def test_projects_are_found(index):
    assert index.all_projects(), "no projects discovered"


def test_project_identities_are_unique(index):
    """Two projects may share a name; they must not share an identity.

    The collection really does contain two projects called
    ``AutoLamella-2025-07-21-15-21_AM01``, differing only by path. Keying the
    index by name silently discarded one of them, which is why identity is the
    relative path.
    """
    projects = index.all_projects()
    identities = [project.relative_path for project in projects]

    assert len(set(identities)) == len(projects)
    assert all(identities), "a project has no relative path"


def test_every_project_has_a_marker_file(root: Path, index):
    for project in index.all_projects():
        markers = [
            name for name in ("experiment.yaml", "AM_metadata.yaml")
            if (project.path / name).exists()
        ]
        assert markers, f"{project.relative_path} has no marker file"


def test_no_unexpected_error_findings(root: Path):
    """Pin the known errors so a new *kind* of error is visible.

    Asserting zero errors would fail today for a condition that is correctly
    reported rather than broken, so the allow-list is by category.
    """
    survey = survey_collection(root)
    unexpected = {
        finding.category
        for finding in survey.all_findings
        if finding.severity == "error"
    } - EXPECTED_ERROR_CATEGORIES

    assert not unexpected, f"new error categories: {sorted(unexpected)}"


# ---------------------------------------------------------------------------
# Does the collection still exercise the awkward cases?
# ---------------------------------------------------------------------------

def test_the_collection_still_covers_the_awkward_layouts(index):
    """Guard the test's own value, not the code.

    If the collection is replaced by one with no collapsed hierarchies and no
    imaging conditions, everything above keeps passing while checking much less.
    Failing here says the smoke test has lost its teeth -- point it at data that
    still has these, or accept the reduced coverage deliberately.

    Only properties the current collection actually has are required. It has no
    dataset lacking a *lamella* directory, so that is not asserted.
    """
    projects = index.all_projects()
    records = records_of(index)

    synthetic_stages = sum(
        1 for p in projects for lam in p.lamellae for s in lam.stages if s.synthetic
    )
    assert synthetic_stages, "no dataset without a stage directory"
    assert any(record.variant for record in records), "no imaging-condition variants"
    assert any(record.label_shared for record in records), "no shared labels"
