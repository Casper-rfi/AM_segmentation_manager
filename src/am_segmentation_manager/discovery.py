"""Locating projects, lamellae and stages on disk.

This module answers "where are the things" using the naming conventions in
:mod:`am_segmentation_manager.layout`. It sits between the conventions and the
two consumers that need to traverse a dataset tree: the scanner, which builds
the project model, and the survey, which reports on what it finds. Both must
agree about what counts as a project and where its stages are, so the traversal
lives here once.

The hierarchy Autolamella writes is project → lamella → stage → image
directories, but the older datasets collapse levels: some have no stage
directory, and the oldest have neither a stage nor a lamella directory. Rather
than require those levels to be created on disk, the missing ones are
*synthesised* -- the parent directory stands in -- which
:class:`DiscoveredStage` records so displays can be honest about it.

Traversal is deliberately tolerant: an unreadable directory yields nothing
rather than raising, so one bad directory cannot abort a scan of a whole
collection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .layout import build_image_sets, directory_role, image_files, subdirectories_by_name

logger = logging.getLogger(__name__)

#: Files whose presence marks a directory as a project root.
#:
#: ``experiment.yaml`` is Autolamella's own file. ``AM_metadata.yaml`` is this
#: package's, and marks datasets that predate Autolamella: writing a synthetic
#: ``experiment.yaml`` for those would misrepresent their provenance and could
#: collide if Autolamella were ever pointed at the directory.
#:
#: ``protocol.yaml`` is deliberately *not* a marker. It is absent from some
#: genuine Autolamella projects, so testing for it misclassifies them.
PROJECT_MARKERS: tuple[str, ...] = ("experiment.yaml", "AM_metadata.yaml")

#: How deep to search below the scan root for project roots, and below a
#: project root for stages. The deepest layout seen needs 4; the limit bounds
#: the cost of scanning an unexpected tree.
DEFAULT_MAX_DEPTH: int = 6


@dataclass(frozen=True)
class DiscoveredStage:
    """A stage, and the lamella it belongs to.

    Attributes:
        path: The stage directory: the one holding the image directories.
        lamella_path: The lamella directory the stage belongs to.
        lamella_is_synthetic: True when the project has no lamella directories,
            so the project root stands in for one.
        stage_is_synthetic: True when the lamella has no stage directories, so
            the lamella directory stands in for one.
    """

    path: Path
    lamella_path: Path
    lamella_is_synthetic: bool
    stage_is_synthetic: bool


def project_markers(directory: Path) -> tuple[str, ...]:
    """Return which project marker files *directory* holds."""
    markers: list[str] = []
    for marker in PROJECT_MARKERS:
        try:
            if (directory / marker).is_file():
                markers.append(marker)
        except OSError as exc:  # pragma: no cover - transient filesystem error
            logger.warning("Could not check %s in %s: %s", marker, directory, exc)
    return tuple(markers)


def find_project_roots(root: Path, max_depth: int = DEFAULT_MAX_DEPTH) -> list[Path]:
    """Return every project root at or below *root*, outermost first.

    The search continues **past** a project root rather than stopping at it.
    Stopping would mean a marker file in a collection directory silently
    swallowed every project beneath it: their stages would be absorbed as that
    directory's lamellae and the real projects would vanish from the index.
    Since :func:`find_stage_directories` refuses to cross a project boundary,
    descending is safe, and nesting becomes something to report rather than
    something that quietly merges data.

    The search stops at a stage that holds images, since nothing below one can
    be a project. Outside a project, directory names carry no guaranteed
    meaning: the reference collection has a collection directory named ``FIB/``
    holding five real projects, so neither a name-based filter nor plain image
    *set* detection can be used here -- both would mistake that directory for
    an image directory and skip everything inside it. Whether the directory
    actually contains image files is the reliable distinction.
    """
    found: list[Path] = []

    def walk(directory: Path, depth: int) -> None:
        if project_markers(directory):
            found.append(directory)
        if depth >= max_depth:
            logger.debug("Stopping at depth %d: %s", depth, directory)
            return
        if holds_images(directory):
            return
        for child in subdirectories_by_name(directory).values():
            walk(child, depth + 1)

    walk(root, 0)
    return found


def holds_images(directory: Path) -> bool:
    """Return whether *directory* is a stage that actually contains images.

    Stricter than "has image directories", which is the test
    :func:`find_stage_directories` uses. Inside a project an empty ``sem/`` is a
    real, if empty, image set worth reporting; outside one, a directory merely
    *named* like an image directory proves nothing.
    """
    return any(image_files(image_set.directory) for image_set in build_image_sets(directory))


def find_dataset_candidates(
    root: Path, max_depth: int = DEFAULT_MAX_DEPTH
) -> list[Path]:
    """Return directories that hold images but are not yet projects.

    A dataset predating Autolamella has no ``experiment.yaml``, so
    :func:`find_project_roots` cannot see it -- which leaves the datasets that
    most need a marker file invisible to the command that writes one. This
    answers the other question: not "where are the projects" but "what looks
    like a dataset and is not one yet".

    A directory qualifies when it has no marker of its own, holds image
    directories somewhere beneath it, and contains **no marked project** in its
    subtree. That last condition is what separates a dataset from a collection
    directory: ``20240806_auNP/`` also holds images beneath it, but it holds
    them inside three real projects, so it is a container and its children are
    the datasets.

    Matching stops at the **outermost** directory that qualifies. Descending
    further would propose a marker inside each *lamella*, splitting one dataset
    into several projects -- the opposite of the intended effect, and
    destructive of the hierarchy the marker is meant to establish.

    A directory that already has a marker yields nothing, including when it is
    the directory searched: its contents belong to that project.

    **Known limitation.** Which level is "the project" is not decidable from
    structure alone: ``collection/dataset/lamella/`` and
    ``dataset/lamella/stage/`` look identical without a marker to anchor them.
    Stopping at the outermost qualifying directory is therefore a choice, and
    it is the right one wherever a collection holds at least one *marked*
    project -- the marked ones anchor the level, which is the case in practice.
    A directory holding nothing but a single unmarked dataset is the ambiguous
    case: this proposes that directory rather than the dataset inside it. The
    proposal names the directory, and ``--dry-run`` shows it before anything is
    written, so the guess is visible rather than silent.
    """
    if max_depth < 0:
        return []
    if project_markers(root):
        return []
    if not _contains_marked_project(root, max_depth) and _contains_images(root, max_depth):
        return [root]

    candidates: list[Path] = []
    for child in subdirectories_by_name(root).values():
        candidates.extend(find_dataset_candidates(child, max_depth - 1))
    return candidates


def _contains_marked_project(directory: Path, max_depth: int) -> bool:
    """Return whether any project marker exists below *directory*."""
    return bool(find_project_roots(directory, max_depth=max_depth))


def _contains_images(directory: Path, max_depth: int) -> bool:
    """Return whether any stage holding images exists at or below *directory*."""
    return any(
        holds_images(stage) for stage in find_stage_directories(directory, max_depth=max_depth)
    )


def find_nested_project_roots(
    project_root: Path, max_depth: int = DEFAULT_MAX_DEPTH
) -> list[Path]:
    """Return project roots found *inside* another project.

    Normally empty. A non-empty result means either a marker file in a
    collection directory that also holds projects, or a stray marker inside a
    lamella. Both are worth reporting: the containing directory's stages are
    split in a way nobody intended.
    """
    return [root for root in find_project_roots(project_root, max_depth) if root != project_root]


def find_stage_directories(
    project_root: Path, max_depth: int = DEFAULT_MAX_DEPTH
) -> list[Path]:
    """Return the stage directories in a project, outermost first.

    A stage is a directory holding at least one image directory. The project
    root itself is a candidate, which is what supports the oldest layouts where
    neither the lamella nor the stage level exists as a folder.

    The search stops at a **nested project root**: those stages belong to that
    project, not this one. It also does not descend into a stage, nor into
    image, label, prediction, plot or state directories, which bounds the cost
    on trees holding thousands of files.
    """
    found: list[Path] = []

    def walk(directory: Path, depth: int) -> None:
        if directory != project_root and project_markers(directory):
            logger.debug("Not crossing into nested project %s", directory)
            return
        if build_image_sets(directory):
            found.append(directory)
            return
        if depth >= max_depth:
            logger.debug("Stopping at depth %d: %s", depth, directory)
            return
        for child in subdirectories_by_name(directory).values():
            if directory_role(child.name) == "unknown":
                walk(child, depth + 1)

    walk(project_root, 0)
    return found


def lamella_directory_for(project_root: Path, stage_directory: Path) -> Path:
    """Return the lamella directory a stage belongs to.

    The lamella is the ancestor of the stage that is a direct child of the
    project root. When the stage *is* the project root there is no lamella
    level on disk, and the project root stands in for it.
    """
    if stage_directory == project_root:
        return project_root
    current = stage_directory
    while current.parent != project_root:
        if current.parent == current:  # pragma: no cover - defensive
            return stage_directory
        current = current.parent
    return current


def discover_stages(
    project_root: Path, max_depth: int = DEFAULT_MAX_DEPTH
) -> list[DiscoveredStage]:
    """Return every stage in a project, with the lamella each belongs to.

    Stages are returned in directory order. Callers that need them in
    acquisition order sort by the timestamp in the folder name.
    """
    stages = [
        DiscoveredStage(
            path=stage_path,
            lamella_path=(lamella_path := lamella_directory_for(project_root, stage_path)),
            lamella_is_synthetic=lamella_path == project_root,
            stage_is_synthetic=stage_path == lamella_path,
        )
        for stage_path in find_stage_directories(project_root, max_depth=max_depth)
    ]
    logger.debug("Discovered %d stage(s) in %s", len(stages), project_root)
    return stages


def group_stages_by_lamella(
    stages: list[DiscoveredStage],
) -> dict[Path, list[DiscoveredStage]]:
    """Group discovered stages by their lamella, preserving discovery order."""
    grouped: dict[Path, list[DiscoveredStage]] = {}
    for stage in stages:
        grouped.setdefault(stage.lamella_path, []).append(stage)
    return grouped
