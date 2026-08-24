"""Locating projects, lamellae and stages on disk.

Answers "where are the things" using the conventions in
:mod:`am_segmentation_manager.layout`, for the two consumers that traverse a
dataset tree -- the scanner and the survey -- which must agree about what counts
as a project and where its stages are.

A stage holds images either as directories or loose among its files, and only
the first kind ends the search: see :func:`find_stage_directories`, whose
invariant the layout rests on. Older datasets collapse levels; the missing ones
are *synthesised*, the parent standing in, which :class:`DiscoveredStage`
records. An unreadable directory yields nothing rather than raising.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .layout import (
    build_image_sets,
    classify_loose_image,
    directory_role,
    image_files,
    is_alignment_directory,
    subdirectories_by_name,
)

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
        in_alignment_run: True when the stage is one alignment run, whose
            images are named too generically to classify without knowing that.
    """

    path: Path
    lamella_path: Path
    lamella_is_synthetic: bool
    stage_is_synthetic: bool
    in_alignment_run: bool = False


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
    """Return whether *directory* holds Adaptive Polishing images.

    Stricter than "has image directories", which is one of the tests
    :func:`find_stage_directories` uses. Inside a project an empty ``sem/`` is a
    real, if empty, image set worth reporting; outside one, a directory merely
    *named* like an image directory proves nothing.

    Deliberately blind to loose reference and alignment images. Its callers ask
    where a *project* is and which unmarked directories hold data worth marking,
    and both questions are about the images this package was built to curate: a
    directory holding only alignment images is not a dataset anyone wants
    proposed as a new project.
    """
    return any(image_files(image_set.directory) for image_set in build_image_sets(directory))


def holds_loose_images(directory: Path, in_alignment_run: bool = False) -> bool:
    """Return whether *directory* holds images loose among its other files.

    Cheaper than building the sets, and that is the point: discovery only needs
    to know whether there is anything here, so it asks the filenames and
    **never opens an image to find out**. Establishing the beam of an image
    whose name does not state it is the scanner's job, once, for a directory
    already known to be a stage.

    Args:
        directory: The directory to look in.
        in_alignment_run: Whether *directory* is one alignment run, whose
            fixed filenames mean nothing without that context.
    """
    return any(
        classify_loose_image(path.name, in_alignment_run=in_alignment_run) is not None
        for path in image_files(directory)
    )


def find_dataset_candidates(
    root: Path, max_depth: int = DEFAULT_MAX_DEPTH
) -> list[Path]:
    """Return directories that hold images but are not yet projects.

    A directory qualifies when it has no marker file of its own, holds image
    directories somewhere beneath it, and contains **no marked project** in its
    subtree. That last condition separates a dataset from a collection
    directory, which also holds images beneath it but holds them inside real
    projects. A directory that already has a marker yields nothing.

    Matching stops at the **outermost** qualifying directory. Descending
    further would propose a marker inside each *lamella*, splitting one dataset
    into several projects.

    **Known limitation.** Which level is "the project" is not decidable from
    structure alone. Stopping at the outermost qualifying directory is right
    wherever a collection holds at least one marked project to anchor the
    level. A directory holding nothing but a single unmarked dataset is the
    ambiguous case, and this proposes that directory rather than the dataset
    inside it; ``--dry-run`` shows the proposal before anything is written.
    """
    if max_depth < 0:
        return []
    if project_markers(root):
        return []
    if not contains_marked_project(root, max_depth) and _contains_images(root, max_depth):
        return [root]

    candidates: list[Path] = []
    for child in subdirectories_by_name(root).values():
        candidates.extend(find_dataset_candidates(child, max_depth - 1))
    return candidates


def contains_marked_project(directory: Path, max_depth: int = DEFAULT_MAX_DEPTH) -> bool:
    """Return whether any project marker exists at or below *directory*.

    What separates a collection directory from a project: a collection holds
    projects, and writing a marker into one would make every unmarked dataset
    beneath it a *lamella* of a single project spanning several samples.
    """
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

    A stage is a directory holding images, in either of the two shapes they
    come in: image *directories* such as ``sem/``, or images sitting loose
    among a directory's other files. The project root itself is a candidate,
    which is what supports the oldest layouts where neither the lamella nor the
    stage level exists as a folder.

    **A loose-image stage does not stop the search, and a directory-shaped one
    does.** This is the invariant the whole layout rests on. A 2026 lamella
    holds reference images loose and its milling tasks below, and each task
    holds reference images loose and its Adaptive Polishing stages below; were
    a lamella full of reference images to end the search the way ``sem/`` does,
    every Adaptive Polishing stage beneath it would vanish from the index.
    Nothing sits below an image directory, so stopping there stays right.

    A directory holding both shapes is returned once and not descended into.
    That loses nothing: the scanner builds both kinds of set for every stage it
    is given, so which branch found the directory does not change what is
    indexed in it.

    The search stops at a **nested project root**: those stages belong to that
    project, not this one. It also does not descend into image, label,
    prediction, plot or state directories, which bounds the cost on trees
    holding thousands of files.
    """
    found: list[Path] = []

    def walk(directory: Path, depth: int, in_alignment_run: bool) -> None:
        if directory != project_root and project_markers(directory):
            logger.debug("Not crossing into nested project %s", directory)
            return
        if build_image_sets(directory):
            found.append(directory)
            return
        if holds_loose_images(directory, in_alignment_run=in_alignment_run):
            found.append(directory)
        if depth >= max_depth:
            logger.debug("Stopping at depth %d: %s", depth, directory)
            return
        for child in subdirectories_by_name(directory).values():
            if directory_role(child.name) == "unknown":
                walk(child, depth + 1, is_alignment_directory(directory.name))

    walk(project_root, 0, False)
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
            in_alignment_run=is_alignment_directory(stage_path.parent.name),
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
