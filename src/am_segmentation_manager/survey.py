"""Read-only inspection of a dataset collection.

The survey reports what the layout conventions in :mod:`am_segmentation_manager.layout`
make of a directory tree: which projects were found, how their hierarchy was
interpreted, which image sets exist, where each one's labels resolve to, and
what does not fit. It never modifies anything.

It exists for three jobs:

* surveying older datasets before their layouts are supported, which is why it
  discovers projects independently of :class:`~am_segmentation_manager.scanner.Scanner`
  -- it must be able to describe layouts the scanner cannot yet index;
* verifying bulk filesystem work, such as reorganising labels back into project
  directories, by reporting images without labels and labels without images;
* debugging, by making the conventions' interpretation of a directory visible.

Discovery here is deliberately tolerant: an unreadable directory or an
unrecognised name produces a finding, never an exception, because a survey of
twenty projects is useless if one bad directory aborts it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

from .discovery import (
    DEFAULT_MAX_DEPTH,
    PROJECT_MARKERS,
    find_nested_project_roots,
    find_project_roots,
    find_stage_directories,
    group_stages_by_lamella,
    discover_stages,
    project_markers,
)
from .io.path import make_path_relative
from .io.yaml import read_yaml
from .layout import (
    ImageSet,
    build_image_sets,
    directory_role,
    image_files_by_name,
    shared_labels_directory_name,
    state_of_path,
    subdirectories_by_name,
)
from .operations.label_io import is_label_file_empty
from .metadata import (
    AM_METADATA_FILENAME,
    EXPERIMENT_FILENAME,
    ExperimentMetadata,
    build_experiment_metadata,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_DEPTH",
    "PROJECT_MARKERS",
    "CollectionSurvey",
    "Finding",
    "ImageSetSurvey",
    "LamellaSurvey",
    "ProjectSurvey",
    "StageSurvey",
    "find_project_roots",
    "find_stage_directories",
    "survey_collection",
    "survey_project",
]

#: Stage directory names that the current workflow expects. A stage named
#: anything else is still indexed, but reported: the "a stage is a directory
#: holding image directories" rule selects exactly the adaptive-polish stages
#: today by coincidence rather than by design, and a layout that breaks the
#: coincidence should be visible immediately.
EXPECTED_STAGE_PREFIX: str = "adaptive_polish"

Severity = Literal["error", "warning", "info"]

#: Ordering for display and for counting, most severe first.
SEVERITY_ORDER: tuple[Severity, ...] = ("error", "warning", "info")


@dataclass(frozen=True)
class Finding:
    """Something worth reporting about a surveyed directory.

    Attributes:
        severity: ``"error"`` for something that will index incorrectly,
            ``"warning"`` for something that needs attention or migration,
            ``"info"`` for an interpretation worth confirming.
        category: A short stable slug, so findings can be filtered or counted.
        path: The directory or file the finding is about.
        message: A single sentence describing what was found.
    """

    severity: Severity
    category: str
    path: Path
    message: str


@dataclass(frozen=True)
class ImageSetSurvey:
    """What was found in one image set.

    Attributes:
        image_set: The resolved image set, including where its labels live.
        image_count: Images found, across all workflow states.
        label_count: Labels found, across all workflow states.
        images_without_labels: Images with no label of the same filename.
        labels_without_images: Labels with no image of the same filename.
        empty_labels: Labels containing nothing but background, which are not
            annotations. None when labels were not inspected.
    """

    image_set: ImageSet
    image_count: int
    label_count: int
    images_without_labels: int
    labels_without_images: int
    empty_labels: int | None = None

    @property
    def annotated_labels(self) -> int | None:
        """Return labels that actually hold an annotation, if known."""
        if self.empty_labels is None:
            return None
        return self.label_count - self.empty_labels


@dataclass(frozen=True)
class StageSurvey:
    """A stage and the image sets inside it.

    Attributes:
        path: The stage directory.
        synthetic: Whether the stage level is absent on disk, so the lamella
            directory stands in for it.
        image_sets: The image sets found, in display order.
    """

    path: Path
    synthetic: bool
    image_sets: tuple[ImageSetSurvey, ...]

    @property
    def name(self) -> str:
        """Return the stage directory's name."""
        return self.path.name


@dataclass(frozen=True)
class LamellaSurvey:
    """A lamella and its stages.

    Attributes:
        path: The lamella directory.
        synthetic: Whether the lamella level is absent on disk, so the project
            directory stands in for it.
        stages: The stages found, in directory order.
    """

    path: Path
    synthetic: bool
    stages: tuple[StageSurvey, ...]

    @property
    def name(self) -> str:
        """Return the lamella directory's name."""
        return self.path.name


@dataclass(frozen=True)
class ProjectSurvey:
    """A project, its interpreted hierarchy, and its findings.

    Attributes:
        path: The project root.
        relative_path: The project root relative to the scan root, which is the
            only reliable identity when two projects share a name.
        markers: The marker filenames found in the project root.
        method: Autolamella's ``method`` value, when present. Absent from the
            2026 schema.
        metadata: The resolved descriptive metadata.
        lamellae: The lamellae found, in directory order.
        findings: Everything reported about this project.
    """

    path: Path
    relative_path: str
    markers: tuple[str, ...]
    method: str | None
    metadata: ExperimentMetadata
    lamellae: tuple[LamellaSurvey, ...]
    findings: tuple[Finding, ...] = ()

    @property
    def name(self) -> str:
        """Return the project directory's name."""
        return self.path.name

    @property
    def stage_count(self) -> int:
        """Return the number of stages across all lamellae."""
        return sum(len(lamella.stages) for lamella in self.lamellae)

    @property
    def image_sets(self) -> Iterator[ImageSetSurvey]:
        """Yield every image set in the project."""
        for lamella in self.lamellae:
            for stage in lamella.stages:
                yield from stage.image_sets

    @property
    def image_count(self) -> int:
        """Return the number of images across all image sets."""
        return sum(entry.image_count for entry in self.image_sets)

    @property
    def label_count(self) -> int:
        """Return the number of labels across all image sets."""
        return sum(entry.label_count for entry in self.image_sets)


@dataclass(frozen=True)
class CollectionSurvey:
    """The result of surveying a directory tree.

    Attributes:
        root: The directory that was scanned.
        projects: The projects found, ordered by relative path.
        findings: Findings that concern the collection rather than one project,
            such as two projects sharing an identity.
    """

    root: Path
    projects: tuple[ProjectSurvey, ...] = ()
    findings: tuple[Finding, ...] = ()

    @property
    def all_findings(self) -> list[Finding]:
        """Return collection findings followed by every project's findings."""
        found = list(self.findings)
        for project in self.projects:
            found.extend(project.findings)
        return found

    def finding_counts(self) -> dict[Severity, int]:
        """Return the number of findings of each severity."""
        counts: dict[Severity, int] = {severity: 0 for severity in SEVERITY_ORDER}
        for finding in self.all_findings:
            counts[finding.severity] += 1
        return counts


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Survey construction
# ---------------------------------------------------------------------------

def survey_collection(
    root: Path, max_depth: int = DEFAULT_MAX_DEPTH, check_labels: bool = True
) -> CollectionSurvey:
    """Survey every project at or below *root*.

    Args:
        root: The directory to scan.
        max_depth: How deep to search for projects and for stages.
        check_labels: Read each label to tell an annotation from an empty
            placeholder. Costs roughly 2.5 ms a label; turn it off for a
            structure-only survey of a very large collection.

    Returns:
        A :class:`CollectionSurvey` describing what was found, including
        findings for anything that does not fit the conventions.
    """
    roots = find_project_roots(root, max_depth=max_depth)
    projects = tuple(
        sorted(
            (
                survey_project(path, root, max_depth=max_depth, check_labels=check_labels)
                for path in roots
            ),
            key=lambda project: project.relative_path,
        )
    )
    findings = tuple(_collection_findings(projects))
    logger.info("Surveyed %d project(s) under %s", len(projects), root)
    return CollectionSurvey(root=root, projects=projects, findings=findings)


def survey_project(
    project_root: Path,
    scan_root: Path | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    check_labels: bool = True,
) -> ProjectSurvey:
    """Survey one project.

    Args:
        project_root: The project directory.
        scan_root: The directory paths should be reported relative to. Defaults
            to *project_root*'s parent.
        max_depth: How deep to search for stages.
        check_labels: Read each label to tell an annotation from an empty
            placeholder.
    """
    base = scan_root if scan_root is not None else project_root.parent
    grouped = group_stages_by_lamella(discover_stages(project_root, max_depth=max_depth))

    findings: list[Finding] = []
    lamellae: list[LamellaSurvey] = []
    for lamella_path in sorted(grouped):
        discovered = grouped[lamella_path]
        stages = tuple(
            _survey_stage(item.path, item.stage_is_synthetic, findings, check_labels)
            for item in sorted(discovered, key=lambda item: item.path)
        )
        lamellae.append(
            LamellaSurvey(
                path=lamella_path,
                synthetic=discovered[0].lamella_is_synthetic,
                stages=stages,
            )
        )

    if not lamellae:
        findings.append(
            Finding(
                severity="warning",
                category="no-images",
                path=project_root,
                message="No image directories found anywhere in this project",
            )
        )

    for nested in find_nested_project_roots(project_root, max_depth=max_depth):
        findings.append(
            Finding(
                severity="error",
                category="nested-project",
                path=nested,
                message=(
                    f"A project marker inside {make_path_relative(project_root, base)} "
                    f"splits it: these stages are indexed as a separate project. Remove "
                    f"the marker if this is one project, or move the directory out if "
                    f"it is two"
                ),
            )
        )

    experiment = read_yaml(project_root / EXPERIMENT_FILENAME)
    am_metadata = read_yaml(project_root / AM_METADATA_FILENAME)
    metadata = build_experiment_metadata(
        experiment,
        am_metadata,
        experiment_path=project_root / EXPERIMENT_FILENAME,
        am_metadata_path=project_root / AM_METADATA_FILENAME,
    )
    findings.extend(_metadata_findings(project_root, metadata))

    return ProjectSurvey(
        path=project_root,
        relative_path=make_path_relative(project_root, base),
        markers=project_markers(project_root),
        method=_clean_method(experiment.get("method")),
        metadata=metadata,
        lamellae=tuple(lamellae),
        findings=tuple(findings),
    )


def _state_findings(
    image_set: ImageSet, images: dict[str, Path], labels: dict[str, Path]
) -> list[Finding]:
    """Report images whose workflow state disagrees with their label's.

    The label is authoritative, so a disagreement means files were moved
    outside this package or an operation was interrupted. ``repair-state``
    resolves it by moving the images to follow the label.
    """
    disagreeing = [
        name for name, image_path in images.items()
        if name in labels and state_of_path(image_path) != state_of_path(labels[name])
    ]
    if not disagreeing:
        return []
    return [
        Finding(
            severity="warning",
            category="state-mismatch",
            path=image_set.directory,
            message=(
                f"{len(disagreeing)} image(s) are in a different workflow state from "
                f"their label; run 'am-seg repair-state' to move them to match"
            ),
        )
    ]


def _clean_method(value: object) -> str | None:
    """Return Autolamella's ``method`` value as a string, if it has one."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata_findings(project_root: Path, metadata: ExperimentMetadata) -> list[Finding]:
    """Report descriptive metadata that is missing or still a placeholder.

    Metadata is filled in progressively, so this is a checklist rather than an
    error: a project with none is perfectly indexable.
    """
    if metadata.todo_fields:
        return [
            Finding(
                severity="info",
                category="metadata-todo",
                path=project_root / AM_METADATA_FILENAME,
                message=(
                    f"{len(metadata.todo_fields)} field(s) still to fill in: "
                    f"{', '.join(metadata.todo_fields)}"
                ),
            )
        ]
    missing = metadata.missing_fields()
    if missing:
        return [
            Finding(
                severity="info",
                category="metadata-missing",
                path=project_root,
                message=(
                    f"{len(missing)} descriptive field(s) not recorded "
                    f"({', '.join(missing)}); run 'am-seg init-project' to add a template"
                ),
            )
        ]
    return []


def _survey_stage(
    stage_path: Path, synthetic: bool, findings: list[Finding], check_labels: bool = True
) -> StageSurvey:
    """Survey one stage, appending any findings to *findings*."""
    if not stage_path.name.lower().startswith(EXPECTED_STAGE_PREFIX):
        findings.append(
            Finding(
                severity="warning",
                category="unexpected-stage-name",
                path=stage_path,
                message=(
                    f"Stage directory is not named {EXPECTED_STAGE_PREFIX}*; confirm "
                    f"these are Adaptive Polishing images"
                ),
            )
        )

    for child in subdirectories_by_name(stage_path).values():
        if directory_role(child.name) == "unknown":
            findings.append(
                Finding(
                    severity="info",
                    category="unrecognised-directory",
                    path=child,
                    message="Directory does not match any known role and was ignored",
                )
            )

    image_sets = tuple(
        _survey_image_set(image_set, findings, check_labels)
        for image_set in build_image_sets(stage_path)
    )
    return StageSurvey(
        path=stage_path,
        synthetic=synthetic,
        image_sets=image_sets,
    )


def _survey_image_set(
    image_set: ImageSet, findings: list[Finding], check_labels: bool = True
) -> ImageSetSurvey:
    """Survey one image set, appending any findings to *findings*."""
    images = image_files_by_name(image_set.directory)
    labels = image_files_by_name(image_set.labels_directory) if image_set.labels_exist else {}

    unlabelled = sorted(set(images) - set(labels))
    orphaned = sorted(set(labels) - set(images))

    if not images:
        findings.append(
            Finding(
                severity="warning",
                category="empty-image-set",
                path=image_set.directory,
                message="Image directory contains no images",
            )
        )
    elif not image_set.labels_exist:
        findings.append(
            Finding(
                severity="info",
                category="no-labels-yet",
                path=image_set.directory,
                message=(
                    f"No labels yet; they would be written to "
                    f"{image_set.labels_directory.name}/"
                ),
            )
        )
    elif unlabelled:
        findings.append(
            Finding(
                severity="warning",
                category="images-without-labels",
                path=image_set.directory,
                message=f"{len(unlabelled)} of {len(images)} images have no label",
            )
        )

    if orphaned:
        findings.append(
            Finding(
                severity="warning",
                category="labels-without-images",
                path=image_set.labels_directory,
                message=(
                    f"{len(orphaned)} label(s) have no matching image in "
                    f"{image_set.directory.name}/"
                ),
            )
        )

    if image_set.labels_are_legacy:
        findings.append(
            Finding(
                severity="warning",
                category="legacy-label-directory",
                path=image_set.labels_directory,
                message=(
                    f"Pre-convention label directory name; the current convention is "
                    f"{shared_labels_directory_name(image_set.modality)}/"
                ),
            )
        )

    if image_set.labels_are_shared:
        findings.append(
            Finding(
                severity="info",
                category="shared-labels",
                path=image_set.labels_directory,
                message=(
                    "Labels are shared with other image sets in this stage; a state "
                    "change to one will apply to all of them"
                ),
            )
        )

    findings.extend(_state_findings(image_set, images, labels))

    empty_labels = _count_empty_labels(labels) if check_labels else None
    if empty_labels:
        findings.append(
            Finding(
                severity="warning",
                category="empty-labels",
                path=image_set.labels_directory,
                message=(
                    f"{empty_labels} of {len(labels)} label(s) contain only background "
                    f"and are not annotations; the segmentation software writes one "
                    f"for every image it is pointed at"
                ),
            )
        )

    return ImageSetSurvey(
        image_set=image_set,
        image_count=len(images),
        label_count=len(labels),
        images_without_labels=len(unlabelled),
        labels_without_images=len(orphaned),
        empty_labels=empty_labels,
    )


def _count_empty_labels(labels: dict[str, Path]) -> int:
    """Return how many of *labels* hold nothing but background.

    Labels that cannot be inspected -- an unreadable file, or a missing image
    stack -- are not counted as empty: reporting "not annotated" for a label
    nobody could read would be a guess presented as a fact.
    """
    return sum(1 for path in labels.values() if is_label_file_empty(path) is True)


def _collection_findings(projects: Iterable[ProjectSurvey]) -> list[Finding]:
    """Return findings that concern the collection rather than one project."""
    findings: list[Finding] = []
    by_name: dict[str, list[ProjectSurvey]] = {}
    for project in projects:
        by_name.setdefault(project.name, []).append(project)

    for name, duplicates in sorted(by_name.items()):
        if len(duplicates) < 2:
            continue
        locations = ", ".join(project.relative_path for project in duplicates)
        for project in duplicates:
            findings.append(
                Finding(
                    severity="error",
                    category="duplicate-project-name",
                    path=project.path,
                    message=(
                        f"{len(duplicates)} projects are named {name!r} ({locations}); "
                        f"they must be identified by path, not by name"
                    ),
                )
            )
    return findings
