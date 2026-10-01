"""Read-only inspection of a dataset collection.

Reports what the conventions in :mod:`am_segmentation_manager.layout` make of a
directory tree: projects found, how their hierarchy was interpreted, which image
sets exist, where labels resolve to, and what does not fit. It never modifies
anything.

Discovery is independent of :class:`~am_segmentation_manager.scanner.Scanner`, so
it can describe layouts the scanner cannot yet index. An unreadable directory or
unrecognised name produces a finding, never an exception.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .discovery import (
    DEFAULT_MAX_DEPTH,
    PROJECT_MARKERS,
    DatasetCandidate,
    describe_dataset_candidates,
    discover_stages,
    find_nested_project_roots,
    find_project_roots,
    find_stage_directories,
    group_stages_by_lamella,
    project_markers,
)
from .families import FamiliesFileError, families_file_path, read_families
from .image_metadata import (
    ImageMetadata,
    condition_disagreements,
    header_reader,
    read_condition_headers,
)
from .io.long_paths import WINDOWS_MAX_PATH, explanation, long_paths_work, unreachable
from .io.path import make_path_relative
from .io.yaml import read_yaml_or_problem
from .layout import (
    IMAGE_EXTENSIONS,
    LABELS_EXPECTED,
    LABELS_SUFFIX,
    NO_CORRECTIONS,
    PREDICTIONS_SUFFIX,
    SERIES,
    FamilyCorrections,
    ImageSet,
    Series,
    build_image_sets,
    build_loose_image_sets,
    conflicting_label_directories,
    conflicting_predictions_directories,
    directory_cache,
    directory_role,
    families_claiming_one_directory,
    image_files,
    image_files_by_name,
    image_files_of,
    is_ignored_image_name,
    shared_labels_directory_name,
    state_directories,
    subdirectories_by_name,
)
from .metadata import (
    AM_METADATA_FILENAME,
    EXPERIMENT_FILENAME,
    ExperimentMetadata,
    build_experiment_metadata,
)
from .operations.archive import projected_archive_path
from .operations.label_io import is_label_file_empty
from .state import (
    AM_STATE_FILENAME,
    StateFileError,
    read_state,
    state_file_path,
    state_key,
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
        image_names: The filenames found, which are what a group is keyed by.
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
    image_names: tuple[str, ...] = ()

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

    @property
    def images_by_series(self) -> dict[Series, int]:
        """Return how many images each series holds, in declared order.

        Only series actually present, so a project of Adaptive Polishing
        images alone reads no differently than it used to.
        """
        counts = {series: 0 for series in SERIES}
        for entry in self.image_sets:
            counts[entry.image_set.series] += entry.image_count
        return {series: count for series, count in counts.items() if count}


@dataclass(frozen=True)
class CollectionSurvey:
    """The result of surveying a directory tree.

    Attributes:
        root: The directory that was scanned.
        projects: The projects found, ordered by relative path.
        findings: Findings that concern the collection rather than one project,
            such as two projects sharing an identity.
        unindexed: Directories holding images that are not projects yet, and so
            appear nowhere in *projects*. Reported because a survey that
            counted only what is indexed would be silent about the data most
            in need of attention.
    """

    root: Path
    projects: tuple[ProjectSurvey, ...] = ()
    findings: tuple[Finding, ...] = ()
    unindexed: tuple[DatasetCandidate, ...] = ()

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
    with directory_cache():
        roots = find_project_roots(root, max_depth=max_depth)
        projects = tuple(
            sorted(
                (
                    survey_project(
                        path, root, max_depth=max_depth, check_labels=check_labels
                    )
                    for path in roots
                ),
                key=lambda project: project.relative_path,
            )
        )
        unindexed = tuple(describe_dataset_candidates(root, max_depth=max_depth))
        stray = _stray_images(root, roots, unindexed, max_depth=max_depth)
    findings = tuple(
        [
            *_collection_findings(projects),
            *_unindexed_dataset_findings(unindexed),
            *_stray_image_findings(root, stray),
            *_long_path_findings(root, projects),
        ]
    )
    logger.info(
        "Surveyed %d project(s) and %d unindexed dataset(s) under %s",
        len(projects), len(unindexed), root,
    )
    return CollectionSurvey(
        root=root, projects=projects, findings=findings, unindexed=unindexed
    )


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
    discovered_stages = discover_stages(project_root, max_depth=max_depth)
    grouped = group_stages_by_lamella(discovered_stages)
    stage_paths = frozenset(item.path for item in discovered_stages)

    findings: list[Finding] = []
    corrections = _families_or_finding(project_root, findings)
    lamellae: list[LamellaSurvey] = []
    for lamella_path in sorted(grouped):
        discovered = grouped[lamella_path]
        stages = tuple(
            _survey_stage(
                item.path,
                item.stage_is_synthetic,
                findings,
                check_labels,
                in_alignment_run=item.in_alignment_run,
                stage_paths=stage_paths,
                corrections=corrections,
            )
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

    experiment_path = project_root / EXPERIMENT_FILENAME
    am_metadata_path = project_root / AM_METADATA_FILENAME
    experiment, experiment_problem = read_yaml_or_problem(experiment_path)
    am_metadata, am_metadata_problem = read_yaml_or_problem(am_metadata_path)
    metadata = build_experiment_metadata(
        experiment,
        am_metadata,
        experiment_path=experiment_path,
        am_metadata_path=am_metadata_path,
    )
    unreadable = [
        (path, problem)
        for path, problem in (
            (experiment_path, experiment_problem), (am_metadata_path, am_metadata_problem)
        )
        if problem is not None
    ]
    findings.extend(_unreadable_yaml_findings(unreadable))
    if not unreadable:
        findings.extend(_metadata_findings(project_root, metadata))

    survey = ProjectSurvey(
        path=project_root,
        relative_path=make_path_relative(project_root, base),
        markers=project_markers(project_root),
        method=_clean_method(experiment.get("method")),
        metadata=metadata,
        lamellae=tuple(lamellae),
        findings=(),
    )
    findings.extend(_unindexed_findings(project_root, survey))

    return ProjectSurvey(
        path=project_root,
        relative_path=make_path_relative(project_root, base),
        markers=project_markers(project_root),
        method=_clean_method(experiment.get("method")),
        metadata=metadata,
        lamellae=tuple(lamellae),
        findings=tuple(findings),
    )


def _families_or_finding(project_root: Path, findings: list[Finding]) -> FamilyCorrections:
    """Return a project's family corrections, reporting a file that will not read.

    Surveyed by directory names alone in that case, as the scan indexes it.
    """
    try:
        return read_families(project_root)
    except FamiliesFileError as error:
        findings.append(
            Finding(
                severity="error",
                category="unreadable-families-file",
                path=families_file_path(project_root),
                message=(
                    f"{error}. The project is indexed by its directory names alone, "
                    f"and nothing is written to it until the file is fixed"
                ),
            )
        )
        return NO_CORRECTIONS


def _unreadable_yaml_findings(unreadable: Iterable[tuple[Path, str]]) -> list[Finding]:
    """Report each project file that is there but would not read.

    The scan indexes the project without that file, so what the metadata lacks
    cannot be told from what was never filled in, and no checklist is reported.
    """
    return [
        Finding(
            severity="error",
            category="unreadable-yaml",
            path=path,
            message=(
                f"{problem}. The project is indexed without this file's metadata, and "
                f"the metadata exports refuse it until the file is fixed"
            ),
        )
        for path, problem in unreadable
    ]


def _unindexed_findings(project_root: Path, survey: ProjectSurvey) -> list[Finding]:
    """Report image files in the project that no image set claims.

    **The last-resort check for a layout nobody anticipated.** Every other
    finding describes something the conventions recognised; this one counts
    what they did not reach at all, which is the failure that hides best: a
    project can be surveyed as holding a fraction of its images with nothing
    saying so.

    Since the ``other`` series, an image no naming rule recognises is indexed
    rather than missed, and ``unrecognised-image`` reports that population per
    stage. So this is normally empty, and what is left in it is an image the
    *traversal* never offered to a set at all -- one below the stage search's
    depth limit, or in a directory whose role stops the search -- which no
    classification rule can fix and which nothing else would say.

    Labels and predictions are excluded by counting only what is not inside a
    label or prediction directory, so a project is not reported as holding
    thousands of unindexed images the moment it is annotated. Images this
    package deliberately omits are excluded too: not indexed and not reported
    is the whole point of :func:`is_ignored_image_name`.
    """
    claimed = {
        entry.image_set.directory / name
        for entry in survey.image_sets
        for name in entry.image_names
    }
    unindexed = sorted(
        path
        for path in project_root.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
        and path not in claimed
        and not is_ignored_image_name(path.name)
        and not any(
            directory_role(part)
            in {"labels", "predictions", "plots", "state", "archive"}
            for part in path.parts
        )
        and path.is_file()
    )
    if not unindexed:
        return []
    shown = ", ".join(make_path_relative(path, project_root) for path in unindexed[:3])
    more = "" if len(unindexed) <= 3 else f", and {len(unindexed) - 3} more"
    return [
        Finding(
            severity="warning",
            category="unindexed-images",
            path=project_root,
            message=(
                f"{len(unindexed)} image file(s) were never offered to an image set "
                f"and are not indexed: {shown}{more}"
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
    stage_path: Path,
    synthetic: bool,
    findings: list[Finding],
    check_labels: bool = True,
    in_alignment_run: bool = False,
    stage_paths: frozenset[Path] = frozenset(),
    *,
    corrections: FamilyCorrections,
) -> StageSurvey:
    """Survey one stage, appending any findings to *findings*.

    Args:
        stage_path: The stage directory.
        synthetic: Whether the lamella directory is standing in for a stage.
        findings: Appended to.
        check_labels: Read each label to tell an annotation from a placeholder.
        in_alignment_run: Whether the stage is one alignment run, whose
            filenames need that context to classify.
        stage_paths: Every stage in the project, so a subdirectory holding one
            is not reported as ignored -- it is descended into, not ignored.
        corrections: The project's family corrections.
    """
    directory_sets = build_image_sets(stage_path, corrections=corrections)
    # Only the beam is wanted here. The survey reports what is on disk rather
    # than ordering it, so it opens a header only for a name that states no
    # beam, where the scanner opens one for every loose image.
    reader = header_reader()
    loose = build_loose_image_sets(
        stage_path,
        in_alignment_run=in_alignment_run,
        resolve_modality=(
            (lambda path: header.modality if (header := reader(path)) is not None else None)
            if reader is not None
            else None
        ),
    )

    # The rule this checks is "a directory holding image directories is a
    # stage", which selects the adaptive-polish stages. A stage found by
    # holding loose images is a lamella, milling task or alignment run, and
    # was never expected to carry that name.
    if directory_sets and not stage_path.name.lower().startswith(EXPECTED_STAGE_PREFIX):
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

    findings.extend(_loose_findings(stage_path, loose))
    findings.extend(_family_conflict_findings(stage_path, directory_sets, corrections))
    if reader is not None:
        findings.extend(_family_header_findings(stage_path, directory_sets, reader))

    for child in subdirectories_by_name(stage_path).values():
        if directory_role(child.name) != "unknown":
            continue
        if any(stage == child or child in stage.parents for stage in stage_paths):
            continue
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
        for image_set in [*directory_sets, *loose.sets]
    )
    findings.extend(_declared_state_findings(stage_path, image_sets))
    return StageSurvey(
        path=stage_path,
        synthetic=synthetic,
        image_sets=image_sets,
    )


def _family_conflict_findings(
    stage_path: Path, image_sets: Iterable[ImageSet], corrections: FamilyCorrections
) -> list[Finding]:
    """Report each family whose conditions have more than one label or
    predictions directory between them, and each directory two families read.

    Such a family's conditions each keep reading their own, so they do not share
    a label -- see :func:`~am_segmentation_manager.layout.resolve_labels_directory`.
    Two families reading one directory share every label in it, which a hand
    edit of ``AM_families.yaml`` can do and ``edit-families`` refuses.
    """
    siblings = subdirectories_by_name(stage_path)
    findings: list[Finding] = [
        Finding(
            severity="error",
            category="family-shared-label-directory",
            path=stage_path,
            message=(
                f"Families {' and '.join(families)} both read {stem}{LABELS_SUFFIX}/ "
                f"and {stem}{PREDICTIONS_SUFFIX}/, so their images share labels; "
                f"correct AM_families.yaml"
            ),
        )
        for stem, families in families_claiming_one_directory(
            siblings.values(), corrections=corrections
        )
    ]
    reported: set[tuple[str, str, str]] = set()
    for image_set in image_sets:
        if image_set.is_loose or image_set.modality is None or image_set.family is None:
            continue
        for kind, conflicting in (
            ("label", conflicting_label_directories),
            ("predictions", conflicting_predictions_directories),
        ):
            key = (image_set.modality, image_set.family.lower(), kind)
            if key in reported:
                continue
            competing = conflicting(
                image_set.directory, image_set.modality, siblings, corrections=corrections
            )
            if not competing:
                continue
            reported.add(key)
            names = ", ".join(f"{directory.name}/" for directory in competing)
            findings.append(
                Finding(
                    severity="warning",
                    category=f"family-{kind}-conflict",
                    path=stage_path,
                    message=(
                        f"Family {image_set.family} has {len(competing)} {kind} "
                        f"directories ({names}); its conditions each read their own "
                        f"until these are merged into one"
                    ),
                )
            )
    return findings


def _family_header_findings(
    stage_path: Path,
    image_sets: Iterable[ImageSet],
    read: Callable[[Path], ImageMetadata | None],
) -> list[Finding]:
    """Report each family whose conditions' headers disagree about one acquisition.

    Every two conditions sharing a filename are compared on one, as
    :func:`~.image_metadata.read_condition_headers` chooses. A different
    image size is an error, since one label cannot fit both; a different pixel
    size or voltage is a warning that the conditions may not be one family.
    """
    families: dict[tuple[str, str], list[ImageSet]] = {}
    for image_set in image_sets:
        if image_set.is_loose or image_set.modality is None or image_set.family is None:
            continue
        families.setdefault((image_set.modality, image_set.family.lower()), []).append(image_set)

    findings: list[Finding] = []
    for members in families.values():
        if len(members) < 2:
            continue
        readings = read_condition_headers(
            {member.directory.name: image_files_by_name(member.directory) for member in members},
            read,
        )
        for filename, headers in readings:
            for disagreement in condition_disagreements(headers):
                blocks = disagreement.blocks_sharing
                findings.append(
                    Finding(
                        severity="error" if blocks else "warning",
                        category=(
                            "family-image-size-mismatch" if blocks else "family-header-mismatch"
                        ),
                        path=stage_path,
                        message=(
                            f"Family {members[0].family} disagrees on {filename} about "
                            f"{disagreement.describe()}; "
                            + (
                                "one label cannot fit both, so correct the family"
                                if blocks
                                else "check these conditions belong in one family"
                            )
                        ),
                    )
                )
    return findings


def _loose_findings(stage_path: Path, loose) -> list[Finding]:
    """Report the loose files that could not be made into an image set.

    Two different problems, and both files are indexed as ``other`` -- what is
    reported here is the reason, which is the part ``other`` cannot carry. A
    name no rule recognises may be an Autolamella version writing something
    new: **this is the tripwire** that made an unanticipated layout visible
    before the series existed, and the count staying non-zero is what keeps it
    one. A beam nothing could establish means the filename says none and the
    header would not read; the series its name does state is known and is worth
    saying, because a set of that series is keyed by beam and so could not take
    it.
    """
    findings: list[Finding] = []
    if loose.unrecognised:
        shown = ", ".join(loose.unrecognised[:3])
        more = "" if len(loose.unrecognised) <= 3 else f", and {len(loose.unrecognised) - 3} more"
        findings.append(
            Finding(
                severity="info",
                category="unrecognised-image",
                path=stage_path,
                message=(
                    f"{len(loose.unrecognised)} image(s) match no naming rule and are "
                    f"indexed as the 'other' series: {shown}{more}"
                ),
            )
        )
    if loose.unresolved:
        findings.append(
            Finding(
                severity="warning",
                category="unresolved-beam",
                path=stage_path,
                message=(
                    f"{len(loose.unresolved)} image(s) state no beam in their filename "
                    f"and none could be read from their header; indexed as the 'other' "
                    f"series rather than their own, starting with {loose.unresolved[0]}"
                ),
            )
        )
    return findings


def _declared_state_findings(
    stage_path: Path, image_sets: tuple[ImageSetSurvey, ...]
) -> list[Finding]:
    """Report how a stage's declared workflow state lines up with its files.

    Three things can be out of step, and each needs a different action: a stage
    whose state is still in its folders has not been migrated; an entry naming a
    group that is not there has been left behind by a rename or a deletion; and
    a state folder still on disk after migration is now only clutter.
    """
    findings: list[Finding] = []
    nested = [
        candidate
        for image_set in image_sets
        for directory in (image_set.image_set.directory, image_set.image_set.labels_directory)
        if directory is not None
        for candidate in state_directories(directory)[1:]
        if candidate.is_dir()
    ]
    holding_files = [candidate for candidate in nested if image_files(candidate)]

    try:
        declared = read_state(stage_path)
    except StateFileError as exc:
        return [
            Finding(
                severity="error",
                category="unreadable-state",
                path=state_file_path(stage_path),
                message=f"{exc}; workflow state cannot be read for this stage",
            )
        ]

    if declared.fingerprint is None:
        if holding_files:
            findings.append(
                Finding(
                    severity="warning",
                    category="unmigrated-state",
                    path=stage_path,
                    message=(
                        f"{len(holding_files)} state folder(s) still hold files and "
                        f"there is no {AM_STATE_FILENAME}; run scripts/migrate_state.py"
                    ),
                )
            )
        return findings

    if holding_files:
        findings.append(
            Finding(
                severity="warning",
                category="nested-files",
                path=stage_path,
                message=(
                    f"{len(holding_files)} state folder(s) still hold files even though "
                    f"state is declared; run scripts/migrate_state.py --move-files"
                ),
            )
        )
    elif nested:
        findings.append(
            Finding(
                severity="info",
                category="empty-state-folders",
                path=stage_path,
                message=(
                    f"{len(nested)} empty state folder(s) left over from migration; "
                    "nothing reads them"
                ),
            )
        )

    known = {
        state_key(image_set.image_set.labels_directory / name, stage_path)
        for image_set in image_sets
        if image_set.image_set.labels_directory is not None
        for name in image_set.image_names
    }
    orphaned = sorted(set(declared.states) - known)
    if orphaned:
        findings.append(
            Finding(
                severity="warning",
                category="orphaned-state",
                path=state_file_path(stage_path),
                message=(
                    f"{len(orphaned)} entr(y/ies) name a group that is not here, "
                    f"starting with {orphaned[0]}; the curation is kept, not dropped"
                ),
            )
        )
    return findings


def _survey_image_set(
    image_set: ImageSet, findings: list[Finding], check_labels: bool = True
) -> ImageSetSurvey:
    """Survey one image set, appending any findings to *findings*."""
    images = image_files_of(image_set)
    labels_directory = image_set.labels_directory
    labels = (
        image_files_by_name(labels_directory)
        if image_set.labels_exist and labels_directory is not None
        else {}
    )

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
        # Saying this for every reference and alignment set buries the findings
        # that need acting on; see layout.LABELS_EXPECTED. A set that *does*
        # have a label directory is reported on normally whatever its series,
        # so starting to label one needs no change here.
        if image_set.series in LABELS_EXPECTED and labels_directory is not None:
            findings.append(
                Finding(
                    severity="info",
                    category="no-labels-yet",
                    path=image_set.directory,
                    message=(
                        f"No labels yet; they would be written to "
                        f"{labels_directory.name}/"
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

    if orphaned and labels_directory is not None:
        findings.append(
            Finding(
                severity="warning",
                category="labels-without-images",
                path=labels_directory,
                message=(
                    f"{len(orphaned)} label(s) have no matching image in "
                    f"{image_set.directory.name}/"
                ),
            )
        )

    if (
        image_set.labels_are_legacy
        and labels_directory is not None
        and image_set.modality is not None
    ):
        findings.append(
            Finding(
                severity="warning",
                category="legacy-label-directory",
                path=labels_directory,
                message=(
                    f"Pre-convention label directory name; the current convention is "
                    f"{shared_labels_directory_name(image_set.modality)}/"
                ),
            )
        )

    if image_set.labels_are_shared and labels_directory is not None:
        findings.append(
            Finding(
                severity="info",
                category="shared-labels",
                path=labels_directory,
                message=(
                    "Labels are shared with other image sets in this stage; a state "
                    "change to one will apply to all of them"
                ),
            )
        )


    empty_labels = _count_empty_labels(labels) if check_labels else None
    if empty_labels and labels_directory is not None:
        findings.append(
            Finding(
                severity="warning",
                category="empty-labels",
                path=labels_directory,
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
        image_names=tuple(sorted(images)),
    )


def _count_empty_labels(labels: dict[str, Path]) -> int:
    """Return how many of *labels* hold nothing but background.

    Labels that cannot be inspected -- an unreadable file, or a missing image
    stack -- are not counted as empty: reporting "not annotated" for a label
    nobody could read would be a guess presented as a fact.
    """
    return sum(1 for path in labels.values() if is_label_file_empty(path) is True)


def _unindexed_dataset_findings(
    candidates: Iterable[DatasetCandidate],
) -> list[Finding]:
    """Return one finding per directory holding images that is not a project.

    Warnings rather than info: this is data the collection does not yet
    contain, and burying it among the per-image notes is what made a folder of
    three unindexed datasets look like a folder of none.
    """
    findings: list[Finding] = []
    for candidate in candidates:
        if candidate.holds_several_datasets:
            shown = ", ".join(candidate.lamellae[:3])
            more = "" if len(candidate.lamellae) <= 3 else ", ..."
            findings.append(
                Finding(
                    severity="warning",
                    category="gathered-datasets",
                    path=candidate.path,
                    message=(
                        f"Holds {len(candidate.lamellae)} dataset(s) laid out "
                        f"{len(candidate.stage_depths)} different ways ({shown}{more}), "
                        f"so it gathers several rather than being one. Make each of "
                        f"them a project rather than this directory"
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    severity="warning",
                    category="unindexed-dataset",
                    path=candidate.path,
                    message=(
                        f"Holds {len(candidate.lamellae)} lamella(e) and "
                        f"{candidate.images} image(s) but has no project marker, so "
                        f"nothing indexes it. A metadata file makes it a project"
                    ),
                )
            )
    return findings


def _stray_images(
    root: Path,
    project_roots: Iterable[Path],
    candidates: Iterable[DatasetCandidate],
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[Path]:
    """Return image files under *root* that lie inside no project at all.

    **The failure a project-scoped check cannot see.** ``unindexed-images``
    walks one project and ``unindexed-dataset`` reports a directory that *looks
    like* a dataset; a handful of loose files beside a collection's projects is
    neither, and before this nothing counted them. The reference collection held
    215 when this was written, 204 of them under one ``log/`` directory, and no
    command would have said so.

    The walk stops at every project root and at the directories a role already
    claims, so it covers only what the rest of the survey does not: on a
    collection this is a few directory listings rather than a second pass over
    the tree. A directory already reported as a dataset candidate is skipped
    too -- it is being reported, and one file is not two findings.
    """
    inside = set(project_roots)
    reported = {candidate.path for candidate in candidates}
    found: list[Path] = []

    def walk(directory: Path, depth: int) -> None:
        if directory in inside or directory in reported:
            return
        for path in image_files(directory):
            if not is_ignored_image_name(path.name):
                found.append(path)
        if depth >= max_depth:
            return
        for child in subdirectories_by_name(directory).values():
            if directory_role(child.name) == "unknown":
                walk(child, depth + 1)

    walk(root, 0)
    return sorted(found)


def _stray_image_findings(root: Path, stray: list[Path]) -> list[Finding]:
    """Return the finding for images belonging to no project, if there are any.

    A warning rather than info, for the reason
    :func:`_unindexed_dataset_findings` gives: this is data the collection does
    not contain, and nothing else in the survey mentions it.
    """
    if not stray:
        return []
    shown = ", ".join(make_path_relative(path, root) for path in stray[:3])
    more = "" if len(stray) <= 3 else f", and {len(stray) - 3} more"
    return [
        Finding(
            severity="warning",
            category="stray-images",
            path=root,
            message=(
                f"{len(stray)} image file(s) sit under the scan root but inside no "
                f"project, so nothing indexes them: {shown}{more}. Move each into the "
                f"project it belongs to, or make its directory one"
            ),
        )
    ]


def _long_path_findings(
    root: Path, projects: Iterable[ProjectSurvey], works: bool | None = None
) -> list[Finding]:
    """Return the finding for paths Windows cannot reach here, if there are any.

    Counts every image, its label, and the archived copy saving that label
    would keep, since the archive's are the longest paths and the ones this
    package writes. Nothing to report wherever long paths work.

    Args:
        works: Whether long paths work; :func:`~.io.long_paths.long_paths_work` by default.
    """
    if long_paths_work() if works is None else works:
        return []
    paths: list[Path] = []
    for project in projects:
        for lamella in project.lamellae:
            for stage in lamella.stages:
                for entry in stage.image_sets:
                    labels = entry.image_set.labels_directory
                    for name in entry.image_names:
                        paths.append(entry.image_set.directory / name)
                        if labels is None:
                            continue
                        paths.append(labels / name)
                        archived = projected_archive_path(project.path, labels / name)
                        if archived is not None:
                            paths.append(archived)
    too_long = unreachable(paths, works=False)
    if not too_long:
        return []
    return [
        Finding(
            severity="warning",
            category="windows-path-length",
            path=root,
            message=(
                f"{len(too_long)} path(s) here -- images, labels, or the archived "
                f"versions saving a label keeps -- are {WINDOWS_MAX_PATH} characters "
                f"or longer, which Windows cannot reach in this process: "
                f"{make_path_relative(too_long[0], root)} is the longest. Saving, "
                f"converting and exporting refuse them. {explanation()}"
            ),
        )
    ]


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
