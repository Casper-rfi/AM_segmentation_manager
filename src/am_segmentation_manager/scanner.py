"""Filesystem scanner and project index.

The scanner is responsible for discovering Autolamella projects by locating
experiment.yaml files and building an in-memory ProjectIndex. The remainder of
the package should operate on the ProjectIndex rather than re-traversing the
filesystem.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .discovery import (
    DiscoveredStage,
    discover_stages,
    find_project_roots,
    group_stages_by_lamella,
)
from .image_metadata import LooseHeader, header_reader
from .io.path import make_path_relative
from .io.yaml import read_yaml
from .layout import (
    PREDICTION_KINDS,
    STAGE_SUMMARY_PLOTS,
    STAGE_TIMESTAMP,
    ImageSet,
    build_image_sets,
    build_loose_image_sets,
    image_files,
    image_files_by_name,
    image_files_of,
    resolve_predictions_directory,
    subdirectories_by_name,
)
from .metadata import (
    AM_METADATA_FILENAME,
    EXPERIMENT_FILENAME,
    build_experiment_metadata,
    merge_metadata,
)
from .models import ImageRecord, Lamella, Project, Stage
from .state import StageState, StateFileError, read_state, state_key

logger = logging.getLogger(__name__)


class ProjectIndex:
    """Container for discovered projects.

    Projects are keyed by their path relative to the scan root, not by name.
    Names are not unique -- a project copied to a second directory keeps its
    name, and so do its ``created_at`` and recorded ``path`` in
    ``experiment.yaml``, leaving the filesystem path as the only thing that
    distinguishes the two. Keying by name silently dropped one of them.
    """

    def __init__(self, start_path: Path | None = None) -> None:
        self.start_path = start_path
        self.projects: dict[str, Project] = {}

    def add(self, project: Project) -> None:
        """Add a project, keyed by its identity.

        A second project with the same identity replaces the first, which can
        only happen if the same directory is added twice.
        """
        if project.key in self.projects:
            logger.warning("Replacing already-indexed project %s", project.key)
        self.projects[project.key] = project

    def all_projects(self) -> list[Project]:
        return list(self.projects.values())

    def find_by_key(self, key: str) -> Project:
        """Return the project with the given identity (its relative path)."""
        return self.projects[key]

    def find_by_name(self, name: str) -> Project:
        """Return the single project called *name*.

        Raises:
            KeyError: If no project has that name.
            ValueError: If several do, since the caller's intent is then
                genuinely ambiguous and picking one would be a guess. Use
                :meth:`find_by_key` with a relative path instead.
        """
        matches = [project for project in self.projects.values() if project.name == name]
        if not matches:
            raise KeyError(name)
        if len(matches) > 1:
            locations = ", ".join(project.key for project in matches)
            raise ValueError(
                f"{len(matches)} projects are named {name!r} ({locations}); "
                f"identify the one you mean by its path"
            )
        return matches[0]

    def relative_path(self, path: Path) -> str:
        """Return a string path relative to the scan start_path."""
        if self.start_path is None:
            return str(path)
        return make_path_relative(path, self.start_path)


def _without_names(image_set: ImageSet, names: set[str]) -> ImageSet | None:
    """Return *image_set* without *names*, or None if nothing is left of it.

    Only a loose set can lose names: a directory-shaped one is its whole
    directory, and nothing in this package removes a file from it.
    """
    if image_set.image_names is None:
        return image_set
    remaining = tuple(name for name in image_set.image_names if name not in names)
    if not remaining:
        return None
    return replace(image_set, image_names=remaining)


class Scanner:
    """Scan a starting directory for Autolamella projects.

    The scanner looks for experiment.yaml files. When found, it loads
    experiment metadata (via read_yaml) and optional AM_metadata.yaml that can
    override experiment values. It then constructs a lightweight Project model
    and adds it to a ProjectIndex.
    """

    EXPERIMENT_FILENAME = EXPERIMENT_FILENAME
    AM_METADATA_FILENAME = AM_METADATA_FILENAME
    PLOTS_DIRNAME = "plots"
    PLOT_EXTENSIONS = {".png"}

    def __init__(self, start_path: Path) -> None:
        self.start_path = start_path
        # One header read per loose image, memoised for the life of this scan.
        # The beam and the acquisition time come out of the same read but are
        # wanted at different moments -- the beam while image sets are built,
        # the time while records are made -- and opening the file twice would
        # double the only part of a scan that is not a filesystem walk.
        self._headers: dict[Path, LooseHeader] = {}
        self._read_header = header_reader()

    def _header(self, image_path: Path) -> LooseHeader:
        """Return a loose image's beam and acquisition time, reading once."""
        cached = self._headers.get(image_path)
        if cached is None:
            cached = (
                self._read_header(image_path)
                if self._read_header is not None
                else (None, None)
            )
            self._headers[image_path] = cached
        return cached

    def _resolve_modality(self, image_path: Path) -> str | None:
        """Return the beam an image's header states, for a name that does not."""
        return self._header(image_path)[0]

    def find_project_roots(self) -> list[Path]:
        """Return the project roots under start_path.

        A project is a directory holding ``experiment.yaml`` or
        ``AM_metadata.yaml``; see :data:`discovery.PROJECT_MARKERS`. Discovery
        stops at a project root, so a marker inside a lamella cannot create a
        phantom nested project.
        """
        logger.info("Scanning for projects under %s", self.start_path)
        roots = find_project_roots(self.start_path)
        logger.debug("Found %d project root(s)", len(roots))
        return roots

    def find_experiment_files(self) -> Iterator[Path]:
        """Yield the experiment.yaml path of every discovered project.

        Retained for callers that want the marker files themselves. Projects
        marked only by ``AM_metadata.yaml`` have no experiment file and are
        skipped here; use :meth:`find_project_roots` to see every project.
        """
        for root in self.find_project_roots():
            candidate = root / self.EXPERIMENT_FILENAME
            if candidate.is_file():
                yield candidate

    def build_index(self) -> ProjectIndex:
        """Scan and build a ProjectIndex for discovered projects.

        Discovers projects, their lamellae and the stages within them,
        indexing each stage's image sets -- of every series -- and matching
        predictions and labels by filename, so downstream operations work on
        in-memory objects rather than re-traversing the filesystem.
        """
        index = ProjectIndex(start_path=self.start_path)

        for project_index, project_root in enumerate(sorted(self.find_project_roots()), start=1):
            project = self._build_project(project_root, project_index)
            index.add(project)
            logger.info("Indexed project %s with %d lamellae", project.key, len(project.lamellae))

        return index

    def _build_project(self, project_root: Path, project_index: int) -> Project:
        """Build one project's model from its directory."""
        experiment_path = project_root / self.EXPERIMENT_FILENAME
        am_meta_path = project_root / self.AM_METADATA_FILENAME
        experiment_meta = read_yaml(experiment_path) if experiment_path.exists() else {}
        am_meta = read_yaml(am_meta_path) if am_meta_path.exists() else {}
        merged_meta = merge_metadata(experiment_meta, am_meta)

        metadata_name = merged_meta.get("name")
        project_name = (
            metadata_name if isinstance(metadata_name, str) and metadata_name else project_root.name
        )
        project = Project(
            name=project_name,
            path=project_root,
            relative_path=make_path_relative(project_root, self.start_path),
            metadata=merged_meta,
            experiment_metadata=build_experiment_metadata(
                experiment_meta,
                am_meta,
                experiment_path=experiment_path if experiment_path.exists() else None,
                am_metadata_path=am_meta_path if am_meta_path.exists() else None,
            ),
        )
        # Project numbers are scan-order identifiers, independent of the
        # project folder name, so users can select projects as 001, 002…
        project.number = f"{project_index:03d}"
        project.lamellae = self._build_lamellae(project_root, project_name)
        return project

    def _build_lamellae(self, project_root: Path, project_name: str) -> list[Lamella]:
        """Build the lamellae of a project, synthesising absent levels.

        Older datasets collapse the hierarchy: some have no stage directory
        below the lamella, and the oldest have neither. Discovery reports which
        levels are absent, and the parent directory stands in for each one, so
        those datasets index without their files having to be reorganised.
        """
        grouped = group_stages_by_lamella(discover_stages(project_root))

        lamellae: list[Lamella] = []
        for lamella_path in sorted(grouped):
            discovered = grouped[lamella_path]
            synthetic = discovered[0].lamella_is_synthetic
            lamella = Lamella(
                name=lamella_path.name,
                path=lamella_path,
                synthetic=synthetic,
            )
            # A synthesised lamella is the project's only one, so numbering it
            # 01 keeps `--lamella 1` working across every layout.
            lamella.number = "01" if synthetic else self._parse_numeric_prefix(lamella_path.name)
            lamella.stages = self._build_stages(
                discovered, project_name=project_name, lamella_name=lamella_path.name
            )
            lamellae.append(lamella)
        return lamellae

    def _build_stages(
        self, discovered: list[DiscoveredStage], project_name: str, lamella_name: str
    ) -> list[Stage]:
        """Build the stages of one lamella, ordered by acquisition time.

        Built before they are ordered, because a stage whose folder name
        carries no timestamp is placed by the images inside it, and those are
        not known until it has been built.
        """
        stages: list[Stage] = []
        for item in discovered:
            stage = Stage(
                name=item.path.name,
                path=item.path,
                synthetic=item.stage_is_synthetic,
            )
            stage.image_sets = self._stage_image_sets(item)
            stage.images = self._discover_images_in_stage(
                item.path,
                stage.image_sets,
                project_name=project_name,
                lamella_name=lamella_name,
            )
            stage.plots = self._discover_plots(item.path)
            stage.summary_plot = self._discover_summary_plot(item.path)
            stages.append(stage)
            logger.debug(
                "Discovered stage %s with %d images and %d plots",
                item.path, len(stage.images), len(stage.plots),
            )

        stages.sort(key=self._stage_sort_key)
        self._drop_copied_images(stages, lamella_name)
        for index, stage in enumerate(stages, start=1):
            stage.number = f"{index:02d}"
        return stages

    @staticmethod
    def _drop_copied_images(stages: list[Stage], lamella_name: str) -> None:
        """Index one record per image, where a lamella holds the same file twice.

        Autolamella copies the image an alignment aligns *against* into every
        run that uses it, so one acquisition appears as up to six files.
        Indexing each would put the same picture in a timeline several times,
        count it several times in an export, and pair it with the same label
        repeatedly in a training set -- and annotating it would mean annotating
        the same picture again in every run.

        A copy is a filename and an acquisition time another directory in this
        lamella has already used. **The filename is what makes this safe.**
        Two images taken within one second are ordinary -- an alignment run
        takes several -- so matching on time and size alone merged 38 groups of
        genuinely different pictures across the reference collection, among
        them ``new_image_00`` and ``new_image_01`` of one run. With the
        filename it merges 34 groups of 70 files, every one of them a
        ``reference_image.tif`` that is byte-identical to the one it is
        merged with.

        Stages must already be in order, so that the copy kept is the earliest.
        """
        seen: set[tuple[str, datetime]] = set()
        for stage in stages:
            kept: list[ImageRecord] = []
            copied: set[str] = set()
            for record in stage.images:
                if record.acquired_at is None:
                    kept.append(record)
                    continue
                key = (record.image_path.name, record.acquired_at)
                if key in seen:
                    copied.add(record.image_path.name)
                    continue
                seen.add(key)
                kept.append(record)
            if not copied:
                continue
            logger.debug(
                "%s: %s holds %d image(s) already indexed in another directory",
                lamella_name, stage.name, len(copied),
            )
            stage.images = kept
            stage.image_sets = [
                narrowed
                for image_set in stage.image_sets
                if (narrowed := _without_names(image_set, copied)) is not None
            ]

    def _stage_image_sets(self, discovered: DiscoveredStage) -> list[ImageSet]:
        """Return every image set in a stage, of both shapes.

        Adaptive Polishing sets first, then the loose reference and alignment
        ones, so display order puts the images this package was built to curate
        at the top. What a loose set's files could not be made of -- an
        unrecognised name, or a beam nothing could establish -- is logged here
        and reported properly by ``inspect-layout``, which surveys the tree
        itself rather than reading this.
        """
        sets = list(build_image_sets(discovered.path))
        loose = build_loose_image_sets(
            discovered.path,
            in_alignment_run=discovered.in_alignment_run,
            resolve_modality=self._resolve_modality,
        )
        if loose.unrecognised:
            logger.debug(
                "%d image(s) in %s match no naming rule: %s",
                len(loose.unrecognised), discovered.path, ", ".join(loose.unrecognised[:5]),
            )
        if loose.unresolved:
            logger.info(
                "%d image(s) in %s state no beam and none could be read; not indexed",
                len(loose.unresolved), discovered.path,
            )
        return sets + list(loose.sets)

    @classmethod
    def _discover_plots(cls, stage_path: Path) -> list[Path]:
        """Return the summary plot images written for a stage.

        Plots are rendered during lamella preparation and combine an image with
        its measurement plots. They are informational only: a stage may have no
        plots folder, and stages routinely contain fewer plots than images.
        """
        plots_directory = stage_path / cls.PLOTS_DIRNAME
        if not plots_directory.exists():
            return []
        return [
            p for p in sorted(plots_directory.iterdir())
            if p.is_file() and p.suffix.lower() in cls.PLOT_EXTENSIONS
        ]

    @staticmethod
    def _discover_summary_plot(stage_path: Path) -> Path | None:
        """Return the plot describing the whole stage, if it wrote one.

        At the stage root rather than in ``plots/``, and singular: it covers
        every acquisition in the stage, so there is nothing to match it to.
        """
        for name in STAGE_SUMMARY_PLOTS:
            candidate = stage_path / name
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def _stage_sort_key(cls, stage: Stage) -> tuple[bool, datetime, str]:
        """Order stages by when their images were taken.

        The timestamp in the folder name where there is one, which every
        Adaptive Polishing stage has and which costs nothing to read. A stage
        found by holding loose images usually has none -- a lamella directory
        is named for the lamella and a milling task for the task -- so those
        are placed by their earliest image instead: the same question answered
        from the images rather than from the name. A stage with neither is
        placed last, then by path for deterministic output.
        """
        timestamp = cls._timestamp_from_filename(stage.name)
        if timestamp is None:
            taken = [
                record.acquired_at
                for record in stage.images
                if record.acquired_at is not None
            ]
            timestamp = min(taken) if taken else None
        return (timestamp is None, timestamp or datetime.max, str(stage.path))

    @staticmethod
    def _timestamp_from_filename(filename: str) -> datetime | None:
        """Extract a common Autolamella timestamp from a folder name.

        The pattern is :data:`~am_segmentation_manager.layout.STAGE_TIMESTAMP`,
        which is also what takes the timestamp *out* of a name for grouping:
        what counts as one has to be the same question in both places. A
        trailing AM/PM suffix is applied to the hour, so an afternoon stage
        sorts after a morning one.
        """
        match = STAGE_TIMESTAMP.search(filename)
        if match is None:
            return None
        year, month, day, hour, minute, second = (int(value) for value in match.groups()[:6])
        hour = Scanner._apply_meridiem(hour, match.group(7), filename)
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            return None

    @staticmethod
    def _apply_meridiem(hour: int, meridiem: str | None, filename: str) -> int:
        """Convert a 12-hour clock *hour* to 24-hour form.

        Returns *hour* unchanged when there is no AM/PM suffix, or when the
        suffix contradicts the hour (for example ``13-00-00PM``), which would
        otherwise turn an already-24-hour timestamp into an invalid one.
        """
        if meridiem is None:
            return hour
        if not 1 <= hour <= 12:
            logger.warning(
                "Ignoring '%s' suffix on 24-hour timestamp in %s", meridiem, filename
            )
            return hour
        is_pm = meridiem.lower() == "pm"
        if hour == 12:
            return 12 if is_pm else 0
        return hour + 12 if is_pm else hour

    @staticmethod
    def _parse_numeric_prefix(value: str) -> str | None:
        """Return a folder name's lamella number, zero-padded, when it has one.

        Autolamella names lamellae with a leading number (``01-neat-ape``).
        Older datasets put it at the end instead (``lam_010``, ``lamella_01``),
        so a trailing number after a separator is accepted too, which keeps
        ``--lamella 10`` working across layouts.

        Only those two positions are read. A number elsewhere in a name is far
        more likely to be part of a date or a condition than a lamella number.
        """
        match = re.match(r"^0*(\d+)", value)
        if match is None:
            match = re.search(r"[_-]0*(\d+)$", value)
        if match is None:
            return None
        return f"{int(match.group(1)):02d}"

    def _discover_images_in_stage(
        self,
        stage_path: Path,
        image_sets: list[ImageSet],
        project_name: str,
        lamella_name: str,
    ) -> list[ImageRecord]:
        """Discover image files, predictions and labels inside a stage folder.

        Directory names are resolved through :mod:`am_segmentation_manager.layout`,
        so labels are found under the current ``<image-dir>_labels`` convention
        or any legacy name it recognises.

        Workflow state comes from the stage's ``AM_state.yaml``; a stage
        without one has every group selected, which is the same answer. Files
        still sitting in ``deselected/`` or ``corrected/`` are not indexed at
        all -- ``inspect-layout`` reports them as needing migration, where
        reading them here would report them as selected.

        Labels and predictions are matched by filename **within an image set**
        rather than across the whole stage: several image directories in one
        stage can hold the same filename, so a stage-wide filename map would be
        ambiguous.

        Every image set is indexed, including variant directories capturing one
        acquisition under different imaging conditions. Variants sharing a label
        directory each reference the same label file, which is why a record
        records whether its label is shared.

        A record carries its stage and its label directory rather than leaving
        them to be derived from its image path. Both derivations only ever
        worked for an Adaptive Polishing image, which sits one level below its
        stage in a directory naming a modality; a reference image sits directly
        in its stage, in a directory naming a lamella.
        """
        image_records: list[ImageRecord] = []
        declared = self._declared_state(stage_path)

        for image_set in image_sets:
            labels_map = image_files_by_name(image_set.labels_directory)
            predictions_map = self._collect_predictions(stage_path, image_set)

            for name, image_path in sorted(image_files_of(image_set).items()):
                key = state_key(image_set.labels_directory / name, stage_path)
                record = ImageRecord(
                    image_path=image_path,
                    project=project_name,
                    lamella=lamella_name,
                    stage=stage_path.name,
                    stage_path=stage_path,
                    modality=image_set.modality,
                    variant=image_set.variant,
                    series=image_set.series,
                    acquired_at=self._header(image_path)[1] if image_set.is_loose else None,
                    labels_directory=image_set.labels_directory,
                    label_shared=image_set.labels_are_shared,
                    state=declared.state_of(key),
                )
                prediction = predictions_map.get(name)
                record.predictions = list(prediction) if prediction else []
                label = labels_map.get(name)
                record.labels = [label] if label is not None else []
                image_records.append(record)

        return image_records

    @staticmethod
    def _declared_state(stage_path: Path) -> StageState:
        """Return a stage's declared states, or none at all if unreadable.

        A damaged file is reported and read as empty rather than raised:
        refusing to open a whole collection because one file in it is damaged
        helps nobody, and ``inspect-layout`` names the file.
        """
        try:
            return read_state(stage_path)
        except StateFileError as exc:
            logger.error("%s; treating this stage as selected throughout", exc)
            return StageState(path=stage_path, states={}, fingerprint=None)

    @staticmethod
    def _collect_predictions(stage_path: Path, image_set) -> dict[str, list[Path]]:
        """Return prediction files for an image set, keyed by filename.

        Predictions live in a directory resolved by the layout conventions,
        optionally split into ``raw/`` and ``clean/`` subdirectories. A filename
        may appear in more than one of those, so every match is kept and the
        caller chooses by kind.

        Resolved for reading, which accepts a directory shared with another
        image set: that is where the pipeline's own output sits.

        A loose set has none. Nothing predicts reference or alignment images
        yet, and resolving for one would fall back to the shared
        ``<modality>_predictions`` -- which in a dataset collapsing the stage
        level is the Adaptive Polishing pipeline's own output, an entirely
        different set of scenes. When those series become predictable this
        needs the exact-name rule its labels already have.
        """
        if image_set.is_loose:
            return {}

        directory, exists, _ = resolve_predictions_directory(
            image_set.directory, image_set.modality, subdirectories_by_name(stage_path)
        )
        if not exists:
            return {}

        collected: dict[str, list[Path]] = {}
        for candidate in [directory, *(directory / kind for kind in PREDICTION_KINDS)]:
            for path in image_files(candidate):
                collected.setdefault(path.name, []).append(path)
        return collected
