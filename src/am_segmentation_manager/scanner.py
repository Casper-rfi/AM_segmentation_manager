"""Turning a directory tree into the index the rest of the package reads.

The one traversal: :mod:`~am_segmentation_manager.discovery` says where things are
and :mod:`~am_segmentation_manager.layout` what they are called, and this decides
what they mean -- which images form a set, which label and predictions a record
resolves to, what state it is declared in. **Only a scan opens an image**, and only
a loose one, whose name may not say which beam or when; that read is memoised and
kept on the :class:`ProjectIndex`, which everything above works on.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from . import parallel
from .discovery import (
    DiscoveredStage,
    discover_stages,
    find_project_roots,
    group_stages_by_lamella,
)
from .image_metadata import ImageMetadata, header_reader
from .io.path import make_path_relative, slashed
from .io.yaml import read_yaml_or_problem
from .layout import (
    PREDICTION_KINDS,
    STAGE_SUMMARY_PLOTS,
    STAGE_TIMESTAMP,
    ImageSet,
    build_image_sets,
    build_loose_image_sets,
    directory_cache,
    image_files,
    image_files_by_name,
    image_files_of,
    is_alignment_directory,
    is_alignment_run_reference,
    is_ignored_image_name,
    resolve_loose_predictions_directory,
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
from .progress import timed
from .state import DEFAULT_STATE, StageState, StateFileError, read_state, state_key
from .tasks import NO_TASKS, TaskVocabulary, read_vocabularies

logger = logging.getLogger(__name__)

#: Called with the projects indexed so far and how many there are. Spelled out
#: because the window and the command line both pass one, and neither should
#: have to guess the argument order.
ProgressCallback = Callable[[int, int], None]

#: Fewer experiment files than this are parsed in this process, parsing them
#: costing less than starting the worker processes would.
EXPERIMENT_POOL_FILES = 16


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
        #: Every header the scan opened, keyed by image path, None where it
        #: found none. The scan wants only the beam and the time; keeping the
        #: rest is what spares anything else opening those files again.
        self.headers: dict[Path, ImageMetadata | None] = {}

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
        """Return the project with the given identity (its relative path).

        Directories may be separated with '\\' as well as '/'.
        """
        if key in self.projects:
            return self.projects[key]
        return self.projects[slashed(key)]

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
    """Build a :class:`ProjectIndex` from a starting directory.

    A project is a directory holding ``experiment.yaml``, or the
    ``AM_metadata.yaml`` this package writes to make a pre-Autolamella dataset
    indexable; where both are present the second overrides the first. Headers
    read along the way are cached on the instance, so one scanner is one scan.
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
        self._headers: dict[Path, ImageMetadata | None] = {}
        self._read_header = header_reader()
        #: How many headers this scan opened and what they cost, reported at
        #: the end of :meth:`build_index`. The only part of a scan that is not
        #: a filesystem walk, so a scan that is slow is one or the other.
        self._header_reads = 0
        self._header_seconds = 0.0

    def _header(self, image_path: Path) -> ImageMetadata | None:
        """Return a loose image's header, reading it once; None if it holds none."""
        if image_path not in self._headers:
            start = time.perf_counter()
            self._headers[image_path] = (
                self._read_header(image_path) if self._read_header is not None else None
            )
            self._header_seconds += time.perf_counter() - start
            self._header_reads += 1
        return self._headers[image_path]

    def _resolve_modality(self, image_path: Path) -> str | None:
        """Return the beam an image's header states, for a name that does not."""
        header = self._header(image_path)
        return header.modality if header is not None else None

    def _acquired_at(self, image_path: Path) -> datetime | None:
        """Return when a loose image was taken, as its header states."""
        header = self._header(image_path)
        return header.acquired_at if header is not None else None

    def _unread_headers(self, discovered: list[DiscoveredStage]) -> list[Path]:
        """Return the loose images of *discovered* whose header is not yet read.

        The predicate is :func:`~..layout.build_loose_image_sets`'s own, so a
        file it will not look at is not opened for it -- which includes the
        names no rule recognises, the header being the only thing that can put
        one of those in an ``other`` set of a stated beam.
        """
        if self._read_header is None:
            return []
        return [
            path
            for stage in discovered
            for path in image_files(stage.path)
            if path not in self._headers and not is_ignored_image_name(path.name)
        ]

    def _read_headers(self, paths: list[Path]) -> None:
        """Read the headers of *paths* together, before anything asks for one.

        Nothing here changes what is read or what it answers -- :meth:`_header`
        still memoises, and a path this misses is simply read when asked for.
        """
        if not paths or self._read_header is None:
            return
        start = time.perf_counter()
        self._headers.update(zip(paths, parallel.map_in_processes(self._read_header, paths)))
        self._header_seconds += time.perf_counter() - start
        self._header_reads += len(paths)

    def _discovered_with_headers(
        self, roots: list[Path]
    ) -> Iterator[tuple[Path, list[DiscoveredStage]]]:
        """Yield each project's stages once the headers of its loose images are read.

        Projects are discovered ahead of being built until their unread headers
        make a batch worth sending to worker processes: read a stage at a time,
        most batches would be too small and be read one after another here.
        Projects are still built, and reported, one at a time.
        """
        pending: list[tuple[Path, list[DiscoveredStage]]] = []
        unread: list[Path] = []
        for position, project_root in enumerate(roots, start=1):
            discovered = list(discover_stages(project_root))
            pending.append((project_root, discovered))
            unread.extend(self._unread_headers(discovered))
            if len(unread) < parallel.MIN_POOL_ITEMS and position < len(roots):
                continue
            self._read_headers(unread)
            yield from pending
            pending, unread = [], []

    def _read_experiments(
        self, roots: list[Path]
    ) -> dict[Path, tuple[dict, str | None]]:
        """Return each project's parsed ``experiment.yaml`` and what would not read, by root.

        Parsed together, in worker processes once there are enough of them:
        PyYAML's parser is C, but building the Python objects is not, and a
        large experiment holds hundreds of thousands of them.
        """
        paths = [
            path for root in roots if (path := root / self.EXPERIMENT_FILENAME).exists()
        ]
        with timed(logger, f"Reading {len(paths)} experiment file(s)"):
            documents = parallel.map_in_processes(
                read_yaml_or_problem, paths, chunk_size=1, min_items=EXPERIMENT_POOL_FILES
            )
        return {path.parent: document for path, document in zip(paths, documents)}

    def find_project_roots(self) -> list[Path]:
        """Return the project roots under start_path.

        A project is a directory holding ``experiment.yaml`` or
        ``AM_metadata.yaml``; see :data:`discovery.PROJECT_MARKERS`. Discovery
        stops at a project root, so a marker inside a lamella cannot create a
        phantom nested project.
        """
        logger.info("Scanning for projects under %s", self.start_path)
        with timed(logger, "Finding project roots"):
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

    def build_index(self, on_progress: ProgressCallback | None = None) -> ProjectIndex:
        """Scan and build a ProjectIndex for discovered projects.

        Discovers projects, their lamellae and the stages within them,
        indexing each stage's image sets -- of every series -- and matching
        predictions and labels by filename, so downstream operations work on
        in-memory objects rather than re-traversing the filesystem.

        Args:
            on_progress: Called as ``on_progress(done, total)`` after each
                project, and once with ``(0, total)`` as soon as there is a
                total to report -- finding the roots comes first and cannot say
                how many there will be. The same contract the worker thread
                uses, so a scan reports the same way wherever it is run from.
        """
        index = ProjectIndex(start_path=self.start_path)

        # One listing per directory for the whole scan: a stage is read four
        # times over on the way to being indexed, and a scan is a snapshot of
        # the filesystem either way. Nothing in here writes.
        with directory_cache(), timed(logger, "Building the index"):
            roots = sorted(self.find_project_roots())
            if on_progress is not None:
                on_progress(0, len(roots))
            experiments = self._read_experiments(roots)
            for project_index, (project_root, discovered) in enumerate(
                self._discovered_with_headers(roots), start=1
            ):
                with timed(logger, f"  {project_root.name}", level=logging.DEBUG):
                    experiment_meta, experiment_problem = experiments.get(
                        project_root, ({}, None)
                    )
                    project = self._build_project(
                        project_root,
                        project_index,
                        experiment_meta,
                        discovered,
                        experiment_problem=experiment_problem,
                    )
                index.add(project)
                if on_progress is not None:
                    on_progress(project_index, len(roots))
                logger.info(
                    "Indexed project %s with %d lamellae", project.key, len(project.lamellae)
                )
        logger.info(
            "Read %d image header(s): %.2fs",
            self._header_reads, self._header_seconds,
        )
        index.headers = self._headers
        return index

    def _build_project(
        self,
        project_root: Path,
        project_index: int,
        experiment_meta: dict,
        discovered: list[DiscoveredStage],
        *,
        experiment_problem: str | None = None,
    ) -> Project:
        """Build one project's model from its parsed experiment and its stages.

        A metadata file that would not read counts as empty and is kept on the
        project as a problem, so one malformed file never stops a scan.
        """
        experiment_path = project_root / self.EXPERIMENT_FILENAME
        am_meta_path = project_root / self.AM_METADATA_FILENAME
        am_meta, am_meta_problem = read_yaml_or_problem(am_meta_path)
        problems = tuple(
            problem for problem in (experiment_problem, am_meta_problem) if problem is not None
        )
        for problem in problems:
            logger.warning(
                "%s. Indexing the project without that file's metadata; the metadata "
                "exports refuse it until the file is fixed",
                problem,
            )
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
            metadata_problems=problems,
        )
        # Project numbers are scan-order identifiers, independent of the
        # project folder name, so users can select projects as 001, 002…
        project.number = f"{project_index:03d}"
        project.lamellae = self._build_lamellae(
            discovered, project_name, read_vocabularies(experiment_meta)
        )
        return project

    def _build_lamellae(
        self,
        discovered_stages: list[DiscoveredStage],
        project_name: str,
        vocabularies: dict[str, TaskVocabulary],
    ) -> list[Lamella]:
        """Build the lamellae of a project, synthesising absent levels.

        Older datasets collapse the hierarchy: some have no stage directory
        below the lamella, and the oldest have neither. Discovery reports which
        levels are absent, and the parent directory stands in for each one, so
        those datasets index without their files having to be reorganised.

        Each lamella takes its own task vocabulary from *vocabularies*, keyed
        by directory name. A lamella the experiment does not list -- every
        dataset predating ``experiment.yaml`` -- gets :data:`~..tasks.NO_TASKS`
        and its images carry no task.
        """
        grouped = group_stages_by_lamella(discovered_stages)

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
                discovered,
                project_name=project_name,
                lamella_name=lamella_path.name,
                lamella_path=lamella_path,
                vocabulary=vocabularies.get(lamella_path.name, NO_TASKS),
            )
            lamellae.append(lamella)
        # After every lamella is built: a copy is only recognisable against the
        # lamella that already holds the acquisition, whichever order they came
        # in. Sorted by path above, so the earliest keeps it.
        self._drop_copies_across_lamellae(lamellae, project_name)
        return lamellae

    def _build_stages(
        self,
        discovered: list[DiscoveredStage],
        project_name: str,
        lamella_name: str,
        lamella_path: Path,
        vocabulary: TaskVocabulary,
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
                lamella_path=lamella_path,
                vocabulary=vocabulary,
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
        """Index one record per acquisition, where a lamella holds it twice.

        Autolamella copies the image an alignment aligns *against* into every
        run that uses it, and saves it under its own name outside the runs as
        well, so one acquisition appears as up to seven files. Indexing each
        would put the same picture in a timeline several times, count it
        several times in an export, and pair it with the same label repeatedly
        in a training set -- and annotating it would mean annotating the same
        picture again in every run.

        Stages must already be in order, so that the copy kept is the earliest.
        """
        Scanner._remove_copies(stages, Scanner._copied_image_paths(stages), lamella_name)

    @staticmethod
    def _remove_copies(stages: list[Stage], copies: set[Path], scope: str) -> None:
        """Take *copies* out of *stages*, narrowing the sets that held them."""
        if not copies:
            return
        for stage in stages:
            dropped = {
                record.image_path.name
                for record in stage.images
                if record.image_path in copies
            }
            if not dropped:
                continue
            logger.debug(
                "%s: %s holds %d image(s) already indexed elsewhere",
                scope, stage.name, len(dropped),
            )
            stage.images = [
                record for record in stage.images if record.image_path not in copies
            ]
            stage.image_sets = [
                narrowed
                for image_set in stage.image_sets
                if (narrowed := _without_names(image_set, dropped)) is not None
            ]

    @staticmethod
    def _drop_copies_across_lamellae(lamellae: list[Lamella], project_name: str) -> None:
        """Index one record per acquisition, where a *project* holds it twice.

        :meth:`_drop_copied_images` sees only within a lamella, which is where
        Autolamella duplicates: it copies an alignment's reference into each run
        below the same lamella. The script-based prototype duplicated *across*
        them instead, writing one lamella's setup image into eleven others, so
        that shape is invisible to a rule scoped to one lamella.

        A copy here is a filename and an acquisition time another lamella has
        already used -- the same key :meth:`_copied_image_paths` was measured
        on, and it is kept rather than loosened to time alone. Over the whole
        reference collection, 2,514 loose images in 301 lamellae, this merges
        11 files in 2 groups, every one byte-identical to what it merged with.
        Dropping the filename from the key would have merged one file more, in
        exchange for a key nothing has measured.

        The earliest lamella keeps the image, so the copy that survives is the
        one whose lamella actually took it.

        **A lamella is never emptied by this.** Where every image a lamella has
        is a copy -- an abandoned entry the software created and wrote a setup
        reference into before the lamella was made again under another name --
        dropping them would take the lamella out of the index and out of the
        survey, leaving nothing to say the directory is there at all. A
        duplicate that can be seen is better than a directory that cannot. The
        within-lamella rule never faces this: it keeps the original in the same
        lamella, so that lamella still shows the acquisition.
        """
        seen: set[tuple[str, datetime]] = set()
        copies: set[Path] = set()
        for lamella in lamellae:
            for stage in lamella.stages:
                for record in stage.images:
                    if record.acquired_at is None:
                        continue
                    key = (record.image_path.name, record.acquired_at)
                    if key in seen:
                        copies.add(record.image_path)
                        continue
                    seen.add(key)

        for lamella in lamellae:
            records = [record for stage in lamella.stages for record in stage.images]
            if records and all(record.image_path in copies for record in records):
                copies.difference_update(record.image_path for record in records)

        for lamella in lamellae:
            Scanner._remove_copies(
                lamella.stages, copies, f"{project_name}/{lamella.name}"
            )

    @staticmethod
    def _copied_image_paths(stages: list[Stage]) -> set[Path]:
        """Return the images that are a second copy of an acquisition indexed already.

        **Two rules, because Autolamella duplicates in two ways**, and one
        filename rule cannot see both.

        A run's ``reference_image.tif`` is a copy wherever the directory holding
        that run's ``Alignment/`` folder already holds an image taken at the
        same moment -- which is where Autolamella saves the original, under a
        name that says what it is (``Polishing_alignment_reference_20-12-30_ib``
        or ``ref_alignment_ib``). The original is kept: it is the one file where
        several runs copy it, its name carries the milling task, and it sits at
        the same point in a time-ordered timeline anyway. Over the reference
        collection this matched 36 of the 51 run references, every one of them
        byte-identical to the image it was matched with and none of the other
        15 -- runs whose reference was never saved outside -- touched. The
        containing directory and the acquisition time agreed on all 36
        independently, so **neither signal is carrying this alone**.

        Otherwise a copy is a filename and an acquisition time another
        directory has already used, which collapses the run copies among
        themselves. **The filename is what makes that one safe**: two images
        taken within one second are ordinary -- an alignment run takes several
        -- so matching on time and size alone merged 38 groups of genuinely
        different pictures, among them ``new_image_00`` and ``new_image_01`` of
        one run.

        Nothing is hashed. The acquisition time is already read for every loose
        image, and with either rule's second condition it partitions these
        exactly as their contents do.
        """
        acquisitions: dict[Path, set[datetime]] = {}
        for stage in stages:
            for record in stage.images:
                if record.acquired_at is not None:
                    acquisitions.setdefault(record.image_path.parent, set()).add(
                        record.acquired_at
                    )

        copies: set[Path] = set()
        seen: set[tuple[str, datetime]] = set()
        for stage in stages:
            for record in stage.images:
                if record.acquired_at is None:
                    continue
                host = Scanner._alignment_run_host(record)
                if host is not None and record.acquired_at in acquisitions.get(host, ()):
                    copies.add(record.image_path)
                    continue
                key = (record.image_path.name, record.acquired_at)
                if key in seen:
                    copies.add(record.image_path)
                    continue
                seen.add(key)
        return copies

    @staticmethod
    def _alignment_run_host(record: ImageRecord) -> Path | None:
        """Return where a run's reference image was copied *from*, if it is one.

        The directory holding the ``Alignment/`` folder the run sits in -- a
        lamella or a milling task -- which is where Autolamella saves the
        original beside its other loose images.

        Returns:
            That directory, or None where *record* is not an alignment run's
            reference image.
        """
        stage_path = record.stage_path
        if stage_path is None or not is_alignment_directory(stage_path.parent.name):
            return None
        if not is_alignment_run_reference(record.image_path.name):
            return None
        return stage_path.parent.parent

    def _stage_image_sets(self, discovered: DiscoveredStage) -> list[ImageSet]:
        """Return every image set in a stage, of both shapes.

        Adaptive Polishing sets first, then the loose reference and alignment
        ones and finally ``other``, so display order puts the images this
        package was built to curate at the top and the residue at the bottom.
        Why a file reached ``other`` rather than a set of its own is logged
        here and reported properly by ``inspect-layout``, which surveys the
        tree itself rather than reading this.
        """
        sets = list(build_image_sets(discovered.path))
        loose = build_loose_image_sets(
            discovered.path,
            in_alignment_run=discovered.in_alignment_run,
            resolve_modality=self._resolve_modality,
        )
        if loose.unrecognised:
            logger.debug(
                "%d image(s) in %s match no naming rule and are indexed as other: %s",
                len(loose.unrecognised), discovered.path, ", ".join(loose.unrecognised[:5]),
            )
        if loose.unresolved:
            logger.info(
                "%d image(s) in %s state no beam and none could be read; "
                "indexed as other", len(loose.unresolved), discovered.path,
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
        lamella_path: Path,
        vocabulary: TaskVocabulary,
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
        # Listed once for the whole stage. A loose set's directory *is* the
        # stage, and an Adaptive Polishing set's is a child of it, so the same
        # mapping answers where either one's labels and predictions resolve to.
        siblings = subdirectories_by_name(stage_path)

        for image_set in image_sets:
            labels_directory = image_set.labels_directory
            labels_map = (
                image_files_by_name(labels_directory) if labels_directory else {}
            )
            predictions_directory = self._predictions_directory(image_set, siblings)
            predictions_map = (
                self._collect_predictions(predictions_directory)
                if predictions_directory
                else {}
            )

            for name, image_path in sorted(image_files_of(image_set).items()):
                record = ImageRecord(
                    image_path=image_path,
                    project=project_name,
                    lamella=lamella_name,
                    stage=stage_path.name,
                    stage_path=stage_path,
                    modality=image_set.modality,
                    variant=image_set.variant,
                    series=image_set.series,
                    acquired_at=self._acquired_at(image_path) if image_set.is_loose else None,
                    labels_directory=labels_directory,
                    predictions_directory=predictions_directory,
                    label_shared=image_set.labels_are_shared,
                    # A set with no label directory has no label group, so
                    # there is nothing for a state file to be keyed on and the
                    # default stands. Nothing may change it: the write paths
                    # refuse this series.
                    state=(
                        declared.state_of(state_key(labels_directory / name, stage_path))
                        if labels_directory
                        else DEFAULT_STATE
                    ),
                    task=self._resolve_task(image_path, lamella_path, vocabulary),
                )
                prediction = predictions_map.get(name)
                record.predictions = list(prediction) if prediction else []
                label = labels_map.get(name)
                record.labels = [label] if label is not None else []
                image_records.append(record)

        return image_records

    @staticmethod
    def _resolve_task(
        image_path: Path, lamella_path: Path, vocabulary: TaskVocabulary
    ) -> str | None:
        """Return the milling task an image belongs to, or None if nothing says.

        The directory decides where it can. The 2026-07 layout puts a task's
        images under ``Milling/<Task>/`` and its alignment runs below that, so
        the nearest directory whose name the lamella's vocabulary knows is the
        task -- and it is the *only* thing that resolves a name Autolamella
        spells differently in the two places, ``Milling/Fiducial/`` for the task
        called ``Mill Fiducial``. The lamella directory itself is not offered:
        a petname sharing a word with a task would claim every image in it.

        The filename answers where the directory does not, which is every
        reference image -- ``ref_Rough Milling_start_eb`` names its task and
        sits in a directory shared by every task's.

        Returns:
            The task as the experiment spells it, or None. None is a real
            answer, not a gap to fill: beam-shift alignment and the images
            inside an alignment run name no task, and neither does an Adaptive
            Polishing stage in a layout predating ``Milling/<Task>/``.
        """
        directory = image_path.parent
        while directory != lamella_path and directory != directory.parent:
            task = vocabulary.resolve(directory.name)
            if task is not None:
                return task
            directory = directory.parent
        return vocabulary.resolve(image_path.stem)

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
    def _predictions_directory(
        image_set: ImageSet, siblings: Mapping[str, Path]
    ) -> Path | None:
        """Return where an image set's predictions live or would live.

        **Reading and writing resolve here, once.** A record carries the answer
        so that :func:`~..operations.predict.prediction_destination` writes
        where the next scan will look; re-deriving it from an image's path put
        a reference image's prediction in a ``<lamella>_predictions`` beside the
        lamella, which nothing then read.

        An Adaptive Polishing set resolves through the fallback chain, which
        accepts a directory shared with another set: that is where the
        segmentation pipeline's own output sits, and only one image of a label
        group is ever predicted, so two conditions cannot claim one filename. A
        loose set resolves by exact name instead, for the reason
        :func:`~..layout.resolve_loose_predictions_directory` gives.

        Returns:
            The directory, whether or not it exists, since that is also where a
            prediction would be written -- or ``None`` for an ``other`` set,
            which is never predicted and whose predictions could only be
            converted into labels it may not have.
        """
        if image_set.series == "other":
            return None
        assert image_set.modality is not None, "only an other set has no modality"
        if image_set.is_loose:
            directory, _ = resolve_loose_predictions_directory(
                image_set.directory, image_set.series, image_set.modality, siblings
            )
            return directory
        directory, _, _ = resolve_predictions_directory(
            image_set.directory, image_set.modality, siblings
        )
        return directory

    @staticmethod
    def _collect_predictions(directory: Path) -> dict[str, list[Path]]:
        """Return the prediction files in *directory*, keyed by filename.

        Predictions sit in the directory itself or split into the
        :data:`~..layout.PREDICTION_KINDS` subdirectories. A filename may appear
        in more than one of those, so every match is kept and the caller
        chooses by kind. A directory that does not exist holds none.
        """
        collected: dict[str, list[Path]] = {}
        for candidate in [directory, *(directory / kind for kind in PREDICTION_KINDS)]:
            for path in image_files(candidate):
                collected.setdefault(path.name, []).append(path)
        return collected
