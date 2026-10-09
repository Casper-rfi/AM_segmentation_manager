"""What the Train tab knows: the run being set up, the runs there are, and how they are looked at.

Holds no Qt and decides nothing an operation does not: preparing, starting, stopping,
marking, validating and comparing are the operations and readers the command line uses.
**Every run shown is read from its directory**, its validations and the epoch marked as
its best with it, so a window opened on another day shows the same; a run started
without am-seg is listed beside the others. What is ticked, which classes and sets are
drawn and what a table's cell holds are the tab's alone, and last only as long as it does.
A metrics file caught mid-rewrite keeps the epochs last read whole.
"""
from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..io.atomic import write_atomically
from ..operations.training_choice import (
    ChosenBest,
    choose_best,
    clear_chosen_best,
    read_chosen_best,
)
from ..operations.training_manifest import read_training_manifest
from ..operations.training_process import RunStatus, run_status, start_training, stop_training
from ..operations.training_run import (
    NAME_PATTERN,
    Mode,
    RunRecord,
    TrainingPlan,
    TrainingSettings,
    apply_training_run,
    default_settings,
    find_run_directories,
    plan_training_run,
    read_pairs,
    read_run_record,
)
from ..operations.validation_run import (
    Rows,
    ValidationPlan,
    apply_validation,
    find_validations,
    plan_validation,
    read_images,
    read_validation_record,
)
from ..training_compare import (
    DEFAULT_COLOUR_RANGE,
    DEFAULT_EPOCH_RANGE,
    EVERY_SET,
    OWN_VALIDATION,
    ComparedRun,
    Comparison,
    ScoredValidation,
    colour_range,
    compare_runs,
    dividing_columns,
    fitted_colour_range,
    validation_sets,
)
from ..training_runs import (
    SEARCH_DEPTH,
    Checkpoint,
    EpochMetrics,
    TrainingRun,
    find_training_runs,
    is_run_directory,
    read_training_run,
)
from ..validation_scores import ImageScores, find_scores, read_scores
from .attribution import Attribution

logger = logging.getLogger(__name__)

#: The states in which a run's files can still change, so are worth reading again.
LIVE_STATES = frozenset({"running", "unknown"})

#: The class whose curve is left out until asked for: it is most of every image,
#: scores near 1 throughout, and says little about the model.
BACKGROUND_CLASS = "background"

#: The columns of the list of runs, each ``(caption, what it holds)``; the first
#: is the run's name, and :meth:`RunSummary.cells` fills the rest.
RUN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Run", "What the run is called."),
    ("State", "How the run stands: training, finished, stopped, failed."),
    ("Epochs", "Epochs trained, of the most the run may train."),
    ("Train", "How many images the run trains on, as it was prepared."),
    ("Val", "How many images the run validates on as it trains."),
    ("Best epoch", "The epoch the run counts as its best, on its own validation list."),
    (
        "Best score",
        "What the run ranks its epochs by, at its best: the mean of its key validation "
        "metrics, the weighted-average IoU and Dice unless the run was told otherwise. "
        "From 0 to 1, higher is better.",
    ),
    ("Marked", "The epoch you marked as best, where you marked one."),
)

#: How recently a run am-seg did not start must have written to count as still
#: training. Nothing else says: it has no process here to ask. An epoch is
#: minutes, so an hour of silence is a run that has stopped.
RECENT_SECONDS = 3600.0


@dataclass(frozen=True)
class ValidationSummary:
    """One validation of a run, as the tab lists it and draws from it.

    Attributes:
        directory: Its directory, in the run's.
        name: What it is called.
        epochs: How many epochs it was asked to score.
        status: How its process stands; None where the record cannot be read.
        scored: What it has scored so far, and on which images.
        problem: Why it cannot be read, or "".
    """

    directory: Path
    name: str
    epochs: int = 0
    status: RunStatus | None = None
    scored: ScoredValidation = ScoredValidation("")
    problem: str = ""

    @property
    def live(self) -> bool:
        """Return whether it may still be scoring."""
        return self.status is not None and self.status.state in LIVE_STATES

    def line(self) -> str:
        """Return the validation in a few words: its name, how it stands, how far it got."""
        if self.problem:
            return f"{self.name}: {self.problem}"
        state = self.status.state if self.status else ""
        said = f"{self.name} {state}, {len(self.scored.scores)} of {self.epochs} epoch(s) scored"
        if state == "failed" and self.status is not None and self.status.error:
            said += f" ({self.status.error})"
        return said


@dataclass(frozen=True)
class RunSummary:
    """One run in the training directory, as the tab lists and draws it.

    Attributes:
        directory: The run's directory; for a run am-seg did not start, the
            directory AM-model-training wrote.
        name: What it is called; a run am-seg did not start is called by the
            folders it sits in.
        folder: The folders between the training directory and the run, or ""
            for one directly in it.
        record: What it was asked to do; None for a run am-seg did not start,
            or where the record cannot be read.
        status: How it stands; None for a run am-seg did not start.
        written: What AM-model-training has written, or None before it has.
        chosen: The epoch a person chose as its best, or None where none is.
        validations: Its validations, by name.
        problem: Why the run cannot be read, or "".
        recent: Whether a run am-seg did not start has written within
            :data:`RECENT_SECONDS`.
    """

    directory: Path
    name: str
    folder: str = ""
    record: RunRecord | None = None
    status: RunStatus | None = None
    written: TrainingRun | None = None
    chosen: ChosenBest | None = None
    validations: tuple[ValidationSummary, ...] = ()
    problem: str = ""
    recent: bool = False

    @property
    def label(self) -> str:
        """Return what the run is called under its folder."""
        return self.name.rsplit("/", 1)[-1]

    @property
    def epochs(self) -> int:
        """Return how many epochs have been trained."""
        return len(self.written.train) if self.written else 0

    @property
    def most_epochs(self) -> int | None:
        """Return the most epochs the run may train, or None where nothing says."""
        if self.record is not None:
            return self.record.settings.max_epochs
        return self.written.max_epochs if self.written is not None else None

    @property
    def saved_epochs(self) -> tuple[int, ...]:
        """Return the epochs the run kept weights after, on disk still or not."""
        if self.written is None:
            return ()
        return self.written.saved_epochs(
            self.record.settings.save_all if self.record is not None else None
        )

    @property
    def kept(self) -> tuple[Checkpoint, ...]:
        """Return the weights still on disk."""
        return self.written.checkpoints if self.written else ()

    def kept_bytes(self) -> int:
        """Return how much disk the weights still there take."""
        total = 0
        for checkpoint in self.kept:
            try:
                total += checkpoint.path.stat().st_size
            except OSError:
                continue
        return total

    def progress(self) -> str:
        """Return how far the run got and what it still holds, a line for each."""
        most = self.most_epochs
        epochs = f"{self.epochs} of at most {most} epochs trained" if most else f"{self.epochs} epoch(s) trained"
        weights = (
            f"{len(self.kept)} of {len(self.saved_epochs)} saved weights on disk, "
            f"{self.kept_bytes() / 1e9:.1f} GB"
        )
        return f"{epochs}\n{weights}"

    @property
    def best(self) -> EpochMetrics | None:
        """Return the validated epoch the run counts as best, if it has one."""
        return self.written.best() if self.written else None

    @property
    def live(self) -> bool:
        """Return whether the run's files, or those of a validation of it, may still change."""
        if any(validation.live for validation in self.validations):
            return True
        return self.training

    @property
    def training(self) -> bool:
        """Return whether the run itself may still be training."""
        if self.status is not None:
            return self.status.state in LIVE_STATES
        return self.written is not None and not self.written.finished and self.recent

    @property
    def state(self) -> str:
        """Return how the run stands in a word or two, from its status or its files."""
        if self.status is not None:
            return self.status.state
        if self.written is None:
            return ""
        if self.written.finished:
            return "finished"
        return "training" if self.recent else "not finished"

    def describe(self) -> str:
        """Return how the run stands in a sentence, for the run selected."""
        if self.problem:
            return self.problem
        if self.status is not None:
            return f"{self.name}: {self.status.describe()}"
        said = {
            "finished": "finished",
            "training": "written to within the last hour, so probably still training",
            "not finished": "not finished: stopped before it ended, or training elsewhere",
        }.get(self.state, "")
        return f"{self.name}: {said}. Not started by am-seg; read from AM-model-training's files."

    def cells(self) -> tuple[str, ...]:
        """Return what the run holds under each of :data:`RUN_COLUMNS` after its name.

        Empty for a run that could not be read, whose :attr:`problem` is all there is to say.
        """
        if self.problem:
            return ()
        most = self.most_epochs
        if most is not None and (self.epochs or self.training):
            epochs = f"{self.epochs}/{most}"
        else:
            epochs = str(self.epochs) if self.epochs else ""
        best = self.best if self.written is not None else None
        record = self.record
        return (
            self.state,
            epochs,
            _counted(record.train_count if record else None),
            _counted(record.validation_count if record else None),
            str(best.epoch) if best is not None else "",
            f"{best.score(self.written.key_metrics):.3f}"
            if best is not None and self.written is not None else "",
            str(self.chosen.epoch) if self.chosen is not None else "",
        )


def _counted(count: int | None) -> str:
    """Return *count* as a cell, empty where nothing says how many."""
    return "" if count is None else str(count)


class TrainingController:
    """The Train tab's state and the operations it asks for.

    Args:
        attribution: Whose name a run and a stop are recorded under; the window's.
        stack_present: Answers whether the training stack is installed.
        defaults: Returns the settings the form opens at for a mode: the
            person's own, where the window has them; the published ones otherwise.
        start: Starts a prepared run; :func:`start_training` unless a test says otherwise.

    Attributes:
        collection: The folder the window has scanned, which a manifest that
            recorded a relative folder is resolved against.
        last_export: The manifest of the training export the window last wrote.
    """

    def __init__(
        self,
        attribution: Attribution | None = None,
        stack_present: Callable[[], bool] | None = None,
        start: Callable[[Path], RunStatus] = start_training,
        defaults: Callable[[Mode], TrainingSettings] = default_settings,
    ) -> None:
        self._defaults = defaults
        self._attribution = attribution
        self._stack_present = stack_present
        self._start = start
        self.collection: Path | None = None
        self.last_export: Path | None = None
        self.training_directory: Path | None = None
        self.manifest: Path | None = None
        self.name: str = ""
        self.settings: TrainingSettings = defaults("new")
        self.base_weights: Path | None = None
        self.device: str = "auto"
        self.selected: Path | None = None
        self.ticked: set[Path] = set()
        self.hidden_classes: set[str] = {BACKGROUND_CLASS}
        self.view: str = "curves"
        self.validated_on: str = ""
        self.split_by: str = ""
        self.subset: str = ""
        self.hidden_sets: set[str] = set()
        self._scores: dict[Path, ImageScores] = {}
        self._images: dict[Path, list[dict[str, str]]] = {}
        self._curves: dict[tuple, dict[str, tuple[EpochMetrics, ...]]] = {}
        self.table_value: str = "best"
        self.epoch_range: tuple[int, int] = DEFAULT_EPOCH_RANGE
        self.colour_range: tuple[float, float] = DEFAULT_COLOUR_RANGE
        self.auto_colours: bool = True
        self._whole: dict[Path, TrainingRun] = {}

    # -- what is installed ----------------------------------------------------

    def can_train(self) -> bool:
        """Return whether the training stack is installed, without importing it."""
        if self._stack_present is not None:
            return self._stack_present()
        from ..ml.environment import training_stack_present

        return training_stack_present()

    def install_hint(self) -> str:
        """Return how to install the training stack."""
        from ..ml.environment import TRAINING_INSTALL_HINT

        return TRAINING_INSTALL_HINT

    def devices(self) -> list[tuple[str, str]]:
        """Return where a run can train as ``(device, caption)``: auto, each NVIDIA GPU, the CPU.

        Read from ``nvidia-smi``, never torch, so listing them costs no import.
        """
        from ..ml.gpus import gpu_inventory

        offered = [("auto", "auto")]
        offered.extend((gpu.device, gpu.describe()) for gpu in gpu_inventory().gpus)
        offered.append(("cpu", "cpu"))
        return offered

    # -- the run being set up -------------------------------------------------

    def exported(self, manifest: Path) -> None:
        """Note the training export the window has just written, choosing it if none is chosen."""
        self.last_export = manifest
        if self.manifest is None:
            self.manifest = manifest

    def training_set_summary(self) -> str:
        """Return the chosen training set in a line, or why it cannot be read, or ""."""
        if self.manifest is None:
            return ""
        try:
            manifest = read_training_manifest(self.manifest)
            counts = [
                len(read_pairs(path)) if path is not None else 0
                for path in (manifest.train, manifest.validation)
            ]
        except (OSError, ValueError) as error:
            return str(error)
        return f"{counts[0]} training and {counts[1]} validation pair(s)"

    @property
    def mode(self) -> Mode:
        """Return whether a new model is trained or an existing one refined."""
        return self.settings.mode

    def set_mode(self, mode: Mode) -> None:
        """Switch between a new model and a refine, taking that mode's default settings."""
        if mode != self.settings.mode:
            self.settings = self._defaults(mode)

    def change(self, **settings: Any) -> None:
        """Change settings of the run being set up, by :class:`TrainingSettings` field."""
        self.settings = replace(self.settings, **settings)

    def reset_settings(self) -> None:
        """Put every setting back to what the form opens at for the mode."""
        self.settings = self._defaults(self.settings.mode)

    def reset_recommended(self) -> None:
        """Put every setting back to what the models were published with, for the mode."""
        self.settings = default_settings(self.settings.mode)

    def suggest_name(self, today: datetime | None = None) -> str:
        """Return a name no run in the training directory has: the date, the mode, a number."""
        stem = f"{(today or datetime.now()):%Y%m%d}_{self.settings.mode}"
        taken = self._taken_names()
        number = 1
        while f"{stem}_{number:02d}" in taken:
            number += 1
        return f"{stem}_{number:02d}"

    def problems(self) -> list[str]:
        """Return why the run cannot be started yet, from what costs nothing to check.

        What needs the export read and each label opened is :meth:`plan`'s to find.
        """
        found = []
        if not self.can_train():
            found.append("The training stack is not installed")
        if self.manifest is None:
            found.append("Choose a training set: the manifest a training export wrote")
        elif self._collection_unknown():
            found.append(
                "Scan the collection the training set was exported from: its manifest "
                "does not say where it is"
            )
        if self.training_directory is None:
            found.append("Choose the training directory, where runs are kept")
        if not self.name:
            found.append("Name the run")
        elif not NAME_PATTERN.match(self.name):
            found.append("A name holds letters, digits, '.', '_' and '-'")
        elif self.name in self._taken_names():
            found.append(f"A run named {self.name} already exists")
        if self.settings.mode == "refine" and self.base_weights is None:
            found.append("Choose the model weights to refine")
        found.extend(self.settings.problems())
        return found

    def plan(self) -> TrainingPlan:
        """Check the run being set up in full, each label opened, and return it ready to write.

        Raises:
            TrainingRunRefused: Naming every reason; nothing has been written.
            ValueError: If no training set or training directory is chosen.
        """
        if self.manifest is None or self.training_directory is None:
            raise ValueError("Choose a training set and a training directory first")
        return plan_training_run(
            self.manifest, self.training_directory, self.name, self.settings,
            base_weights=self.base_weights if self.settings.mode == "refine" else None,
            device=self.device, start_path=self._start_path(),
        )

    def start(self, plan: TrainingPlan, user: str | None = None) -> RunSummary:
        """Write the run *plan* describes, start it, and return it selected.

        Args:
            user: The name to record in place of the window's.

        Raises:
            FileExistsError: If its directory has appeared since the plan was made.
            TrainingNotStarted: If the process did not come up; the directory stays.
        """
        directory = apply_training_run(plan, user=user or self._user())
        self.selected = directory
        self._start(directory)
        return self.summary(directory)

    # -- the runs there are ---------------------------------------------------

    def runs(self) -> list[RunSummary]:
        """Return every run in the training directory, the newest first.

        Those am-seg prepared, and each directory AM-model-training wrote that
        belongs to none of them: a run started with it directly, here or elsewhere.
        """
        if self.training_directory is None or not self.training_directory.is_dir():
            return []
        prepared = find_run_directories(self.training_directory, depth=SEARCH_DEPTH)
        others = [
            written for written in find_training_runs(self.training_directory)
            if written.parent not in prepared
        ]
        summaries = [self.summary(directory) for directory in (*prepared, *others)]
        return sorted(summaries, key=lambda run: (_when(run), run.name), reverse=True)

    def summary(self, directory: Path) -> RunSummary:
        """Return the run in *directory* as its files say it stands now.

        Args:
            directory: A run am-seg prepared, or a directory AM-model-training wrote.
        """
        if is_run_directory(directory):
            name = self._name_of(directory)
            return RunSummary(
                directory=directory,
                name=name,
                folder=name.rpartition("/")[0],
                written=self._steady(read_training_run(directory)),
                chosen=_chosen(directory),
                validations=self._validations(directory),
                recent=_written_recently(directory),
            )
        try:
            record = read_run_record(directory)
            status = run_status(directory)
        except ValueError as error:
            return RunSummary(
                directory, directory.name, folder=self._folder_of(directory), problem=str(error)
            )
        written = find_training_runs(directory, depth=0)
        return RunSummary(
            directory=directory,
            name=record.name if record else directory.name,
            folder=self._folder_of(directory),
            record=record,
            status=status,
            # A run directory is started once, so it holds one of these at most.
            written=self._steady(read_training_run(written[-1])) if written else None,
            chosen=_chosen(directory),
            validations=self._validations(directory),
        )

    def _validations(self, run_directory: Path) -> tuple[ValidationSummary, ...]:
        """Return the validations of the run in *run_directory*, a scores file read once."""
        found = []
        for directory in find_validations(run_directory):
            try:
                record = read_validation_record(directory)
                status = run_status(directory)
            except ValueError as error:
                found.append(ValidationSummary(directory, directory.name, problem=str(error)))
                continue
            assert record is not None
            scores = []
            for path in find_scores(directory):
                if path not in self._scores:
                    try:
                        self._scores[path] = read_scores(path)
                    except ValueError as error:
                        logger.warning("%s", error)
                        continue
                scores.append(self._scores[path])
            if record.images_file not in self._images:
                try:
                    self._images[record.images_file] = read_images(record.images_file)
                except OSError as error:
                    logger.warning("Could not read %s: %s", record.images_file, error)
                    self._images[record.images_file] = []
            found.append(ValidationSummary(
                directory, record.name, len(record.epochs), status,
                ScoredValidation(record.name, tuple(scores), self._images[record.images_file]),
            ))
        return tuple(found)

    def selected_run(self) -> RunSummary | None:
        """Return the run selected, read now, or None if none is or it has gone."""
        if self.selected is None or not self.selected.is_dir():
            return None
        return self.summary(self.selected)

    def start_cautions(
        self, plan: TrainingPlan, runs: Sequence[RunSummary] | None = None
    ) -> list[str]:
        """Return what a person should know before starting *plan* beside the runs training now.

        Empty where none is. Only the runs of the training directory are known.

        Args:
            runs: The runs to look among; read now where omitted.
        """
        training = [run for run in (self.runs() if runs is None else runs) if run.training]
        if not training:
            return []
        names = ", ".join(run.name for run in training)
        said = [f"{len(training)} run(s) here may be training now: {names}."]
        if plan.device == "auto":
            said.append(
                "auto takes the GPU with the most memory free at the moment a run starts, "
                "and a run started moments ago has taken little yet: name a GPU to keep "
                "two runs apart."
            )
        return said

    # -- the runs ticked ------------------------------------------------------

    def tick(self, directory: Path, ticked: bool) -> None:
        """Tick the run in *directory*, or clear its tick."""
        if ticked:
            self.ticked.add(directory)
        else:
            self.ticked.discard(directory)

    def tick_all(self, ticked: bool, runs: Sequence[RunSummary] | None = None) -> None:
        """Tick every run listed, or clear every tick.

        Args:
            runs: The runs listed, where they have just been read.
        """
        self.ticked = (
            {run.directory for run in (self.runs() if runs is None else runs)} if ticked else set()
        )

    def ticked_runs(self) -> list[RunSummary]:
        """Return the ticked runs still in the training directory, in the order listed."""
        return [run for run in self.runs() if run.directory in self.ticked]

    # -- the classes drawn ----------------------------------------------------

    def show_class(self, name: str, shown: bool) -> None:
        """Draw class *name*'s curve, or leave it out."""
        if shown:
            self.hidden_classes.discard(name)
        else:
            self.hidden_classes.add(name)

    def shown_classes(self, names: Sequence[str]) -> tuple[str, ...]:
        """Return those of a run's classes that are drawn, in its order.

        What is left out is remembered by name, so it stays out from one run to the next.
        """
        return tuple(name for name in names if name not in self.hidden_classes)

    # -- the ticked runs side by side -----------------------------------------

    def comparison(self, metric: str, runs: Sequence[RunSummary] | None = None) -> Comparison:
        """Return the ticked runs that have validated an epoch, side by side on *metric*.

        The classes drawn are its columns; a cell's value and the epochs a mean
        or median is over are :attr:`table_value` and :attr:`epoch_range`.

        Args:
            runs: The runs to compare in place of those ticked.
        """
        compared = [
            ComparedRun(
                run.name, run.written, run.chosen.epoch if run.chosen else None, self.sets_of(run)
            )
            for run in (self.ticked_runs() if runs is None else runs)
            if run.written is not None and run.written.val
        ]
        names = tuple(dict.fromkeys(name for run in compared for name in run.run.label_names))
        offered = self.set_names([run.sets for run in compared])
        return compare_runs(
            compared, metric=metric, classes=self.shown_classes(names),
            sets=[name for name in offered if name not in self.hidden_sets],
            value=self.table_value, epoch_range=self.epoch_range,
        )

    def set_colour_range(self, low: float, high: float) -> None:
        """Have the table's colours run from *low* to *high*, and stop following the table."""
        self.auto_colours = False
        self.colour_range = colour_range(low, high)

    def colours_for(self, table: Comparison) -> tuple[float, float]:
        """Return the range *table* is shaded over: its own while following it, else the one set."""
        if self.auto_colours:
            self.colour_range = fitted_colour_range(table)
        return self.colour_range

    def fit_colours(self, table: Comparison) -> None:
        """Set the range to what *table* spans, once, following it or not as before."""
        self.colour_range = fitted_colour_range(table)

    # -- what a run is validated on -------------------------------------------

    def validation_names(self, runs: Sequence[RunSummary]) -> tuple[str, ...]:
        """Return what the validations of *runs* are called, each once, in name order."""
        return tuple(sorted({v.name for run in runs for v in run.validations if not v.problem}))

    def dividing_columns(self, runs: Sequence[RunSummary]) -> tuple[str, ...]:
        """Return the metadata columns the validation chosen can be divided by, over *runs*."""
        if self.validated_on in ("", EVERY_SET):
            return ()
        return dividing_columns([
            v.scored for run in runs for v in run.validations if v.name == self.validated_on
        ])

    def sets_of(self, run: RunSummary) -> dict[str, tuple[EpochMetrics, ...]]:
        """Return *run*'s validation curves beyond its own, as :attr:`validated_on` and :attr:`split_by` ask.

        Kept until the run has scored another epoch: a curve over a part of a
        set is every image's matrix added up for every epoch.
        """
        scored = [v.scored for v in run.validations if not v.problem]
        key = (
            run.directory, self.validated_on, self.split_by,
            tuple((v.name, len(v.scores)) for v in scored),
        )
        if key not in self._curves:
            self._curves[key] = validation_sets(scored, self.validated_on, self.split_by)
        return self._curves[key]

    def set_names(self, sets: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
        """Return the sets a table of runs holding *sets* can give a column to, the runs' own first.

        The runs' own validation is offered unless one validation is asked for:
        its parts are then the question, and a run's own list is another set of images.
        """
        own = (OWN_VALIDATION,) if self.validated_on in ("", EVERY_SET) else ()
        return (*own, *dict.fromkeys(name for held in sets for name in held))

    def curve_run(self, run: RunSummary) -> tuple[TrainingRun | None, str]:
        """Return what to draw *run*'s curves from, and what its validation is called.

        Its own files, or those with the validation side replaced by the set
        chosen: :attr:`subset` of it where it is divided, else all of it.
        """
        written = run.written
        if written is None or self.validated_on in ("", EVERY_SET):
            return written, ""
        sets = self.sets_of(run)
        name = self.subset if self.split_by and self.subset in sets else next(iter(sets), "")
        if not name:
            return written, ""
        said = self.validated_on if name == self.validated_on else f"{self.validated_on}, {name}"
        return replace(written, val=sets[name]), said

    # -- validating -----------------------------------------------------------

    def plan_validation(
        self, run_directory: Path, manifest: Path, name: str, rows: Rows = "validation"
    ) -> ValidationPlan:
        """Check a validation of the run in *run_directory* in full, each label opened.

        Raises:
            TrainingRunRefused: Naming every reason; nothing has been written.
        """
        return plan_validation(
            run_directory, manifest, name, rows=rows, device=self.device,
            start_path=self.collection,
        )

    def start_validation(self, plan: ValidationPlan, user: str | None = None) -> Path:
        """Write the validation *plan* describes, start it, and return its directory.

        Raises:
            FileExistsError: If its directory has appeared since the plan was made.
            TrainingNotStarted: If the process did not come up; the directory stays.
        """
        directory = apply_validation(plan, user=user or self._user())
        self._start(directory)
        return directory

    def stop_validations(self, run_directory: Path, user: str | None = None) -> None:
        """Stop every validation of the run in *run_directory* that is running."""
        for validation in self._validations(run_directory):
            if validation.live:
                stop_training(validation.directory, user=user or self._user())

    def _folder_of(self, directory: Path) -> str:
        """Return the folders between the training directory and a prepared run, or ""."""
        if self.training_directory is None:
            return ""
        try:
            folder = directory.parent.relative_to(self.training_directory).as_posix()
        except ValueError:
            return ""
        return "" if folder == "." else folder

    def _name_of(self, written: Path) -> str:
        """Return what to call a run am-seg did not start: the folder it was written into.

        Its own directory's name is added where that folder holds several runs,
        so that no two are called the same.
        """
        parent = written.parent
        if self.training_directory is None or parent == self.training_directory:
            return written.name
        try:
            name = parent.relative_to(self.training_directory).as_posix()
        except ValueError:
            name = parent.name
        alone = sum(is_run_directory(child) for child in parent.iterdir()) == 1
        return name if alone else f"{name}/{written.name}"

    def stop(self, directory: Path, user: str | None = None) -> RunSummary:
        """Stop the run in *directory* if it is running, and return how it stands.

        A run am-seg did not start has no process here to end, and is left as it is.
        """
        if not is_run_directory(directory):
            stop_training(directory, user=user or self._user())
        return self.summary(directory)

    # -- the epoch chosen as best ---------------------------------------------

    def choose_best(self, directory: Path, epoch: int, user: str | None = None) -> RunSummary:
        """Record *epoch* as the best of the run in *directory*, and return the run.

        Raises:
            ValueError: If the run cannot be read or did not validate *epoch*.
        """
        run = self.summary(directory)
        if run.problem or run.written is None:
            raise ValueError(run.problem or f"{run.name} has validated no epoch yet")
        choose_best(directory, run.written, epoch, user=user or self._user())
        return self.summary(directory)

    def clear_best(self, directory: Path) -> RunSummary:
        """Take back the choice for the run in *directory*, and return the run."""
        clear_chosen_best(directory)
        return self.summary(directory)

    # -- what a run produced --------------------------------------------------

    def copy_checkpoint(self, checkpoint: Checkpoint, destination: Path) -> Path:
        """Copy a checkpoint out of its run to *destination*, to predict with or share.

        Copied, never moved: the run directory stays as the run left it.

        Raises:
            ValueError: If the checkpoint was cut short by a stop and is not whole.
            FileExistsError: If *destination* exists, or appears meanwhile.
        """
        if not checkpoint.whole():
            raise ValueError(
                f"{checkpoint.path.name} is incomplete: the run was stopped while writing it"
            )
        if destination.exists():
            raise FileExistsError(f"{destination} already exists; nothing was written")
        landed = write_atomically(
            destination,
            lambda temporary: shutil.copyfile(checkpoint.path, temporary),
            commit_if=lambda _temporary: not destination.exists(),
        )
        if not landed:
            raise FileExistsError(
                f"{destination} appeared while it was being written; nothing was written"
            )
        logger.info("Copied %s to %s", checkpoint.path, destination)
        return destination

    def other_checkpoints(
        self, directory: Path, keep: Collection[int], keep_run_best: bool = False
    ) -> list[Checkpoint]:
        """Return the checkpoints of the run in *directory* that :meth:`remove_other_checkpoints` would delete.

        **The epoch chosen as best is never among them**, whatever *keep* says:
        the choice is cleared first by whoever means to delete it.

        Args:
            keep_run_best: Whether the epoch the run counts as best is kept as well.

        Raises:
            ValueError: If the run may still be training, no epoch is kept, a
                kept epoch has no checkpoint, or the choice cannot be read.
                Nothing is deleted on any of them.
        """
        chosen = read_chosen_best(directory)
        run = self.summary(directory)
        if run.problem:
            raise ValueError(run.problem)
        if run.live:
            raise ValueError(
                f"{run.name} may still be training, or being validated; its checkpoints are "
                f"kept until that has ended"
            )
        kept = set(keep)
        if not kept:
            raise ValueError("Choose at least one checkpoint to keep")
        checkpoints = run.written.checkpoints if run.written else ()
        missing = sorted(kept - {checkpoint.epoch for checkpoint in checkpoints})
        if missing:
            raise ValueError(
                f"{run.name} has no checkpoint for epoch(s) {', '.join(map(str, missing))}"
            )
        if chosen is not None:
            kept.add(chosen.epoch)
        if keep_run_best and run.best is not None:
            kept.add(run.best.epoch)
        return [checkpoint for checkpoint in checkpoints if checkpoint.epoch not in kept]

    def remove_other_checkpoints(
        self, directory: Path, keep: Collection[int], keep_run_best: bool = False
    ) -> list[Path]:
        """Delete every checkpoint of the run in *directory* but those of the epochs in *keep*.

        **Never called without the person having been shown what goes and saying yes**:
        a deleted checkpoint cannot be got back. Only that run's are touched, and
        its metrics, lists and records stay, so the run still reads as it did.

        Returns:
            The files deleted.

        Raises:
            ValueError: As :meth:`other_checkpoints`, before anything is deleted.
        """
        doomed = [
            checkpoint.path
            for checkpoint in self.other_checkpoints(directory, keep, keep_run_best)
        ]
        for path in doomed:
            path.unlink()
            logger.info("Deleted the checkpoint %s", path)
        return doomed

    # -- internals ------------------------------------------------------------

    def _user(self) -> str | None:
        """Return the name the window records under, if it has one."""
        return self._attribution.recorder.by if self._attribution is not None else None

    def _start_path(self) -> Path | None:
        """Return the scanned folder, which is where the collection is whatever a manifest says."""
        return self.collection

    def _collection_unknown(self) -> bool:
        """Return whether nothing says where the chosen training set's collection is."""
        if self.manifest is None or self.collection is not None:
            return False
        try:
            return read_training_manifest(self.manifest).collection is None
        except ValueError:
            return False

    def _taken_names(self) -> set[str]:
        """Return the name of everything in the training directory, run or not."""
        directory = self.training_directory
        if directory is None or not directory.is_dir():
            return set()
        return {child.name for child in directory.iterdir()}

    def _steady(self, run: TrainingRun) -> TrainingRun:
        """Return *run* with any list caught mid-rewrite replaced by the one last read whole."""
        earlier = self._whole.get(run.directory)
        if run.unreadable and earlier is not None:
            run = replace(
                run,
                train=run.train or earlier.train,
                val=run.val or earlier.val,
            )
        if run.train or run.val:
            self._whole[run.directory] = run
        return run


def _chosen(directory: Path) -> ChosenBest | None:
    """Return the epoch chosen as best for a listing, a choice that cannot be read logged as none."""
    try:
        return read_chosen_best(directory)
    except ValueError as error:
        logger.warning("%s", error)
        return None


def _when(run: RunSummary) -> str:
    """Return when *run* began, as text that sorts by time, or "" where nothing says."""
    if run.record is not None and run.record.created_at:
        return run.record.created_at
    if run.written is not None and run.written.started is not None:
        return f"{run.written.started:%Y-%m-%dT%H:%M:%SZ}"
    return ""


def _written_recently(directory: Path) -> bool:
    """Return whether anything in *directory* was written within :data:`RECENT_SECONDS`."""
    try:
        newest = max((child.stat().st_mtime for child in directory.iterdir()), default=0.0)
    except OSError:
        return False
    return time.time() - newest < RECENT_SECONDS
