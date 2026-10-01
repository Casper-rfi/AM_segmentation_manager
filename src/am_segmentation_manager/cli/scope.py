"""The argument vocabulary every command group shares.

Where to look, which series and beam, which projects and lamellae, the
metadata filters, and whether an output may be replaced -- declared once so two
commands cannot offer the same flag under different names or defaults, and
resolved once so they cannot read it differently either.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from ..filtering import (
    METADATA_FILTERS,
    NO_TASK,
    filter_property,
    iter_matching_records,
)
from ..history import Recorder
from ..io.tables import refuse_existing
from ..layout import LABELLED_SERIES
from ..models import Project
from ..operations.init_history import writing_recorder
from ..operations.label_groups import record_from_image_path
from ..scanner import Scanner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

#: What ``--series`` defaults to for a command that acts, exports or measures.
#:
#: Autolamella writes reference and alignment images alongside the Adaptive
#: Polishing ones, and indexing them must not change what an existing command
#: prints or writes. So a command that *reports* on the collection covers every
#: series -- ``inspect-layout`` and ``stats`` -- and every other command stays
#: on Adaptive Polishing until told otherwise. Widening is one flag.
DEFAULT_SERIES: str = "adaptive"

#: What ``--series`` defaults to for a command whose job is to report what is
#: there. Narrowing these silently is the bug this work exists to fix.
REPORTING_SERIES: str = "all"


def _series_argument_type(choices: tuple[str, ...]) -> Callable[[str], str]:
    """Return the parser for one ``--series`` value, accepting only *choices*.

    **A command that cannot act on a series does not offer it**, rather than
    accepting the word and failing once a scan has run. ``other`` has no label
    directory and no predictions directory, so pointing ``predict`` or
    ``correct`` at it can only end in a refusal; refusing at the argument is
    the same answer given before any work, in the words argparse uses for every
    other bad value.

    One value may name several series comma-separated, which is why this is a
    ``type`` rather than argparse's own ``choices``.
    """
    def parse(raw: str) -> str:
        named = {value.strip() for value in raw.split(",") if value.strip()}
        unknown = sorted(named - set(choices) - {"all"})
        if unknown:
            raise argparse.ArgumentTypeError(
                f"{', '.join(unknown)}; this command covers "
                f"{', '.join(choices)} or 'all'"
            )
        return raw

    return parse


def _add_scope_arguments(
    parser: argparse.ArgumentParser,
    default_modality: str | None = None,
    include_state: bool = True,
    default_series: str = DEFAULT_SERIES,
    series_choices: tuple[str, ...] = LABELLED_SERIES,
) -> None:
    """Add the common scope-selection arguments to *parser*.

    Set *include_state* to False when the caller registers its own
    ``--state`` argument (for example to make it required). Set
    *default_series* to :data:`REPORTING_SERIES` for a command that reports on
    the collection rather than acting on part of it, and *series_choices* to
    :data:`~..layout.SERIES` where that command can also say something about
    an image with no labels.
    """
    parser.add_argument("paths", nargs="*", help="Explicit image paths or scan directory")
    parser.add_argument("--start-path", default=".", help="Directory to start scanning from")
    parser.add_argument(
        "--project", action="append", default=[],
        help="Project name or number (for example 001); use 'all' explicitly",
    )
    parser.add_argument(
        "--lamella", action="append", default=[],
        help="Lamella name/number; use 'all' explicitly",
    )
    parser.add_argument(
        "--stage", action="append", default=[],
        help="Stage name/number; use 'all' explicitly",
    )
    parser.add_argument(
        "--image", action="append", default=[],
        help="Image name/number; use 'all' explicitly",
    )
    parser.add_argument(
        "--variant", action="append", default=[],
        help=(
            "Imaging-condition variant, from the image directory name "
            "(sem-5kV_fi4 -> fi4); 'base' means a family's unsuffixed directory "
            "and 'all' every condition. Defaults to every variant"
        ),
    )
    parser.add_argument(
        "--family", action="append", default=[],
        help=(
            "Family of imaging conditions sharing a label, from the image "
            "directory name (sem-5kV_fi4 -> sem-5kV) or the project's "
            "AM_families.yaml, in any case; 'all' every family. Defaults to "
            "every family"
        ),
    )
    parser.add_argument(
        "--series", action="append", default=[],
        metavar="SERIES", type=_series_argument_type(series_choices),
        help=(
            "What the images were taken for: "
            + ", ".join(series_choices)
            + ", or 'all'. Repeat or comma-separate to include several "
            f"(default: {default_series})"
        ),
    )
    parser.set_defaults(default_series=default_series, series_choices=series_choices)
    parser.add_argument(
        "--task", action="append", default=[], metavar="TASK",
        help=(
            "Milling task the images belong to, as the project's "
            "experiment.yaml names it (Polishing, Rough Milling, MillRough); "
            "separators and case are ignored. Stages roll up into their task, "
            f"so 'Rough 1' and 'Rough 2' are both Rough Milling. '{NO_TASK}' "
            "selects the images naming no task, 'all' every image. Repeat or "
            "comma-separate to include several (default: every task)"
        ),
    )
    if include_state:
        parser.add_argument(
            "--state",
            choices=["corrected", "selected", "deselected", "all"],
            default="all",
            help="Restrict to images currently in this workflow state (default: 'all')",
        )
    if default_modality is not None:
        parser.add_argument(
            "--modality", choices=["SEM", "FIB", "all"], default=default_modality,
            help=f"Image modality to operate on (default: {default_modality})",
        )
    else:
        parser.add_argument(
            "--modality", choices=["SEM", "FIB", "all"], required=True,
            help="Image modality to operate on; specify 'all' for both SEM and FIB",
        )



def _format_duration(seconds: float) -> str:
    """Return a rough human-readable duration.

    Rough on purpose: this reports how long a run of thousands of images has
    left, where minutes are the unit anyone acts on.
    """
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _add_metadata_filters(parser: argparse.ArgumentParser) -> None:
    """Add the per-image metadata filters to *parser*.

    These read image headers, so they are applied after the structural
    selectors: only what those keep is ever opened.
    """
    for name in sorted(METADATA_FILTERS):
        parser.add_argument(
            f"--{name}", action="append", default=None, metavar="VALUE",
            help=(
                f"Keep only images whose {filter_property(name)} matches; "
                f"repeatable for alternatives"
            ),
        )


def _metadata_selectors(args: argparse.Namespace) -> dict[str, list[str]]:
    """Return the metadata filters the user asked for, if any."""
    return {
        name: values
        for name in METADATA_FILTERS
        if (values := getattr(args, name.replace("-", "_"), None))
    }


def _resolve_records(args: argparse.Namespace) -> tuple[list, Path]:
    """Resolve target records and scan start_path from CLI arguments.

    Returns a tuple of `(records, start_path)`.
    """
    records, start_path, _projects = _resolve_scope(args)
    return records, start_path


def _resolve_scope(args: argparse.Namespace) -> tuple[list, Path, list]:
    """Resolve the records, the scan root, and the projects they came from.

    The projects are what a command needs when it acts on a *dataset* rather
    than on images -- copying one out takes the files describing it along.
    Empty where explicit image paths were given, there being no scan to take
    them from.

    Returns a tuple of `(records, start_path, projects)`.
    """

    state = getattr(args, "state", "all")
    start_path = Path(args.start_path)
    file_paths: list[Path] = []

    for item in args.paths:
        p = Path(item)
        if p.is_dir():
            if args.start_path == ".":
                start_path = p
        else:
            file_paths.append(p)

    if file_paths and not any([args.project, args.lamella, args.stage, args.image]):
        records = [record_from_image_path(p) for p in file_paths]
        matching = [r for r in records if args.modality == "all" or r.modality == args.modality]
        return matching, start_path, []

    missing_scopes = [
        flag
        for flag, values in (
            ("--project", args.project),
            ("--lamella", args.lamella),
            ("--stage", args.stage),
            ("--image", args.image),
        )
        if not values
    ]
    if missing_scopes:
        logger.error(
            "Indexed operations require explicit selectors for every scope. "
            "Use 'all' where appropriate (missing: %s).",
            ", ".join(missing_scopes),
        )
        return [], start_path, []

    index = Scanner(start_path).build_index()
    records = list(
        iter_matching_records(
            index,
            project_selectors=args.project,
            lamella_selectors=args.lamella,
            stage_selectors=args.stage,
            image_selectors=args.image,
            modality=args.modality,
            state=state,
            variant_selectors=getattr(args, "variant", None) or None,
            family_selectors=getattr(args, "family", None) or None,
            series_selectors=_series_selectors(args),
            task_selectors=getattr(args, "task", None) or None,
        )
    )
    return records, start_path, index.all_projects()


def _projects_in_scope(projects: Sequence[Project], start_path: Path) -> list[Project]:
    """Return the projects :func:`_resolve_scope` found, scanning only where it found none.

    It finds none where explicit image paths were given, and a scan of the
    collection being the slowest thing the package does, a command takes the
    projects from its scope rather than scanning *start_path* a second time.
    """
    return list(projects) or Scanner(start_path).build_index().all_projects()


def _series_selectors(args: argparse.Namespace) -> list[str]:
    """Return the series to match, falling back to the command's default.

    A command declares its own default and its own vocabulary through
    :func:`_add_scope_arguments`, so which series a command covers when nobody
    says is decided where the command is registered rather than here.

    ``all`` is expanded to that vocabulary rather than passed on, because it
    means "every series this command covers" and those differ: ``stats --series
    all`` counts the ``other`` images and ``predict --series all`` must not
    reach them.
    """
    chosen = getattr(args, "series", None) or [
        getattr(args, "default_series", DEFAULT_SERIES)
    ]
    named = {value.strip() for raw in chosen for value in raw.split(",") if value.strip()}
    choices = getattr(args, "series_choices", LABELLED_SERIES)
    return list(choices) if "all" in named else chosen



#: The selectors that narrow an export, in the order they are asked for. Used
#: to write down what an export covered, so a merge can say what it is merging.
_SCOPE_SELECTORS: tuple[str, ...] = (
    "project", "lamella", "stage", "image", "variant", "family", "series", "state",
    "modality",
)


def _describe_scope(args: argparse.Namespace) -> str:
    """Return what the export was asked for, in the words the caller used."""
    parts: list[str] = []
    for name in _SCOPE_SELECTORS:
        value = getattr(args, name, None)
        if not value:
            continue
        chosen = value if isinstance(value, list) else [value]
        parts.extend(f"--{name} {item}" for item in chosen)
    if getattr(args, "paths", None):
        parts.extend(str(path) for path in args.paths)
    return " ".join(parts)


def _recorder(args: argparse.Namespace) -> Recorder:
    """Return the recorder for this run.

    One per command, so everything a single invocation changes shares a batch.
    """
    return Recorder.for_run(getattr(args, "user", None))


def _writing_recorder(args: argparse.Namespace, projects: Sequence[Project]) -> Recorder:
    """Return the recorder for a command that writes, per :func:`writing_recorder`."""
    recorder = _recorder(args)
    return writing_recorder(projects, recorder) if projects else recorder


def _add_overwrite_argument(parser: argparse.ArgumentParser) -> None:
    """Offer ``--overwrite`` on a command writing files a person may already have."""
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Replace an output file that already exists. Without it, a run that "
            "would replace one stops before reading anything and writes nothing"
        ),
    )


def _outputs_refused(outputs: Iterable[str | Path | None], overwrite: bool) -> bool:
    """Return whether an output is already there and may not be replaced, saying so."""
    try:
        refuse_existing([Path(output) for output in outputs if output], overwrite=overwrite)
    except FileExistsError as error:
        report_refused(error)
        return True
    return False


def report_refused(error: FileExistsError) -> None:
    """Say that an output was left in place rather than replaced, and how to replace it."""
    logger.error(
        "Could not export: %s. Pass --overwrite to replace it, or choose another file",
        error,
    )
