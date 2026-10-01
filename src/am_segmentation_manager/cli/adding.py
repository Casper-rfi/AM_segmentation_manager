"""Adding data to the collection, and describing it.

Finding out what the conventions make of a directory tree, naming its datasets
as projects, and filling in the metadata every export later reads. Nothing here
predicts, labels or exports.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from ..discovery import describe_dataset_candidates, find_project_roots, project_markers
from ..families import StaleFamilies
from ..filtering import matches_lamella, matches_project
from ..io.path import make_path_relative
from ..metadata import DESCRIPTIVE_FIELDS, TODO_MARKER, VALUE_SEPARATOR, field_named
from ..metadata import DESCRIPTIVE_FIELDS as FIELDS
from ..operations.families_edit import (
    FamiliesValidationError,
    apply_families_change,
    describe_lamella_sections,
    plan_families_change,
    summarise_families,
)
from ..operations.init_project import init_projects, project_placement_problem, read_defaults
from ..operations.metadata_edit import MetadataValidationError, update_project_metadata
from ..reporting import print_project_summary, print_survey
from ..scanner import Scanner
from ..state import StaleState
from ..survey import DEFAULT_MAX_DEPTH as SURVEY_MAX_DEPTH
from ..survey import survey_collection
from .scope import _recorder, _writing_recorder

logger = logging.getLogger(__name__)


#: The heading ``am-seg --help`` files these under.
TITLE = "Add data to the collection"


def add_arguments(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the commands that add data and describe it."""


    # -- inspect-layout -----------------------------------------------------
    inspect_layout = sub.add_parser(
        "inspect-layout",
        help="Report how a directory tree is interpreted, without changing anything",
        description=(
            "Survey a collection of datasets and report the projects found, how "
            "their lamella and stage hierarchy was interpreted, which image sets "
            "exist, where each one's labels resolve to and whether they are "
            "shared, plus anything that does not fit the naming conventions. "
            "Read-only: nothing is created, moved or modified. Use it to survey "
            "older datasets, to check bulk filesystem work such as reorganising "
            "labels, and to debug how a directory is being read."
        ),
    )
    inspect_layout.add_argument(
        "start_path", nargs="?", default=".", help="Directory to survey (default: current directory)",
    )
    inspect_layout.add_argument(
        "--detail", action="store_true",
        help="Show the full lamella/stage/image-set tree instead of per-project totals",
    )
    inspect_layout.add_argument(
        "--max-depth", type=int, default=SURVEY_MAX_DEPTH,
        help=(
            "How deep to search for projects and stages "
            f"(default: {SURVEY_MAX_DEPTH}); raise it only for unusually nested trees"
        ),
    )
    inspect_layout.add_argument(
        "--severity", choices=["error", "warning", "info"], default="info",
        help="Lowest finding severity to report (default: info, meaning all)",
    )
    inspect_layout.add_argument(
        "--no-label-check", action="store_true",
        help=(
            "Skip reading labels to tell an annotation from an empty "
            "placeholder. The check costs about 2.5 ms a label; skipping it "
            "makes a survey of a very large collection structure-only"
        ),
    )

    # -- init-project -------------------------------------------------------
    init_project_cmd = sub.add_parser(
        "init-project",
        help="Write an AM_metadata.yaml template into a project",
        description=(
            "Create an AM_metadata.yaml template holding the descriptive "
            "metadata Autolamella does not record (sample, user, organisation "
            "and so on). Values that can be derived are filled in; the rest are "
            "written as TODO placeholders for you to complete. A dataset with no "
            "experiment.yaml also becomes indexable, because AM_metadata.yaml is "
            "itself a project marker. Existing files are never overwritten."
        ),
    )
    init_project_cmd.add_argument(
        "start_path", nargs="?", default=".",
        help="Project directory, or a directory to search with --recursive",
    )
    init_project_cmd.add_argument(
        "--recursive", action="store_true",
        help=(
            "Write a template into every project found under the directory, and "
            "into any directory that holds images but is not a project yet"
        ),
    )
    init_project_cmd.add_argument(
        "--defaults", default=None,
        help=(
            "YAML file of field values shared by these projects (for example "
            "organisation and user), copied into each generated file. Fields "
            "that vary between projects, such as sample and description, are "
            "never defaulted and stay as placeholders"
        ),
    )
    init_project_cmd.add_argument(
        "--dry-run", action="store_true",
        help="Report which files would be written without creating anything",
    )

    # -- show-metadata / edit-metadata ---------------------------------------
    show_metadata = sub.add_parser(
        "show-metadata",
        help="Show each project's resolved metadata, and where each value came from",
        description=(
            "Print the descriptive metadata in effect for each project, with the "
            "file that supplied each value. Values in AM_metadata.yaml override "
            "experiment.yaml per key, so a surprising value is easiest to "
            "explain by seeing which file it came from. Read-only."
        ),
    )
    show_metadata.add_argument("start_path", nargs="?", default=".", help="Directory to scan")
    show_metadata.add_argument(
        "--project", action="append", default=None,
        help="Project name, relative path or scan number; repeatable. Default: all",
    )

    edit_metadata = sub.add_parser(
        "edit-metadata",
        help="Set or clear fields in a project's AM_metadata.yaml",
        description=(
            "Edit the descriptive metadata of one or more projects. Only "
            "AM_metadata.yaml is written; experiment.yaml belongs to Autolamella "
            "and is never touched. The file is edited in place, so comments, key "
            "order and any keys this package does not model are preserved. "
            "Nothing is written unless every requested edit is valid."
        ),
    )
    edit_metadata.add_argument("start_path", nargs="?", default=".", help="Directory to scan")
    edit_metadata.add_argument(
        "--project", action="append", default=None,
        help="Project name, relative path or scan number; repeatable",
    )
    edit_metadata.add_argument(
        "--all", action="store_true",
        help=(
            "Apply to every project found. Values are copied into each file "
            "rather than resolved from a shared one, because nothing is "
            "inherited from parent directories"
        ),
    )
    edit_metadata.add_argument(
        "--set", action="append", default=[], metavar="FIELD=VALUE",
        help=(
            f"Set a field; repeatable. Fields: {', '.join(DESCRIPTIVE_FIELDS)}. "
            f"A field holding several values takes them separated by "
            f"'{VALUE_SEPARATOR.strip()}', or one --set apiece"
        ),
    )
    edit_metadata.add_argument(
        "--clear", action="append", default=[], metavar="FIELD",
        help=(
            "Remove a field from AM_metadata.yaml, so experiment.yaml's value "
            "applies again; repeatable"
        ),
    )
    edit_metadata.add_argument(
        "--todo", action="append", default=[], metavar="FIELD",
        help=(
            "Mark a field as outstanding work. Different from --clear: an "
            "unfilled placeholder is reported by inspect-layout"
        ),
    )
    edit_metadata.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing anything",
    )

    # -- show-families / edit-families ---------------------------------------
    show_families = sub.add_parser(
        "show-families",
        help="Show each project's families of imaging conditions",
        description=(
            "Print each project's families: the imaging conditions sharing one "
            "label, and how many stages hold each. A condition marked * is placed "
            "in its family by the project's AM_families.yaml rather than by its "
            "folder name, and a lamella's own entries are listed after. Read-only."
        ),
    )
    show_families.add_argument("start_path", nargs="?", default=".", help="Directory to scan")
    show_families.add_argument(
        "--project", action="append", default=None,
        help="Project name, relative path or scan number; repeatable. Default: all",
    )

    edit_families = sub.add_parser(
        "edit-families",
        help="Move imaging conditions between families in AM_families.yaml",
        description=(
            "Correct which family a condition belongs to, and so which label it "
            "shares, without renaming any folder. The change is checked in every "
            "stage first: it is refused if a label or prediction would be left "
            "unread, replaced by another, or taken from a condition the change "
            "does not name. A condition joining a family shares its labels and "
            "state; one leaving starts unlabelled and keeps a deselection. With "
            "--lamella, the entries are that lamella's own, replacing the "
            "project's there alone. "
            "Shows what would change; nothing is written without --apply."
        ),
    )
    edit_families.add_argument("start_path", nargs="?", default=".", help="Directory to scan")
    edit_families.add_argument(
        "--project", action="append", default=None, required=True,
        help="Project name, relative path or scan number; repeatable",
    )
    edit_families.add_argument(
        "--set", action="append", default=[], metavar="CONDITION=FAMILY",
        help="Place a condition in a family, e.g. sem2kVb=sem-2kV; repeatable",
    )
    edit_families.add_argument(
        "--clear", action="append", default=[], metavar="CONDITION",
        help="Return a condition to the family its folder name gives; repeatable",
    )
    edit_families.add_argument(
        "--primary", action="append", default=[], metavar="FAMILY=CONDITION",
        help="Name the condition a family is predicted from; repeatable",
    )
    edit_families.add_argument(
        "--clear-primary", action="append", default=[], metavar="FAMILY",
        help="Return a family to the default primary condition; repeatable",
    )
    edit_families.add_argument(
        "--lamella", default=None, metavar="LAMELLA",
        help=(
            "Change one lamella's entries, by name or number, over the project's. "
            "--clear and --clear-primary then return it to the project's entry"
        ),
    )
    edit_families.add_argument(
        "--confirm-corrected", action="store_true",
        help="Confirm images leaving a label group declared corrected",
    )
    edit_families.add_argument(
        "--apply", action="store_true",
        help="Write the change. Without it, only show what would change",
    )

    # -- list-projects / scan -----------------------------------------------
    list_projects = sub.add_parser(
        "list-projects",
        aliases=["scan"],
        help="Scan a directory and list discovered projects",
        description="Recursively scan for Autolamella projects and print a project summary",
    )
    list_projects.add_argument("start_path", nargs="?", default=".", help="Directory to start scanning from")



# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------

def _cmd_list_projects(args: argparse.Namespace) -> int:
    """Handle ``am-seg list-projects``."""
    scanner = Scanner(Path(args.start_path))
    index = scanner.build_index()
    for project in index.all_projects():
        print_project_summary(project)
    return 0


def _cmd_inspect_layout(args: argparse.Namespace) -> int:
    """Handle ``am-seg inspect-layout``."""
    survey = survey_collection(
        Path(args.start_path),
        max_depth=args.max_depth,
        check_labels=not args.no_label_check,
    )
    print_survey(survey, detail=args.detail, min_severity=args.severity)
    # A non-zero exit lets the command be used as a check in a script,
    # without making warnings -- which are expected on legacy data -- fail.
    return 1 if survey.finding_counts()["error"] else 0



def _cmd_init_project(args: argparse.Namespace) -> int:
    """Handle ``am-seg init-project``."""

    start_path = Path(args.start_path)
    if args.recursive:
        existing = find_project_roots(start_path)
        # Datasets predating Autolamella have no marker, so project
        # discovery cannot see them -- which would leave the ones that most
        # need a marker out of the command that writes one.
        candidates = describe_dataset_candidates(start_path)
        # A candidate holding datasets of several layout eras is a folder
        # someone gathered them in, and marking it would make each of them
        # a lamella of one project. Which level is right is not decidable
        # here, so this reports it rather than guessing: name the datasets
        # inside it, or pick them out in the window.
        gathering = [item for item in candidates if item.holds_several_datasets]
        for item in gathering:
            logger.warning(
                "%s holds %d dataset(s) laid out %d different ways; skipped, "
                "because marking it would make each of them a lamella of one "
                "project. Run init-project on each dataset inside it instead.",
                make_path_relative(item.path, start_path),
                len(item.lamellae), len(item.stage_depths),
            )
        proposed = [item for item in candidates if not item.holds_several_datasets]
        for item in proposed:
            logger.info(
                "%s holds images but is not a project yet; a metadata file "
                "makes it one", make_path_relative(item.path, start_path),
            )
        roots = [*existing, *(item.path for item in proposed)]
        if not roots:
            logger.error("No projects or unindexed datasets found under %s", start_path)
            return 2
    else:
        problem = project_placement_problem(start_path)
        if problem is not None:
            logger.error(
                "%s Use --recursive to write a file into every project that "
                "can be identified unambiguously, or name each dataset inside "
                "it in turn.",
                problem,
            )
            return 2
        roots = [start_path]
        if not project_markers(start_path):
            logger.info(
                "%s has no project marker yet; writing one makes it a project",
                start_path,
            )

    try:
        defaults = read_defaults(Path(args.defaults)) if args.defaults else None
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    results = init_projects(roots, defaults=defaults, dry_run=args.dry_run)
    for result in results:
        location = make_path_relative(result.path, start_path)
        if result.skipped_existing:
            logger.info("Kept existing %s", location)
        elif args.dry_run:
            logger.info("Would write %s (%d field(s) to fill in)", location, len(result.todo))
        else:
            logger.info("Wrote %s (%d field(s) to fill in)", location, len(result.todo))

    written = sum(1 for result in results if result.written)
    skipped = sum(1 for result in results if result.skipped_existing)
    remaining = sum(len(result.todo) for result in results if not result.skipped_existing)
    logger.info(
        "Total: %d project(s), %d %s, %d already had one, %d field(s) to fill in",
        len(results), written or (len(results) - skipped if args.dry_run else 0),
        "would be written" if args.dry_run else "written", skipped, remaining,
    )
    return 0



def _cmd_metadata(args: argparse.Namespace) -> int:
    """Handle ``am-seg edit-metadata``."""

    start_path = Path(args.start_path)
    projects = Scanner(start_path).build_index().all_projects()
    selectors = args.project
    if selectors is not None:
        projects = [p for p in projects if matches_project(p, selectors)]
    if not projects:
        logger.error("No projects matched under %s", start_path)
        return 2

    if args.command == "show-metadata":
        for project in projects:
            resolved = project.experiment_metadata
            print(f"{project.display_name}")
            if resolved is None:
                print("    (no metadata)")
                continue
            for field in FIELDS:
                value = getattr(resolved, field)
                source = resolved.sources.get(field)
                origin = f"  [{source.name}]" if source else ""
                if value is None:
                    state = "TODO" if field in resolved.todo_fields else "-"
                    print(f"    {field:14} {state}")
                else:
                    print(f"    {field:14} {value}{origin}")
        return 0

    if args.project is None and not args.all:
        logger.error("Choose projects with --project, or all of them with --all")
        return 2

    values: dict[str, str | None] = {}
    for item in args.set:
        if "=" not in item:
            logger.error("--set expects FIELD=VALUE, got %r", item)
            return 2
        field, value = item.split("=", 1)
        field, value = field.strip(), value.strip()
        # Repeating --set for a field that holds several values adds to it
        # rather than replacing what the earlier one asked for, so two
        # organisations need no quoting or separator on the command line.
        spec = field_named(field)
        if spec is not None and spec.multi and values.get(field):
            value = f"{values[field]}{VALUE_SEPARATOR}{value}"
        values[field] = value
    for field in args.clear:
        values[field.strip()] = None
    for field in args.todo:
        values[field.strip()] = TODO_MARKER
    if not values:
        logger.error("Nothing to do: pass --set, --clear or --todo")
        return 2

    # Validate against every project before writing to any of them, so a
    # bad field name cannot leave half the collection edited.
    try:
        planned = [
            update_project_metadata(
                project.path, values, dry_run=True, project_name=project.display_name
            )
            for project in projects
        ]
    except MetadataValidationError as exc:
        for problem in exc.problems:
            logger.error("%s", problem)
        return 2

    changed = [change for change in planned if not change.is_empty]
    for change in planned:
        for line in change.describe():
            print(line)
    if not changed:
        logger.info("Nothing to change across %d project(s)", len(projects))
        return 0
    if args.dry_run:
        logger.info(
            "Dry run: %d of %d project(s) would change", len(changed), len(projects)
        )
        return 0

    recorder = _recorder(args)
    for project in projects:
        update_project_metadata(
            project.path, values, project_name=project.display_name, recorder=recorder
        )
    logger.info("Updated %d of %d project(s)", len(changed), len(projects))
    return 0




def _cmd_show_families(args: argparse.Namespace) -> int:
    """Handle ``am-seg show-families``."""
    start_path = Path(args.start_path)
    projects = Scanner(start_path).build_index().all_projects()
    if args.project is not None:
        projects = [p for p in projects if matches_project(p, args.project)]
    if not projects:
        logger.error("No projects matched under %s", start_path)
        return 2
    for project in projects:
        print(project.display_name)
        summaries = summarise_families(project)
        if not summaries:
            print("    (no image folders)")
        for summary in summaries:
            for line in summary.describe():
                print(f"  {line}")
        for line in describe_lamella_sections(project):
            print(f"  {line}")
    return 0


def _pairs(items: list[str], flag: str) -> dict[str, str] | None:
    """Return ``NAME=VALUE`` arguments as a mapping, or None if one is malformed."""
    pairs: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name.strip() or not value.strip():
            logger.error("%s expects NAME=VALUE, got %r", flag, item)
            return None
        pairs[name.strip()] = value.strip()
    return pairs


def _cmd_edit_families(args: argparse.Namespace) -> int:
    """Handle ``am-seg edit-families``."""
    start_path = Path(args.start_path)
    projects = [
        project for project in Scanner(start_path).build_index().all_projects()
        if matches_project(project, args.project)
    ]
    if not projects:
        logger.error("No projects matched under %s", start_path)
        return 2

    families_set = _pairs(args.set, "--set")
    primary_set = _pairs(args.primary, "--primary")
    if families_set is None or primary_set is None:
        return 2
    families: dict[str, str | None] = {**families_set}
    families.update({name.strip(): None for name in args.clear})
    primary: dict[str, str | None] = {**primary_set}
    primary.update({name.strip(): None for name in args.clear_primary})
    if not families and not primary:
        logger.error("Nothing to do: pass --set, --clear, --primary or --clear-primary")
        return 2

    # Every project is planned before any is written, so one refusal leaves
    # the whole collection as it was.
    changes = []
    refused = False
    for project in projects:
        lamella = None
        if args.lamella is not None:
            matched = [
                candidate for candidate in project.lamellae
                if matches_lamella(candidate, [args.lamella])
            ]
            if len(matched) != 1:
                refused = True
                logger.error(
                    "%s: --lamella %s matches %d lamellae, not one",
                    project.display_name, args.lamella, len(matched),
                )
                continue
            lamella = matched[0]
        try:
            changes.append(
                plan_families_change(
                    project, families=families, primary=primary, lamella=lamella,
                    allow_leaving_corrected=args.confirm_corrected,
                )
            )
        except FamiliesValidationError as exc:
            refused = True
            logger.error("%s: refused", project.display_name)
            for problem in exc.problems:
                logger.error("  %s", problem)
    if refused:
        return 2

    for change in changes:
        for line in change.describe():
            print(line)
    changed = [change for change in changes if not change.is_empty]
    if not changed:
        logger.info("Nothing to change across %d project(s)", len(projects))
        return 0
    if not args.apply:
        logger.info(
            "Preview only: %d of %d project(s) would change; pass --apply to write",
            len(changed), len(projects),
        )
        return 0

    roots = {change.project_root for change in changed}
    recorder = _writing_recorder(args, [project for project in projects if project.path in roots])
    try:
        for change in changed:
            apply_families_change(change, recorder=recorder)
    except (StaleFamilies, StaleState) as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Updated %d of %d project(s)", len(changed), len(projects))
    return 0


#: What each of this group's subcommands runs. Registered above and
#: dispatched here, so the two cannot name different sets.
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "list-projects": _cmd_list_projects,
    "scan": _cmd_list_projects,
    "inspect-layout": _cmd_inspect_layout,
    "init-project": _cmd_init_project,
    "edit-metadata": _cmd_metadata,
    "show-metadata": _cmd_metadata,
    "show-families": _cmd_show_families,
    "edit-families": _cmd_edit_families,
}
