"""Parse, dispatch, return an exit code.

The work every command does lives in ``operations/`` and above; this is the
assembly. **Groups are registered in workflow order** and ``--help`` lists each
under its heading. A group module declares its heading, subparsers and handlers
together, so the help, the parser and the dispatch table cannot disagree about
what exists or where it belongs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from .. import __version__
from ..history import CONFIG_DISPLAY, USER_ENV_VAR, USER_FILENAME
from ..io.path import NoRelativePath
from ..parallel import MAX_DEFAULT_WORKERS, WORKERS_ENV_VAR
from . import adding, curating, installing, quantifying, recording, restoring
from .logs import DEFAULT_GUI_LOG, LOG_FILE_MAX_BYTES, open_log_file
from .scope import report_refused

logger = logging.getLogger(__name__)

#: The command groups, in the order ``am-seg --help`` lists them: the order of
#: the workflow, then setting this computer up, which is no step of it. Adding a group is a line here and nothing else -- its heading,
#: its subparsers and its handlers all come from the module.
GROUPS = (adding, curating, quantifying, recording, restoring, installing)

#: Which heading each command is listed under, built from the groups so the
#: help and the modules cannot disagree about what belongs where.
HEADINGS: dict[str, str] = {
    name: group.TITLE for group in GROUPS for name in group.COMMANDS
}


class GroupedHelp(argparse.HelpFormatter):
    """Lists the commands under their group's heading rather than in one run.

    Twenty-nine commands in a flat list is a list nobody reads to the end, and
    the order alone was carrying which step of the workflow each belongs to.
    Reaching into ``argparse``'s private ``_SubParsersAction`` is the only way:
    there is no public API for grouping subcommands, and the alternative --
    writing the list into the epilog -- prints every command twice.
    """

    def _format_action(self, action: argparse.Action) -> str:
        """Insert a heading before the first command of each group."""
        if not isinstance(action, argparse._SubParsersAction):
            return super()._format_action(action)
        parts: list[str] = []
        heading: str | None = None
        for command in action._get_subactions():
            # An alias shares its parser's position, and argparse has already
            # folded it into that parser's row as "name (alias)".
            group = HEADINGS.get(command.dest, heading)
            if group != heading:
                parts.append(f"\n  {group}\n")
                heading = group
            parts.append(super()._format_action(command))
        return "".join(parts)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for every subcommand.

    Kept separate from :func:`main` so the parser can be inspected without
    running a command, which is what lets the help text be tested.
    """
    parser = argparse.ArgumentParser(
        "am-seg",
        description="Manage Autolamella segmentation projects",
        epilog=(
            f"A scan reads image headers in worker processes: every CPU this "
            f"process may use but two, at most {MAX_DEFAULT_WORKERS}. Set "
            f"{WORKERS_ENV_VAR} to a whole number to use another count."
        ),
        formatter_class=GroupedHelp,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--log-file", default=None, metavar="PATH",
        help=(
            "Also write the log to this file, and write a native stack trace "
            "there if the process crashes outright. Appends, so runs "
            f"accumulate. 'gui' already does this in {CONFIG_DISPLAY}/"
            f"{DEFAULT_GUI_LOG.name}; name a path here to send one session "
            "somewhere of its own"
        ),
    )
    parser.add_argument(
        "--user", default=None, metavar="NAME",
        help=(
            "Record changes under this name. Otherwise taken from "
            f"{USER_ENV_VAR}, then from {CONFIG_DISPLAY}/{USER_FILENAME}; with "
            "none of those, changes are recorded under no name"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"am-seg {__version__}",
        help="Print the installed version and exit",
    )
    # Subcommands are registered in workflow order, because argparse lists them
    # in registration order and that listing is how the commands are found.
    # A metavar rather than the default: without one, argparse prints all
    # twenty-nine names into the usage line as a single brace-list that wraps
    # over four lines and says nothing.
    sub = parser.add_subparsers(dest="command", metavar="<command>", title="commands")

    for group in GROUPS:
        group.add_arguments(sub)

    return parser


#: Every subcommand and what runs it, collected from the same modules that
#: registered them. A duplicate name would mean two groups claiming one
#: command, which `test_no_two_groups_claim_one_command` refuses.
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    name: handler
    for group in GROUPS
    for name, handler in group.COMMANDS.items()
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate operation.

    Takes *argv* so the suite can drive a command the way a shell would, which
    is how all but ``gui`` and ``predict`` are covered.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # -- configure logging --------------------------------------------------
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    # No console at all when started by a launcher through pythonw on Windows.
    handlers: list[logging.Handler] = (
        [logging.StreamHandler(sys.stdout)] if sys.stdout is not None else []
    )
    # The window logs without being asked to. It is the one command whose
    # failures are not visible where it was started: it detaches from the
    # terminal on some platforms, and a crash that kills the process takes
    # anything stderr had not flushed with it. Every other command prints where
    # you can see it, and writes a file only when you name one.
    chosen = getattr(args, "log_file", None)
    log_file = Path(chosen) if chosen else None
    cap = None
    if log_file is None and args.command == "gui":
        log_file, cap = DEFAULT_GUI_LOG, LOG_FILE_MAX_BYTES

    log_failure: OSError | None = None
    if log_file is not None:
        try:
            handlers.append(open_log_file(log_file, max_bytes=cap))
        except OSError as error:
            # Never fatal. A command that will not run because it could not
            # open its own log is worse than one that runs without a log.
            log_failure, log_file = error, None

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )
    if log_file is not None:
        logger.info("Logging to %s", log_file)
    if log_failure is not None:
        logger.warning("Could not open the log file: %s", log_failure)

    # -- dispatch -----------------------------------------------------------
    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return handler(args)
    except FileExistsError as error:
        # An output that appeared after the command checked for it. Only a
        # command offering --overwrite refuses one; from any other it is a fault.
        if not hasattr(args, "overwrite"):
            raise
        report_refused(error)
        return 2
    except NoRelativePath as error:
        logger.error("Could not export: %s", error)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI runner
    raise SystemExit(main())
