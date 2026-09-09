"""Parse, dispatch, return an exit code.

The work every command does lives in ``operations/`` and above; what is here is
the assembly. **The groups are registered in workflow order**, and ``--help``
lists them under a heading each, so the step a command belongs to is stated
rather than left to be inferred from what it sits next to.

A group module declares its heading, its subparsers and its handlers together.
So a command missing from one of the three is a mistake one file wide, and the
help, the parser and the dispatch table cannot disagree about what exists or
about where it belongs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from .. import __version__
from ..history import CONFIG_DISPLAY, USER_ENV_VAR, USER_FILENAME
from . import adding, curating, quantifying, recording, restoring
from .logs import DEFAULT_GUI_LOG, LOG_FILE_MAX_BYTES, open_log_file

logger = logging.getLogger(__name__)

#: The command groups, in the order ``am-seg --help`` lists them: the order of
#: the workflow. Adding a group is a line here and nothing else -- its heading,
#: its subparsers and its handlers all come from the module.
GROUPS = (adding, curating, quantifying, recording, restoring)

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

    def _format_action(self, action) -> str:
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
        "am_segmentation_manager",
        description="Manage Autolamella segmentation projects",
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
            f"{USER_ENV_VAR}, then from {CONFIG_DISPLAY}/{USER_FILENAME}, then "
            "from the account running the command"
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
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
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
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI runner
    raise SystemExit(main())
