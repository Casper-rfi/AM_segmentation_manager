"""Setting this computer up to start the window like any other application.

Nothing here reads or writes a collection: ``create-shortcut`` writes into the
user's own menus and Desktop, through :mod:`~am_segmentation_manager.launcher`,
once for each environment the window is started from.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from ..launcher import (
    VIEWER,
    WINDOW,
    Launcher,
    LauncherRefused,
    check_write,
    existing,
    remove,
    this_computer,
    where_to_find,
    write,
)

logger = logging.getLogger(__name__)

#: The heading ``am-seg --help`` files these under.
TITLE = "Set up this computer"


def add_arguments(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the commands that set this computer up."""

    # -- create-shortcut ----------------------------------------------------
    shortcut = sub.add_parser(
        "create-shortcut",
        help="Add the window to this computer's applications",
        description=(
            "Add the window to the applications of this computer: ~/Applications on macOS, the "
            "applications menu on Linux, the Start menu on Windows. It uses the Python environment "
            "this command is run from, so run it again with --replace if that environment moves. "
            "With --viewer it adds the viewer instead. Only launchers written by AM-seg are "
            "replaced or removed."
        ),
    )
    shortcut.add_argument(
        "--viewer", action="store_true",
        help=(
            "Add, replace or remove the viewer launcher ('am-seg gui --viewer') instead of the "
            "window launcher"
        ),
    )
    shortcut.add_argument(
        "--desktop", action=argparse.BooleanOptionalAction, default=None,
        help="Also add one to the Desktop (default: on Windows only)",
    )
    shortcut.add_argument(
        "--replace", action="store_true",
        help="Replace a launcher AM-seg wrote before",
    )
    shortcut.add_argument(
        "--remove", action="store_true",
        help="Remove the launchers AM-seg wrote, from the menu and the Desktop, and nothing else",
    )
    shortcut.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be written or removed, without changing anything",
    )


def _cmd_create_shortcut(args: argparse.Namespace) -> int:
    """Handle ``am-seg create-shortcut``."""
    if args.remove and args.replace:
        logger.error("--remove and --replace ask for opposite things; give one")
        return 2
    try:
        computer = this_computer()
        # Removing reaches the Desktop whatever was asked, as a launcher put
        # there before is still one of ours.
        launcher = computer.launcher(
            True if args.remove else args.desktop, VIEWER if args.viewer else WINDOW
        )
        if args.remove:
            return _remove(launcher, dry_run=args.dry_run)
        if args.dry_run:
            check_write(launcher, replace=args.replace)
            for entry in launcher.entries:
                print(f"Would write {entry.path}")
            print(f"Starting {launcher.python}")
            return 0
        written = write(launcher, replace=args.replace)
    except LauncherRefused as error:
        logger.error("%s", error)
        return 1
    for path in written:
        print(f"Wrote {path}")
    print(f"Starting {launcher.python}")
    print(where_to_find(launcher.platform, launcher.app))
    return 0


def _remove(launcher: Launcher, *, dry_run: bool) -> int:
    """Remove *launcher*'s entries that am-seg wrote, or say which it would."""
    if dry_run:
        ours, others = existing(launcher)
        removed = [entry.path for entry in ours]
    else:
        removed, others = remove(launcher)
    verb = "Would remove" if dry_run else "Removed"
    for path in removed:
        print(f"{verb} {path}")
    for path in others:
        print(f"Left {path}: am-seg did not write it")
    if not removed and not others:
        print("No launcher to remove")
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "create-shortcut": _cmd_create_shortcut,
}
