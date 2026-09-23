"""Setting this computer up to start the window like any other application.

Nothing here reads or writes a collection: ``create-shortcut`` writes into the
user's own menus and Desktop, through :mod:`~am_segmentation_manager.launcher`,
once for each environment the window is started from.
"""
from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from pathlib import Path

from ..identity import PRODUCT_NAME
from ..launcher import (
    Launcher,
    LauncherRefused,
    check_write,
    existing,
    interpreter,
    places_for,
    plan,
    platform_family,
    remove,
    write,
)

logger = logging.getLogger(__name__)

#: The heading ``am-seg --help`` files these under.
TITLE = "Set up this computer"

#: Where each platform's launcher is found once written.
_WHERE_TO_FIND = {
    "darwin": (
        f"Find '{PRODUCT_NAME}' with Spotlight or in Launchpad; drag it to the Dock "
        "to keep it there."
    ),
    "linux": (
        f"'{PRODUCT_NAME}' is in the applications menu. A launcher on the GNOME "
        "Desktop must be allowed once: right-click it and choose Allow Launching."
    ),
    "win32": (
        f"'{PRODUCT_NAME}' is in the Start menu; right-click it there to pin it to "
        "the taskbar."
    ),
}


def add_arguments(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the commands that set this computer up."""

    # -- create-shortcut ----------------------------------------------------
    shortcut = sub.add_parser(
        "create-shortcut",
        help="Add the window to this computer's applications, to start it like any other",
        description=(
            "Add the window to this computer's applications: an app in "
            "~/Applications on macOS, an entry in the applications menu on "
            "Linux, and a Start menu entry on Windows. It starts the window "
            "with the Python environment this command is run from, so run it "
            "again with --replace if that environment moves. Only launchers "
            "am-seg wrote are ever replaced or removed."
        ),
    )
    shortcut.add_argument(
        "--desktop", action=argparse.BooleanOptionalAction, default=None,
        help="Put one on the Desktop as well (default: on Windows only)",
    )
    shortcut.add_argument(
        "--replace", action="store_true",
        help="Write again over a launcher am-seg wrote before",
    )
    shortcut.add_argument(
        "--remove", action="store_true",
        help="Remove the launchers am-seg wrote, from the menu and the Desktop, and nothing else",
    )
    shortcut.add_argument(
        "--dry-run", action="store_true",
        help="Say what would be written or removed, and change nothing",
    )


def _cmd_create_shortcut(args: argparse.Namespace) -> int:
    """Handle ``am-seg create-shortcut``."""
    if args.remove and args.replace:
        logger.error("--remove and --replace ask for opposite things; give one")
        return 2
    try:
        family = platform_family()
        places = places_for(family, Path.home(), os.environ)
        desktop = args.desktop if args.desktop is not None else family == "win32"
        # Removing reaches the Desktop whatever was asked, as a launcher put
        # there before is still one of ours.
        launcher = plan(family, places, interpreter(), desktop=desktop or args.remove)
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
    print(_WHERE_TO_FIND[family])
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
