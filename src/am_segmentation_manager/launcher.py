"""Launchers that start the window like any other application, one per platform.

A surface outside the four layers: it writes into the user's own menus and
Desktop, never into a collection, and reads nothing of one. A launcher is
planned whole and checked before anything is written. **An existing file is
replaced or removed only where it carries this package's mark**, so nothing
this did not write is ever touched. The window and the viewer each have a
launcher of their own, told apart by :class:`Application`, so neither is ever
mistaken for the other. A launcher starts the interpreter of the environment
that made it, and one made from an environment that moves must be made again.
"""
from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import struct
import subprocess
import sys
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Final

from .identity import (
    APPLICATION_ID,
    ICON_PATH,
    MAC_ICON_PATH,
    PRODUCT_NAME,
    VIEWER_ID,
    VIEWER_NAME,
    WINDOWS_ICON_PATH,
)

#: What the window's launcher asks the interpreter to run.
ARGUMENTS: Final[tuple[str, ...]] = ("-m", "am_segmentation_manager.cli", "gui")

#: Said beside the name: the tooltip on Windows, the comment on Linux. A
#: Windows shortcut can carry nothing else of ours, so there it is the mark,
#: and like every mark here it never changes once shipped.
DESCRIPTION: Final[str] = "Curate, proofread and export FIB/SEM segmentation labels"

#: The viewer's, and so the mark on its Windows shortcut.
VIEWER_DESCRIPTION: Final[str] = "Look through FIB/SEM images and their labels, changing nothing"

#: The key marking a macOS bundle or a Linux desktop entry as written here.
PLIST_MARK: Final[str] = "AMSegLauncher"
DESKTOP_MARK: Final[str] = "X-AM-Seg-Launcher"

#: The text chunk marking a Linux icon as written here: a PNG's own place for
#: saying what made it.
PNG_MARK: Final[bytes] = b"Software\x00am-seg create-shortcut"

#: The executable inside the macOS bundle, a shell script.
_MAC_EXECUTABLE: Final[str] = "am-seg"

#: What a menu offers the launcher as, in each platform's own words. Never
#: "shortcut": in the window that word means a key.
MENU_LABELS: Final[dict[str, str]] = {
    "darwin": "Add to Applications...",
    "linux": "Add to applications menu...",
    "win32": "Add to Start menu...",
}

#: Where each platform's launcher is found once written, for an application
#: called ``{name}``; :func:`where_to_find` fills it in.
WHERE_TO_FIND: Final[dict[str, str]] = {
    "darwin": (
        "Find '{name}' with Spotlight or in Launchpad; drag it to the Dock "
        "to keep it there."
    ),
    "linux": (
        "'{name}' is in the applications menu. A launcher on the GNOME "
        "Desktop must be allowed once: right-click it and choose Allow Launching."
    ),
    "win32": (
        "'{name}' is in the Start menu; right-click it there to pin it to "
        "the taskbar."
    ),
}


@dataclass(frozen=True)
class Application:
    """What one launcher starts, and every name the desktop knows it by.

    Attributes:
        name: What the menu shows, and what the macOS bundle and the Windows
            shortcut are called.
        identifier: The macOS bundle identifier, and the Linux desktop entry's
            and icon's names.
        description: Said beside the name, and the Windows shortcut's mark.
        arguments: What the interpreter is asked to run.
        option: What ``create-shortcut`` is given to write this one.
    """

    name: str
    identifier: str
    description: str
    arguments: tuple[str, ...]
    option: str = ""

    @property
    def bundle_name(self) -> str:
        """The macOS bundle's directory name."""
        return f"{self.name}.app"

    @property
    def desktop_entry_name(self) -> str:
        """The Linux desktop entry's file name."""
        return f"{self.identifier}.desktop"

    @property
    def shortcut_name(self) -> str:
        """The Windows shortcut's file name."""
        return f"{self.name}.lnk"

    @property
    def icon_name(self) -> str:
        """The Linux icon's file name: its own, so removing one keeps the other's."""
        return f"{self.identifier}.png"

    @property
    def command(self) -> str:
        """The command that writes this launcher again."""
        return " ".join(part for part in ("am-seg create-shortcut", self.option) if part)


#: The window, which curates, proofreads and exports.
WINDOW: Final[Application] = Application(
    PRODUCT_NAME, APPLICATION_ID, DESCRIPTION, ARGUMENTS
)

#: The window opened only to look; see ``am-seg gui --viewer``.
VIEWER: Final[Application] = Application(
    VIEWER_NAME, VIEWER_ID, VIEWER_DESCRIPTION, (*ARGUMENTS, "--viewer"), "--viewer"
)


def where_to_find(family: str, app: Application = WINDOW) -> str:
    """Say where *app*'s launcher is found on *family* once written."""
    return WHERE_TO_FIND[family].format(name=app.name)


class LauncherRefused(Exception):
    """A launcher that cannot be written or removed as asked; nothing was changed."""


@dataclass(frozen=True)
class Places:
    """Where one platform's launchers go on this computer.

    Attributes:
        menu: The directory the menu entry goes in: ``~/Applications``, the
            desktop's ``applications`` directory, or the Start menu's Programs.
        desktop: The user's Desktop.
        icons: Where a Linux icon goes; None elsewhere.
    """

    menu: Path
    desktop: Path
    icons: Path | None = None


@dataclass(frozen=True)
class Entry:
    """One thing a launcher puts in a folder: a bundle, a file, or a link.

    Attributes:
        path: Where it goes. A macOS bundle's path is its directory.
        files: What to write, relative to *path* for a bundle and keyed by
            ``""`` for a single file; empty for a Windows shortcut, which
            PowerShell writes, and for a link.
        executable: Relative paths among *files* to mark executable.
        link_to: For a link, what it points at.
    """

    path: Path
    files: Mapping[str, bytes]
    executable: tuple[str, ...] = ()
    link_to: Path | None = None


@dataclass(frozen=True)
class Launcher:
    """Everything one platform's launcher consists of, and where it goes."""

    platform: str
    python: Path
    entries: tuple[Entry, ...]
    app: Application = WINDOW


# -- what a launcher runs ----------------------------------------------------


def platform_family(platform: str = sys.platform) -> str:
    """Return ``darwin``, ``win32`` or ``linux`` for *platform*.

    Raises:
        LauncherRefused: For a platform none of the three describes.
    """
    if platform in ("darwin", "win32"):
        return platform
    if platform.startswith(("linux", "freebsd")):
        return "linux"
    raise LauncherRefused(f"No launcher is written on {platform}")


def menu_label(platform: str | None = None) -> str | None:
    """Return what a menu offers the launcher as here, or None where there is none."""
    try:
        return MENU_LABELS[platform_family(platform or sys.platform)]
    except LauncherRefused:
        return None


def interpreter(platform: str = sys.platform, executable: str = sys.executable) -> Path:
    """Return the interpreter a launcher should start.

    *executable* as given, never resolved: a virtual environment's python is a
    link, and following it leaves the environment behind. On Windows its
    ``pythonw.exe`` sibling, which opens no console window, where there is one.
    """
    python = Path(executable)
    if platform_family(platform) == "win32":
        windowed = python.with_name("pythonw.exe")
        if windowed.exists():
            return windowed
    return python


# -- what each platform's launcher holds ---------------------------------------


def mac_info_plist(app: Application = WINDOW) -> bytes:
    """Return the bundle's ``Info.plist``, naming it and its icon."""
    return plistlib.dumps({
        "CFBundleName": app.name,
        "CFBundleDisplayName": app.name,
        "CFBundleIdentifier": app.identifier,
        "CFBundleExecutable": _MAC_EXECUTABLE,
        "CFBundleIconFile": MAC_ICON_PATH.name,
        "CFBundlePackageType": "APPL",
        "NSHighResolutionCapable": True,
        PLIST_MARK: True,
    })


def mac_launch_script(python: Path, app: Application = WINDOW) -> bytes:
    """Return the bundle's executable: start *python*, or say it has gone."""
    quoted = shlex.quote(str(python))
    arguments = " ".join(app.arguments)
    return (
        "#!/bin/sh\n"
        "# Written by am-seg create-shortcut.\n"
        f"python={quoted}\n"
        'if [ ! -x "$python" ]; then\n'
        "    osascript -e 'on run argv' \\\n"
        "        -e 'display alert (item 1 of argv) message (item 2 of argv) as critical' \\\n"
        "        -e 'end run' \\\n"
        f'        "{app.name} cannot start" \\\n'
        '        "The Python it was set up with is no longer there: $python. Run '
        f"'{app.command} --replace' from the environment you use now.\"\n"
        "    exit 1\n"
        "fi\n"
        f'exec "$python" {arguments}\n'
    ).encode()


def _exec_argument(argument: str) -> str:
    """Quote one argument for a desktop entry's ``Exec`` key.

    Quoted where the specification reserves a character, then escaped again as
    a string value, which is why one backslash becomes four.
    """
    reserved = set(" \t\n\"'\\><~|&;$*?#()`")
    if any(character in reserved for character in argument):
        inner = "".join(
            "\\" + character if character in '"`$\\' else character for character in argument
        )
        argument = f'"{inner}"'
    return argument.replace("\\", "\\\\").replace("%", "%%")


def desktop_entry(python: PurePath, icon: PurePath, app: Application = WINDOW) -> bytes:
    """Return a Linux desktop entry starting *python*, drawn with *icon*.

    ``TryExec`` hides the entry once *python* is gone, rather than leaving one
    that does nothing when chosen.
    """
    command = " ".join(_exec_argument(part) for part in (str(python), *app.arguments))
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Version=1.5\n"
        f"Name={app.name}\n"
        f"Comment={app.description}\n"
        f"TryExec={python}\n"
        f"Exec={command}\n"
        f"Icon={icon}\n"
        "Terminal=false\n"
        "Categories=Science;Graphics;\n"
        f"{DESKTOP_MARK}=true\n"
    ).encode()


def marked_png(png: bytes) -> bytes:
    """Return *png* with :data:`PNG_MARK` as a text chunk after its header."""
    header_end = 8 + 4 + 4 + 13 + 4  # signature, then IHDR's length, type, data, CRC
    chunk = b"tEXt" + PNG_MARK
    marked = struct.pack(">I", len(PNG_MARK)) + chunk + struct.pack(">I", zlib.crc32(chunk))
    return png[:header_end] + marked + png[header_end:]


#: Creates one Windows shortcut. Every value arrives in an environment
#: variable, so no path is ever spliced into the script, whatever it holds.
WINDOWS_SHORTCUT_SCRIPT: Final[str] = """\
$ErrorActionPreference = 'Stop'
$link = (New-Object -ComObject WScript.Shell).CreateShortcut($env:AM_SEG_SHORTCUT)
$link.TargetPath = $env:AM_SEG_TARGET
$link.Arguments = $env:AM_SEG_ARGUMENTS
$link.IconLocation = "$($env:AM_SEG_ICON),0"
$link.Description = $env:AM_SEG_DESCRIPTION
$link.WorkingDirectory = $env:AM_SEG_WORKDIR
$link.Save()
"""

#: Names the Start menu's Programs folder and the Desktop, one per line, as
#: Windows has them -- the Desktop moved into OneDrive where it has been.
WINDOWS_PLACES_SCRIPT: Final[str] = """\
[Console]::OutputEncoding = [Text.Encoding]::UTF8
[Environment]::GetFolderPath('Programs')
[Environment]::GetFolderPath('Desktop')
"""


def windows_shortcut_environment(
    shortcut: Path, python: Path, home: Path, app: Application = WINDOW
) -> dict[str, str]:
    """Return what :data:`WINDOWS_SHORTCUT_SCRIPT` reads to write *shortcut*."""
    return {
        "AM_SEG_SHORTCUT": str(shortcut),
        "AM_SEG_TARGET": str(python),
        "AM_SEG_ARGUMENTS": " ".join(app.arguments),
        "AM_SEG_ICON": str(WINDOWS_ICON_PATH),
        "AM_SEG_DESCRIPTION": app.description,
        "AM_SEG_WORKDIR": str(home),
    }


# -- where a launcher goes -----------------------------------------------------


def plan(
    family: str,
    places: Places,
    python: Path,
    *,
    desktop: bool,
    app: Application = WINDOW,
) -> Launcher:
    """Return the launcher for *family*, writing nothing.

    Args:
        family: As :func:`platform_family` returns it.
        places: Where this computer keeps menus and the Desktop.
        python: The interpreter it starts.
        desktop: Whether a Desktop entry comes too.
        app: What it starts: the window or the viewer.
    """
    entries: list[Entry] = []
    if family == "darwin":
        bundle = places.menu / app.bundle_name
        entries.append(Entry(
            bundle,
            {
                "Contents/Info.plist": mac_info_plist(app),
                f"Contents/MacOS/{_MAC_EXECUTABLE}": mac_launch_script(python, app),
                f"Contents/Resources/{MAC_ICON_PATH.name}": MAC_ICON_PATH.read_bytes(),
            },
            executable=(f"Contents/MacOS/{_MAC_EXECUTABLE}",),
        ))
        if desktop:
            entries.append(Entry(places.desktop / app.bundle_name, {}, link_to=bundle))
    elif family == "linux":
        if places.icons is None:
            raise LauncherRefused("No icon directory to put the launcher's icon in")
        icon = places.icons / app.icon_name
        entry = desktop_entry(python, icon, app)
        entries.append(Entry(icon, {"": marked_png(ICON_PATH.read_bytes())}))
        entries.append(Entry(places.menu / app.desktop_entry_name, {"": entry}))
        if desktop:
            entries.append(
                Entry(places.desktop / app.desktop_entry_name, {"": entry}, executable=("",))
            )
    elif family == "win32":
        entries.append(Entry(places.menu / app.shortcut_name, {}))
        if desktop:
            entries.append(Entry(places.desktop / app.shortcut_name, {}))
    else:
        raise LauncherRefused(f"No launcher is written for {family}")
    return Launcher(family, python, tuple(entries), app)


def is_ours(entry: Entry, app: Application = WINDOW) -> bool:
    """Whether what is at *entry*'s path was written here for *app*, and so may be replaced."""
    path = entry.path
    if entry.link_to is not None:
        return path.is_symlink() and Path(path.readlink()) == entry.link_to
    if path.suffix == ".app":
        try:
            with (path / "Contents" / "Info.plist").open("rb") as handle:
                return plistlib.load(handle).get(PLIST_MARK) is True
        except (OSError, plistlib.InvalidFileException, AttributeError):
            return False
    try:
        content = path.read_bytes()
    except OSError:
        return False
    if path.suffix == ".desktop":
        return f"\n{DESKTOP_MARK}=true\n".encode() in content
    if path.suffix == ".lnk":
        return app.description.encode("utf-16-le") in content
    if path.suffix == ".png":
        return b"tEXt" + PNG_MARK in content
    return False


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def check_write(launcher: Launcher, *, replace: bool) -> None:
    """Refuse the whole launcher if any part of it may not be written.

    Raises:
        LauncherRefused: Naming every entry in the way, and why.
    """
    problems = []
    for entry in launcher.entries:
        if not _present(entry.path):
            continue
        if not is_ours(entry, launcher.app):
            problems.append(f"{entry.path} is already there and was not written by AM-seg")
        elif not replace:
            problems.append(f"{entry.path} is already there; --replace writes it again")
    if problems:
        raise LauncherRefused("\n".join(problems))


def write(launcher: Launcher, *, replace: bool = False, home: Path | None = None) -> list[Path]:
    """Write *launcher*, returning the entries written.

    Raises:
        LauncherRefused: Before writing anything, where :func:`check_write` does;
            or where PowerShell cannot make a Windows shortcut.
    """
    check_write(launcher, replace=replace)
    written = []
    for entry in launcher.entries:
        if _present(entry.path):
            _delete(entry)
        entry.path.parent.mkdir(parents=True, exist_ok=True)
        if entry.link_to is not None:
            entry.path.symlink_to(entry.link_to)
        elif launcher.platform == "win32":
            environment = windows_shortcut_environment(
                entry.path, launcher.python, home or Path.home(), launcher.app
            )
            run_powershell(WINDOWS_SHORTCUT_SCRIPT, environment)
        else:
            for relative, content in entry.files.items():
                target = entry.path / relative if relative else entry.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                if relative in entry.executable:
                    target.chmod(0o755)
        written.append(entry.path)
    return written


def existing(launcher: Launcher) -> tuple[list[Entry], list[Path]]:
    """Return what is already at *launcher*'s paths.

    Returns:
        ``(ours, others)``: the entries written here, and the paths holding
        something AM-seg did not write.
    """
    ours, others = [], []
    for entry in launcher.entries:
        if _present(entry.path):
            if is_ours(entry, launcher.app):
                ours.append(entry)
            else:
                others.append(entry.path)
    return ours, others


def remove(launcher: Launcher) -> tuple[list[Path], list[Path]]:
    """Remove every entry of *launcher* that was written here.

    Returns:
        ``(removed, left)``: what went, and what was at an entry's path but
        not written by AM-seg, and so stayed.
    """
    ours, others = existing(launcher)
    for entry in ours:
        _delete(entry)
    return [entry.path for entry in ours], others


def _delete(entry: Entry) -> None:
    """Delete an entry :func:`is_ours` has vouched for."""
    if entry.path.is_dir() and not entry.path.is_symlink():
        shutil.rmtree(entry.path)
    else:
        entry.path.unlink()


# -- this computer ---------------------------------------------------------------


@dataclass(frozen=True)
class ThisComputer:
    """Where launchers go on this computer, and what they start, asked once.

    Asked once because on Windows the asking is a PowerShell process.
    """

    family: str
    places: Places
    python: Path

    @property
    def desktop_by_default(self) -> bool:
        """Whether a Desktop entry comes unless asked otherwise: on Windows only."""
        return self.family == "win32"

    def launcher(self, desktop: bool | None = None, app: Application = WINDOW) -> Launcher:
        """Return *app*'s launcher for this computer, on the Desktop as well where *desktop*."""
        wanted = self.desktop_by_default if desktop is None else desktop
        return plan(self.family, self.places, self.python, desktop=wanted, app=app)


def this_computer(platform: str | None = None) -> ThisComputer:
    """Return where launchers go here, for the environment this process runs in.

    Raises:
        LauncherRefused: On a platform with no launcher, or where Windows will
            not name its folders.
    """
    platform = platform or sys.platform
    family = platform_family(platform)
    return ThisComputer(
        family, places_for(family, Path.home(), os.environ), interpreter(platform)
    )


def run_powershell(script: str, environment: Mapping[str, str] | None = None) -> str:
    """Run *script* in Windows PowerShell and return what it printed.

    Raises:
        LauncherRefused: Where PowerShell is missing or the script fails -- a
            managed machine's constrained language mode refuses COM objects.
    """
    try:
        result = subprocess.run(  # noqa: S603 - our own scripts; data goes in through the environment
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],  # noqa: S607 - Windows' own
            capture_output=True, text=True, encoding="utf-8", timeout=60,
            env={**os.environ, **(environment or {})},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LauncherRefused(f"Could not run PowerShell: {error}") from error
    if result.returncode != 0:
        raise LauncherRefused(f"PowerShell could not write the shortcut:\n{result.stderr.strip()}")
    return result.stdout


def places_for(family: str, home: Path, environ: Mapping[str, str]) -> Places:
    """Return where *family* keeps menus and the Desktop, asking the system."""
    if family == "darwin":
        return Places(menu=home / "Applications", desktop=home / "Desktop")
    if family == "linux":
        data = Path(environ.get("XDG_DATA_HOME") or home / ".local" / "share")
        return Places(
            menu=data / "applications",
            desktop=_linux_desktop(home),
            icons=data / "icons" / "hicolor" / "256x256" / "apps",
        )
    if family == "win32":
        lines = [line.strip() for line in run_powershell(WINDOWS_PLACES_SCRIPT).splitlines()]
        folders = [line for line in lines if line]
        if len(folders) != 2:
            raise LauncherRefused(f"Windows did not name its Start menu and Desktop: {lines}")
        return Places(menu=Path(folders[0]), desktop=Path(folders[1]))
    raise LauncherRefused(f"No launcher is written for {family}")


def _linux_desktop(home: Path) -> Path:
    """Return the Desktop as ``xdg-user-dir`` names it, in the user's language."""
    try:
        named = subprocess.run(
            ["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=10  # noqa: S607 - the desktop's own
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        named = ""
    return Path(named) if named else home / "Desktop"
