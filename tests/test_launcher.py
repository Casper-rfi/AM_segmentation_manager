"""Launchers: what each platform's holds, and that only ours is ever replaced.

Contents are checked as data; writing, refusing and removing run against a
temporary home, never the real menus. A Windows shortcut is PowerShell's to
write, so here the script's inputs are checked and its output is stood in for.
"""
from __future__ import annotations

import configparser
import importlib
import os
import plistlib
import shlex
import struct
import subprocess
import sys
import zlib
from pathlib import Path, PurePosixPath

import pytest

from am_segmentation_manager import launcher
from am_segmentation_manager.identity import (
    APPLICATION_ID,
    PRODUCT_NAME,
    VIEWER_ID,
    VIEWER_NAME,
)
from am_segmentation_manager.launcher import (
    ARGUMENTS,
    DESCRIPTION,
    VIEWER,
    VIEWER_DESCRIPTION,
    WINDOW,
    LauncherRefused,
    Places,
    plan,
)

cli_main = importlib.import_module("am_segmentation_manager.cli.main")

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="a POSIX shell and permissions")

#: Paths a user's environment could really sit under.
AWKWARD_PATHS = [
    "/home/ada/.venv/bin/python",
    "/Users/ada/My Data/env/bin/python",
    "/Users/o'brien/env/bin/python",
    "/srv/$HOME/100%/env/bin/python",
    '/srv/a "quoted" \\ dir/python',
]


def _places(root: Path) -> Places:
    return Places(menu=root / "menu", desktop=root / "Desktop", icons=root / "icons")


# -- what a launcher runs ----------------------------------------------------------


def test_the_interpreter_is_the_environments_own_link_not_what_it_points_at(tmp_path):
    """A venv's python is a link to the base interpreter; following it would
    start Python without the package."""
    base = tmp_path / "base-python"
    base.write_text("")
    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(base)
    except OSError:
        pytest.skip("this system cannot make a link")

    assert launcher.interpreter("linux", str(link)) == link


def test_on_windows_the_interpreter_is_the_one_without_a_console(tmp_path):
    (tmp_path / "python.exe").write_text("")
    (tmp_path / "pythonw.exe").write_text("")

    assert launcher.interpreter("win32", str(tmp_path / "python.exe")) == tmp_path / "pythonw.exe"


def test_on_windows_without_pythonw_the_console_one_still_starts_it(tmp_path):
    (tmp_path / "python.exe").write_text("")

    assert launcher.interpreter("win32", str(tmp_path / "python.exe")) == tmp_path / "python.exe"


@pytest.mark.parametrize("platform, family", [
    ("darwin", "darwin"), ("win32", "win32"), ("linux", "linux"), ("freebsd14", "linux"),
])
def test_each_platform_is_one_of_three(platform, family):
    assert launcher.platform_family(platform) == family


def test_a_platform_with_no_launcher_is_refused():
    with pytest.raises(LauncherRefused):
        launcher.platform_family("aix")


# -- what each platform's launcher holds -------------------------------------------


def test_the_windows_launcher_keeps_every_name_it_shipped_with():
    """The desktop and :func:`launcher.is_ours` recognise a launcher written
    before by these, so a change would orphan every one already on a computer."""
    assert ARGUMENTS == ("-m", "am_segmentation_manager.cli", "gui")
    assert DESCRIPTION == "Curate, proofread and export FIB/SEM segmentation labels"
    assert (WINDOW.name, WINDOW.identifier, WINDOW.description, WINDOW.arguments) == (
        PRODUCT_NAME, APPLICATION_ID, DESCRIPTION, ARGUMENTS
    )
    assert (WINDOW.bundle_name, WINDOW.desktop_entry_name, WINDOW.shortcut_name, WINDOW.icon_name) == (
        "AM segmentation manager.app", "adaptive-milling-seg.desktop",
        "AM segmentation manager.lnk", "adaptive-milling-seg.png",
    )
    assert WINDOW.command == "am-seg create-shortcut"


def test_the_viewer_is_named_and_started_as_the_viewer():
    assert (VIEWER.name, VIEWER.identifier, VIEWER.description) == (
        VIEWER_NAME, VIEWER_ID, VIEWER_DESCRIPTION
    )
    assert VIEWER.arguments == (*ARGUMENTS, "--viewer")
    assert VIEWER.command == "am-seg create-shortcut --viewer"


def test_the_window_and_the_viewer_share_no_path_and_no_mark(tmp_path):
    """So that writing, replacing or removing one can never reach the other."""
    places = _places(tmp_path)
    for family in ("darwin", "linux", "win32"):
        paths = {
            app: {e.path for e in plan(family, places, Path("/env/python"), desktop=True, app=app).entries}
            for app in (WINDOW, VIEWER)
        }
        assert not paths[WINDOW] & paths[VIEWER], family
    assert WINDOW.identifier != VIEWER.identifier
    assert WINDOW.description not in VIEWER.description
    assert VIEWER.description not in WINDOW.description


def test_the_viewer_bundle_and_desktop_entry_name_the_viewer():
    info = plistlib.loads(launcher.mac_info_plist(VIEWER))
    entry = _read_desktop_entry(
        launcher.desktop_entry(PurePosixPath("/env/bin/python"), PurePosixPath("/i/v.png"), VIEWER)
    )

    assert info["CFBundleName"] == VIEWER_NAME
    assert info["CFBundleIdentifier"] == VIEWER_ID
    assert entry["Name"] == VIEWER_NAME
    assert entry["Comment"] == VIEWER_DESCRIPTION
    assert _split_exec(entry["Exec"]) == ["/env/bin/python", *VIEWER.arguments]


def test_where_a_launcher_is_found_names_its_application():
    for family in ("darwin", "linux", "win32"):
        assert f"'{VIEWER_NAME}'" in launcher.where_to_find(family, VIEWER)
        assert f"'{PRODUCT_NAME}'" in launcher.where_to_find(family)


def test_the_mac_bundle_is_named_iconed_and_marked():
    info = plistlib.loads(launcher.mac_info_plist())

    assert info["CFBundleName"] == PRODUCT_NAME
    assert info["CFBundleIdentifier"] == APPLICATION_ID
    assert info["CFBundleIconFile"] == launcher.MAC_ICON_PATH.name
    assert info[launcher.PLIST_MARK] is True


@posix_only
@pytest.mark.parametrize("app", [WINDOW, VIEWER], ids=["window", "viewer"])
@pytest.mark.parametrize("python_path", AWKWARD_PATHS)
def test_the_mac_script_starts_its_python_whatever_its_path_holds(tmp_path, python_path, app):
    """The script is run for real, with a stand-in python recording what it
    was asked to run."""
    python = tmp_path / python_path.lstrip("/")
    python.parent.mkdir(parents=True)
    record = tmp_path / "ran.txt"
    python.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {shlex.quote(str(record))}\n')
    python.chmod(0o755)
    script = tmp_path / "launch"
    script.write_bytes(launcher.mac_launch_script(python, app))

    subprocess.run(["/bin/sh", str(script)], check=True, timeout=30)

    assert record.read_text().splitlines() == list(app.arguments)


@posix_only
def test_the_mac_script_fails_rather_than_starting_nothing_when_its_python_is_gone(tmp_path):
    script = tmp_path / "launch"
    script.write_bytes(launcher.mac_launch_script(tmp_path / "gone" / "python"))

    # No PATH, so no alert is put on the screen of whoever runs the suite.
    result = subprocess.run(["/bin/sh", str(script)], env={"PATH": ""}, timeout=30)

    assert result.returncode == 1


def _read_desktop_entry(content: bytes) -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read_string(content.decode("utf-8"))
    return dict(parser["Desktop Entry"])


def _split_exec(value: str) -> list[str]:
    """Undo an ``Exec`` value as the desktop entry specification reads it: the
    string escape first, then arguments split on spaces, where one in double
    quotes may escape ``"``, backtick, ``$`` and backslash."""
    command = value.replace("\\\\", "\\")
    arguments, current, quoted, index = [], "", False, 0
    while index < len(command):
        character = command[index]
        if quoted and character == "\\" and command[index + 1] in '"`$\\':
            current += command[index + 1]
            index += 1
        elif character == '"':
            quoted = not quoted
        elif character == " " and not quoted:
            arguments.append(current)
            current = ""
        else:
            current += character
        index += 1
    arguments.append(current)
    return [argument.replace("%%", "%") for argument in arguments]


@pytest.mark.parametrize("python_path", AWKWARD_PATHS)
def test_the_desktop_entry_runs_its_python_whatever_its_path_holds(python_path):
    entry = _read_desktop_entry(launcher.desktop_entry(
        PurePosixPath(python_path), PurePosixPath("/i/icon.png")
    ))

    assert _split_exec(entry["Exec"]) == [python_path, *ARGUMENTS]


def test_the_desktop_entry_is_named_iconed_marked_and_hidden_once_its_python_goes():
    entry = _read_desktop_entry(
        launcher.desktop_entry(PurePosixPath("/env/bin/python"), PurePosixPath("/icons/am.png"))
    )

    assert entry["Name"] == PRODUCT_NAME
    assert entry["Icon"] == "/icons/am.png"
    assert entry["TryExec"] == "/env/bin/python"
    assert entry["Terminal"] == "false"
    assert entry[launcher.DESKTOP_MARK] == "true"


def _chunks(png: bytes) -> list[bytes]:
    """Return a PNG's chunk types, checking each chunk's CRC on the way."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    types, offset = [], 8
    while offset < len(png):
        (length,) = struct.unpack(">I", png[offset:offset + 4])
        body = png[offset + 4:offset + 8 + length]
        (crc,) = struct.unpack(">I", png[offset + 8 + length:offset + 12 + length])
        assert zlib.crc32(body) == crc, f"bad CRC on {body[:4]!r}"
        types.append(body[:4])
        offset += 12 + length
    return types


def test_the_linux_icon_is_the_mark_with_a_note_of_what_wrote_it():
    marked = launcher.marked_png(launcher.ICON_PATH.read_bytes())

    types = _chunks(marked)

    assert types[:2] == [b"IHDR", b"tEXt"] and types[-1] == b"IEND"
    assert b"tEXt" + launcher.PNG_MARK in marked


def test_the_windows_shortcut_is_given_every_value_by_environment_not_by_script():
    """Nothing is spliced into the script, so no path can break out of it."""
    environment = launcher.windows_shortcut_environment(
        Path("C:/Users/o'brien/Desktop/AM.lnk"), Path("C:/env/pythonw.exe"), Path("C:/Users/o'brien")
    )

    assert set(environment) == set(
        name for name in environment if f"$env:{name}" in launcher.WINDOWS_SHORTCUT_SCRIPT
    )
    assert environment["AM_SEG_ARGUMENTS"] == " ".join(ARGUMENTS)
    assert environment["AM_SEG_DESCRIPTION"] == DESCRIPTION


# -- where a launcher goes, and what may be replaced ---------------------------------


def test_the_desktop_is_only_reached_when_asked(tmp_path):
    places = _places(tmp_path)

    for family in ("darwin", "linux", "win32"):
        without = plan(family, places, Path("/env/python"), desktop=False)
        with_desktop = plan(family, places, Path("/env/python"), desktop=True)

        assert not any(places.desktop in e.path.parents for e in without.entries), family
        assert any(places.desktop in e.path.parents for e in with_desktop.entries), family


@posix_only
@pytest.mark.parametrize("family", ["darwin", "linux"])
def test_a_launcher_written_is_recognised_as_ours(tmp_path, family):
    built = plan(family, _places(tmp_path), Path("/env/python"), desktop=True)

    written = launcher.write(built)

    assert written == [entry.path for entry in built.entries]
    assert all(launcher.is_ours(entry) for entry in built.entries)


@posix_only
def test_the_mac_bundle_can_be_started(tmp_path):
    built = plan("darwin", _places(tmp_path), Path("/env/python"), desktop=False)
    launcher.write(built)

    executable = built.entries[0].path / "Contents" / "MacOS" / "am-seg"

    assert os.access(executable, os.X_OK)


@posix_only
def test_a_launcher_already_there_is_written_again_only_when_asked(tmp_path):
    built = plan("linux", _places(tmp_path), Path("/env/python"), desktop=False)
    launcher.write(built)

    with pytest.raises(LauncherRefused, match="--replace"):
        launcher.write(built)
    launcher.write(built, replace=True)


@posix_only
def test_something_else_in_the_way_refuses_the_whole_launcher(tmp_path):
    """Checked before anything is written, so the icon, first in line, is not
    left behind by a refusal the menu entry caused."""
    places = _places(tmp_path)
    built = plan("linux", places, Path("/env/python"), desktop=False)
    theirs = places.menu / WINDOW.desktop_entry_name
    theirs.parent.mkdir(parents=True)
    theirs.write_text("[Desktop Entry]\nName=Someone else's\n")

    with pytest.raises(LauncherRefused, match="not written by am-seg"):
        launcher.write(built, replace=True)

    assert theirs.read_text() == "[Desktop Entry]\nName=Someone else's\n"
    assert not (places.icons / WINDOW.icon_name).exists()


@posix_only
def test_removing_takes_only_what_am_seg_wrote(tmp_path):
    places = _places(tmp_path)
    ours = plan("linux", places, Path("/env/python"), desktop=False)
    launcher.write(ours)
    theirs = places.desktop / WINDOW.desktop_entry_name
    theirs.parent.mkdir(parents=True)
    theirs.write_text("[Desktop Entry]\nName=Someone else's\n")

    removed, left = launcher.remove(plan("linux", places, Path("/env/python"), desktop=True))

    assert sorted(removed) == sorted(entry.path for entry in ours.entries)
    assert left == [theirs]
    assert theirs.exists()


@posix_only
def test_a_desktop_link_to_someone_elses_app_is_not_ours(tmp_path):
    places = _places(tmp_path)
    built = plan("darwin", places, Path("/env/python"), desktop=True)
    link = built.entries[1]
    link.path.parent.mkdir(parents=True)
    link.path.symlink_to(tmp_path / "elsewhere.app")

    assert not launcher.is_ours(link)


def test_a_windows_shortcut_is_ours_by_its_description(tmp_path):
    entry = plan("win32", _places(tmp_path), Path("C:/env/pythonw.exe"), desktop=False).entries[0]
    entry.path.parent.mkdir(parents=True)

    entry.path.write_bytes(b"L\x00\x00\x00" + "Someone else's".encode("utf-16-le"))
    assert not launcher.is_ours(entry)

    entry.path.write_bytes(b"L\x00\x00\x00" + DESCRIPTION.encode("utf-16-le"))
    assert launcher.is_ours(entry)
    assert not launcher.is_ours(entry, VIEWER)


def test_the_viewer_shortcut_is_given_the_viewers_arguments_and_mark(tmp_path):
    environment = launcher.windows_shortcut_environment(
        Path("C:/Start/V.lnk"), Path("C:/env/pythonw.exe"), Path("C:/Users/ada"), VIEWER
    )

    assert environment["AM_SEG_ARGUMENTS"] == " ".join(VIEWER.arguments)
    assert environment["AM_SEG_DESCRIPTION"] == VIEWER_DESCRIPTION


def test_a_windows_shortcut_is_written_by_powershell(tmp_path, monkeypatch):
    asked: list[dict[str, str]] = []
    monkeypatch.setattr(
        launcher, "run_powershell", lambda script, environment=None: asked.append(environment)
    )
    built = plan("win32", _places(tmp_path), Path("C:/env/pythonw.exe"), desktop=True)

    launcher.write(built, home=tmp_path)

    assert [Path(env["AM_SEG_SHORTCUT"]) for env in asked] == [e.path for e in built.entries]
    assert {env["AM_SEG_TARGET"] for env in asked} == {str(Path("C:/env/pythonw.exe"))}


# -- this computer -------------------------------------------------------------------


def test_this_computer_is_asked_once_and_puts_a_desktop_entry_on_windows_only(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(
        launcher, "places_for", lambda family, home, environ: asked.append(family) or _places(tmp_path)
    )

    for platform in ("darwin", "linux", "win32"):
        computer = launcher.this_computer(platform)
        computer.launcher()
        computer.launcher(desktop=True)

        assert computer.desktop_by_default == (platform == "win32")
    assert asked == ["darwin", "linux", "win32"]


def test_no_menu_calls_the_launcher_a_shortcut():
    """In the window, a shortcut is a key."""
    assert set(launcher.MENU_LABELS) == set(launcher.WHERE_TO_FIND) == {"darwin", "linux", "win32"}
    assert not any("shortcut" in label.lower() for label in launcher.MENU_LABELS.values())


def test_linux_places_follow_xdg_data_home(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "_linux_desktop", lambda home: home / "Schreibtisch")

    places = launcher.places_for("linux", tmp_path, {"XDG_DATA_HOME": str(tmp_path / "data")})

    assert places.menu == tmp_path / "data" / "applications"
    assert places.icons == tmp_path / "data" / "icons" / "hicolor" / "256x256" / "apps"
    assert places.desktop == tmp_path / "Schreibtisch"


def test_windows_places_are_the_folders_windows_names(monkeypatch):
    """The Desktop especially, which OneDrive moves."""
    monkeypatch.setattr(
        launcher, "run_powershell",
        lambda script, environment=None: "C:\\Start\\Programs\r\nC:\\OneDrive\\Desktop\r\n",
    )

    places = launcher.places_for("win32", Path("C:/Users/ada"), {})

    assert places.menu == Path("C:\\Start\\Programs")
    assert places.desktop == Path("C:\\OneDrive\\Desktop")


# -- the command ---------------------------------------------------------------------


@pytest.fixture
def linux_home(tmp_path, monkeypatch):
    """The command pointed at a temporary home, as on Linux."""
    installing = importlib.import_module("am_segmentation_manager.cli.installing")
    computer = launcher.ThisComputer("linux", _places(tmp_path), Path("/env/python"))
    monkeypatch.setattr(installing, "this_computer", lambda: computer)
    return computer.places


@posix_only
def test_a_dry_run_writes_nothing(linux_home, capsys):
    assert cli_main.main(["create-shortcut", "--dry-run"]) == 0

    assert "Would write" in capsys.readouterr().out
    assert not linux_home.menu.exists()


@posix_only
def test_the_command_writes_replaces_and_removes(linux_home, capsys):
    entry = linux_home.menu / WINDOW.desktop_entry_name

    assert cli_main.main(["create-shortcut"]) == 0
    assert entry.exists() and not (linux_home.desktop / WINDOW.desktop_entry_name).exists()
    assert cli_main.main(["create-shortcut"]) == 1
    assert cli_main.main(["create-shortcut", "--replace"]) == 0
    assert cli_main.main(["create-shortcut", "--remove"]) == 0

    assert not entry.exists()
    assert "Removed" in capsys.readouterr().out


@posix_only
def test_the_command_adds_and_removes_the_viewer_apart_from_the_window(linux_home, capsys):
    window_entry = linux_home.menu / WINDOW.desktop_entry_name
    viewer_entry = linux_home.menu / VIEWER.desktop_entry_name
    assert cli_main.main(["create-shortcut"]) == 0

    assert cli_main.main(["create-shortcut", "--viewer"]) == 0
    assert viewer_entry.exists()
    assert f"'{VIEWER_NAME}'" in capsys.readouterr().out
    assert cli_main.main(["create-shortcut", "--viewer", "--remove"]) == 0

    assert not viewer_entry.exists()
    assert window_entry.exists()
    assert (linux_home.icons / WINDOW.icon_name).exists()


def test_remove_and_replace_together_are_refused(linux_home):
    assert cli_main.main(["create-shortcut", "--remove", "--replace"]) == 2
