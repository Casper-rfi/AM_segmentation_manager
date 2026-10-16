"""Every command and option must explain itself, and be findable.

The GUI already asserts that every keyboard shortcut carries a description
(``test_gui_window.test_every_shortcut_is_documented``). The CLI is the larger
surface and had no equivalent, which is how ``convert-predictions`` went without
a description paragraph unnoticed.

Findable is the second half: ``--help`` reaches only somebody already running
the program, so ``guides/command_line.md`` catalogues every command, in a table
generated from this parser and held to it by ``test_guides.py``.
"""
from __future__ import annotations

import argparse

import pytest

from am_segmentation_manager.cli import build_parser


def _subparser_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    """Return the parser's subcommand action."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("the CLI parser has no subcommands")


def _subcommands() -> dict[str, argparse.ArgumentParser]:
    """Return every subcommand name mapped to its parser, aliases included."""
    return dict(_subparser_action(build_parser()).choices)


def _canonical_subcommands() -> list[tuple[str, argparse.ArgumentParser]]:
    """Return one entry per distinct subparser, so aliases are not counted twice."""
    seen: dict[int, tuple[str, argparse.ArgumentParser]] = {}
    for name, parser in _subcommands().items():
        seen.setdefault(id(parser), (name, parser))
    return list(seen.values())


#: The order ``am-seg --help`` must list the commands in, which is the order of
#: the workflow in CLAUDE.md: add data to the collection, curate it and prepare
#: labels, then quantify and export. argparse lists subcommands in registration
#: order, so this pins the order they are registered in.
WORKFLOW_ORDER = [
    "inspect-layout",
    "init-project",
    "show-metadata",
    "edit-metadata",
    "list-projects",
    "gui",
    "predict",
    "convert-predictions",
    "deselect",
    "select",
    "correct",
    "show-families",
    "edit-families",
    "find-specks",
    "compress-labels",
    "stats",
    "quantify-labels",
    "export-metadata-csv",
    "export-projects-csv",
    "export-files",
    "create-training-csv",
    "check-training-csv",
    "train",
    "show-training",
    "stop-training",
    "validate",
    "compare-training",
    "compare-predictions",
    "choose-best",
    "init-history",
    "savepoint",
    "list-savepoints",
    "history",
    "merge",
    "revert",
    "archive",
    "prune-archive",
    # Setting this computer up, never a step of the workflow.
    "check-environment",
    "create-shortcut",
]


def test_parser_builds_without_running_a_command():
    parser = build_parser()

    assert parser.description
    assert _subcommands()


@pytest.mark.parametrize("name", [n for n, _ in _canonical_subcommands()])
def test_every_subcommand_has_a_help_line(name):
    """The one-line help shown in the top-level command list.

    Aliases share their parser's entry, so only the primary name appears here.
    """
    action = _subparser_action(build_parser())

    choice = next((c for c in action._choices_actions if c.dest == name), None)
    assert choice is not None, f"{name} has no entry in the command list"
    assert choice.help, f"{name} has no help= for the command list"


@pytest.mark.parametrize("name", [n for n, _ in _canonical_subcommands()])
def test_every_subcommand_has_a_description(name):
    """The paragraph shown by ``<command> --help``."""
    parser = _subcommands()[name]

    assert parser.description, f"{name} has no description=, so its --help explains nothing"


@pytest.mark.parametrize("name", [n for n, _ in _canonical_subcommands()])
def test_every_argument_has_help(name):
    """Every flag and positional explains itself."""
    parser = _subcommands()[name]

    undocumented = [
        action.dest
        for action in parser._actions
        if not action.help and action.dest != "help"
    ]

    assert not undocumented, f"{name} has arguments with no help=: {undocumented}"


def test_every_subcommand_has_a_handler():
    """A command the parser accepts and the dispatch table does not know falls
    through to the help, which looks like the command doing nothing. Aliases
    are included: they are separate names on the parser and each needs an
    entry."""
    from am_segmentation_manager.cli import COMMANDS

    registered = set(_subcommands())

    assert registered - set(COMMANDS) == set()


def test_the_dispatch_table_offers_nothing_the_parser_will_not_accept():
    """The other direction: an entry for a command nobody can type is either a
    command that was removed or a name spelt two ways."""
    from am_segmentation_manager.cli import COMMANDS

    assert set(COMMANDS) - set(_subcommands()) == set()


def test_subcommands_are_listed_in_workflow_order():
    """The command list is how the commands are found, so its order teaches the
    workflow. Aliases are excluded: they share their parser's position."""
    listed = [name for name, _ in _canonical_subcommands()]

    assert listed == WORKFLOW_ORDER


def test_top_level_flags_have_help():
    """The subcommand container itself carries no help; every real flag must."""
    undocumented = [
        action.dest
        for action in build_parser()._actions
        if not action.help
        and action.dest != "help"
        and not isinstance(action, argparse._SubParsersAction)
    ]

    assert not undocumented


def test_convert_predictions_explains_which_predictions_it_reads():
    """A user with a populated clean/ has to be told why it is passed over.

    Otherwise the command looks broken: the predictions are visibly there and
    nothing is written.
    """
    description = _subcommands()["convert-predictions"].description

    assert "raw" in description and "clean" in description


def test_version_flag_reports_the_package_version():
    from am_segmentation_manager import __version__

    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])

    assert excinfo.value.code == 0
    assert __version__


def test_usage_names_the_command_a_user_types():
    """Not the import name, which is neither what is installed nor what is run."""
    assert build_parser().format_usage().startswith("usage: am-seg ")
    assert _subcommands()["gui"].format_usage().startswith("usage: am-seg gui ")


def test_no_help_text_carries_the_home_directory_it_was_built_on():
    """Help is documentation: it goes into screenshots, bug reports and READMEs.

    A resolved path puts whoever ran the command into all of them, and makes
    the same flag read differently on two machines.
    """
    from pathlib import Path

    home = str(Path.home())
    parser = build_parser()
    texts = [action.help or "" for action in parser._actions]
    for _, sub in _canonical_subcommands():
        texts.extend(action.help or "" for action in sub._actions)
        texts.append(sub.description or "")

    assert not [text for text in texts if home in text]



# ---------------------------------------------------------------------------
# The headings
#
# Twenty-nine commands in one flat list is a list nobody reads to the end, so
# `--help` files them under the step of the workflow they belong to. The
# headings come from the command modules, and these hold them to it.
# ---------------------------------------------------------------------------

def test_every_command_is_filed_under_a_heading():
    """A command with no heading is one `--help` prints under whatever heading
    happened to come before it, which is worse than an ungrouped list: it
    states the wrong step rather than none."""
    from am_segmentation_manager.cli.main import HEADINGS

    unfiled = [name for name, _parser in _canonical_subcommands() if name not in HEADINGS]

    assert not unfiled, f"{unfiled} would be listed under the group above them"


def test_each_group_declares_its_heading():
    """The heading is the module's, so a group added to `GROUPS` brings one
    with it rather than needing a table here kept in step.

    A group with *no* `TITLE` never reaches this: `HEADINGS` is built at import
    and raises, so a parser cannot be built without one. That is deliberate --
    a blank heading would ship a group filed under nothing. This catches the
    emptier mistake of declaring one and leaving it blank.
    """
    from am_segmentation_manager.cli.main import GROUPS

    missing = [g.__name__ for g in GROUPS if not getattr(g, "TITLE", "")]

    assert not missing, f"{missing} have no TITLE for --help to file them under"


def test_the_headings_appear_in_the_help_in_group_order():
    """The order of the headings is the order of the workflow, and it is the
    order `GROUPS` registers them in -- not a second list that can drift."""
    from am_segmentation_manager.cli.main import GROUPS

    text = build_parser().format_help()
    seen = [g.TITLE for g in GROUPS if g.TITLE in text]

    assert seen == [g.TITLE for g in GROUPS], "a heading is missing or out of order"


def test_the_usage_line_does_not_list_every_command():
    """Without a metavar argparse prints all twenty-nine names into the usage
    line as one brace-list that wraps over four lines and says nothing."""
    usage = build_parser().format_usage()

    assert "<command>" in usage
    assert "inspect-layout" not in usage, "the usage line is listing the commands again"
