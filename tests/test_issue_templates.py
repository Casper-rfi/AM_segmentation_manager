"""The GitHub issue forms held to the commands they ask a reporter to run.

A form is read by someone who has hit a problem and copies its commands as
written, so a command renamed in the package must fail here first.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest
import yaml

from am_segmentation_manager.cli import build_parser

TEMPLATES = Path(__file__).resolve().parent.parent / ".github" / "ISSUE_TEMPLATE"
FORMS = sorted(path for path in TEMPLATES.glob("*.yml") if path.name != "config.yml")

#: An am-seg command a form names, in backticks: its subcommand and flags.
COMMAND = re.compile(r"`(?:uv run )?am-seg ([^`]+)`")


def _subcommands() -> dict[str, argparse.ArgumentParser]:
    action = next(
        action for action in build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return dict(action.choices)


def _text(form: dict) -> str:
    return yaml.safe_dump(form["body"], allow_unicode=True, width=10_000)


def test_there_are_forms():
    assert FORMS


@pytest.mark.parametrize("path", FORMS, ids=lambda path: path.name)
def test_a_form_is_one_github_can_show(path: Path):
    form = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert form["name"] and form["description"]
    ids = [field["id"] for field in form["body"] if "id" in field]
    assert len(ids) == len(set(ids)), "two fields share an id"


@pytest.mark.parametrize("path", FORMS, ids=lambda path: path.name)
def test_a_form_asks_for_the_version(path: Path):
    """Without it a report cannot be matched to the code that produced it."""
    assert "`am-seg --version`" in _text(yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("path", FORMS, ids=lambda path: path.name)
def test_every_command_a_form_names_is_one_am_seg_has(path: Path):
    subcommands = _subcommands()

    for command in COMMAND.findall(_text(yaml.safe_load(path.read_text(encoding="utf-8")))):
        words = command.split()
        parser = build_parser()
        if not words[0].startswith("-"):
            assert words[0] in subcommands, f"{path.name} names am-seg {words[0]}, which does not exist"
            parser = subcommands[words[0]]
            words = words[1:]
        flags = {option for action in parser._actions for option in action.option_strings}
        for word in words:
            if word.startswith("-"):
                assert word in flags, f"{path.name} gives am-seg {command}, which takes no {word}"
