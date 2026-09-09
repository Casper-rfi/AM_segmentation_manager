"""Tests for selecting and counting images by series.

The series is registered once, as a breakdown, which is what gives it both a
``--series`` selector and a filter row in the Export tab without a second
declaration that could disagree with the first. These check that registration,
the matching, and the default each command falls back to.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from am_segmentation_manager.aggregate import BREAKDOWNS, by_series, count_by
from am_segmentation_manager.cli import build_parser
from am_segmentation_manager.cli.scope import (
    DEFAULT_SERIES,
    REPORTING_SERIES,
    _series_selectors,
)
from am_segmentation_manager.filtering import matches_series
from am_segmentation_manager.layout import LABELLED_SERIES, SERIES
from am_segmentation_manager.models import ImageRecord


def record(series: str, name: str = "img.tif") -> ImageRecord:
    return ImageRecord(image_path=Path("/p") / name, series=series, modality="SEM")


RECORDS = [
    record("adaptive", "a_SEM.tif"),
    record("adaptive", "b_SEM.tif"),
    record("reference", "ref_Polishing_start_eb.tif"),
    record("alignment", "ref_alignment_ib.tif"),
]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "selectors, expected",
    [
        (None, ["adaptive", "adaptive", "reference", "alignment"]),
        ([], ["adaptive", "adaptive", "reference", "alignment"]),
        (["all"], ["adaptive", "adaptive", "reference", "alignment"]),
        (["adaptive"], ["adaptive", "adaptive"]),
        (["reference"], ["reference"]),
        (["reference,alignment"], ["reference", "alignment"]),
        (["reference", "alignment"], ["reference", "alignment"]),
    ],
)
def test_series_selectors_match_what_they_name(selectors, expected):
    assert [r.series for r in RECORDS if matches_series(r, selectors)] == expected


def test_an_unknown_series_is_an_error_rather_than_an_empty_result():
    """A closed vocabulary, unlike a variant, so a typo is reportable."""
    with pytest.raises(ValueError, match="Unknown series: refrence"):
        matches_series(RECORDS[0], ["refrence"])


def test_an_unknown_series_is_reported_even_beside_a_known_one():
    with pytest.raises(ValueError, match="Unknown series"):
        matches_series(RECORDS[0], ["adaptive,alignmnet"])


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def test_counting_by_series_uses_the_names_the_selector_takes():
    """A value counted here must be a value --series accepts."""
    counts = count_by(RECORDS, by_series)

    assert counts == {"adaptive": 2, "alignment": 1, "reference": 1}
    for value in counts:
        assert matches_series(record(value), [value])


def test_series_is_registered_as_a_breakdown_of_record_kind():
    """Kind 'record' is what keeps it out of the header and label passes."""
    entry = BREAKDOWNS["series"]

    assert entry.kind == "record"
    assert entry.factory is by_series


# ---------------------------------------------------------------------------
# What a command covers when nobody says
# ---------------------------------------------------------------------------

def subcommands() -> dict:
    """Return the registered subparsers, keyed by command name."""
    parser = build_parser()
    action = next(
        item for item in parser._actions if isinstance(item, argparse._SubParsersAction)
    )
    return action.choices


def default_series_of(command: str) -> str:
    """Return what *command* falls back to when --series is not given.

    Read off the parser rather than by parsing a command line: several of
    these have required arguments of their own, and none of that is what this
    is about.
    """
    return subcommands()[command].get_default("default_series")


@pytest.mark.parametrize(
    "command",
    ["predict", "convert-predictions", "select", "deselect", "correct",
     "find-specks", "compress-labels", "quantify-labels", "export-metadata-csv",
     "create-training-csv"],
)
def test_a_command_that_acts_stays_on_adaptive_polishing(command):
    """Indexing 5,237 more images must not change what these write."""
    assert default_series_of(command) == DEFAULT_SERIES == "adaptive"


def test_a_command_that_reports_covers_every_series():
    """Narrowing a report silently is the bug this work exists to fix."""
    assert default_series_of("stats") == REPORTING_SERIES == "all"


def test_naming_a_series_overrides_the_default():
    args = subcommands()["stats"].parse_args(["--series", "reference"])

    assert args.series == ["reference"]


def test_every_command_with_a_scope_offers_the_selector():
    for name, sub in subcommands().items():
        options = {option for action in sub._actions for option in action.option_strings}
        if "--variant" in options:
            assert "--series" in options, name


# ---------------------------------------------------------------------------
# Which series a command will take at all
# ---------------------------------------------------------------------------

#: Commands that write, measure or export label data. An `other` record has no
#: label directory and no predictions directory, so none of these could do
#: anything but refuse it once a scan had run.
ACTING_COMMANDS = [
    "predict", "convert-predictions", "select", "deselect", "correct",
    "find-specks", "compress-labels", "quantify-labels", "create-training-csv",
    "revert",
]

#: Commands that describe what is on disk. An image with no labels still has a
#: header, a place in the collection and a file to copy.
REPORTING_COMMANDS = ["stats", "export-metadata-csv", "export-files"]


@pytest.mark.parametrize("command", ACTING_COMMANDS)
def test_a_command_that_acts_will_not_take_other_at_all(command, capsys):
    """Refused at the argument, before a scan, rather than at the record -- and
    the message names what the command does take, so the refusal is a usable
    answer rather than only a rejection. Reading stderr also proves the exit is
    this refusal and not a required argument these commands have of their own.
    """
    with pytest.raises(SystemExit):
        subcommands()[command].parse_args(["--series", "other", "--modality", "all"])

    assert "this command covers adaptive, reference, alignment" in capsys.readouterr().err


def test_other_is_refused_even_beside_a_series_that_is_offered():
    with pytest.raises(SystemExit):
        subcommands()["predict"].parse_args(
            ["--series", "adaptive,other", "--modality", "all"]
        )


def scope_of(command: str, series: list[str] | None = None) -> argparse.Namespace:
    """Return what *command* would resolve --series to, without parsing a line.

    Read off the parser for the reason :func:`default_series_of` gives: several
    of these have required arguments of their own, and none of that is what
    this is about.
    """
    sub = subcommands()[command]
    return argparse.Namespace(
        series=series or [],
        default_series=sub.get_default("default_series"),
        series_choices=sub.get_default("series_choices"),
    )


@pytest.mark.parametrize("command", REPORTING_COMMANDS)
def test_a_command_that_only_describes_takes_other(command):
    assert _series_selectors(scope_of(command, ["other"])) == ["other"]


def test_all_means_every_series_this_command_covers_not_every_series():
    """The two differ, which is the whole point: `stats --series all` counts the
    residue and `predict --series all` must not reach it."""
    acting = _series_selectors(scope_of("predict", ["all"]))

    assert acting == list(LABELLED_SERIES)
    assert "other" not in acting
    assert _series_selectors(scope_of("stats", ["all"])) == list(SERIES)


def test_the_acting_vocabulary_is_every_series_but_the_residue():
    """Derived from SERIES, so a series added later has to pick a side."""
    assert set(SERIES) - set(LABELLED_SERIES) == {"other"}


def test_a_command_that_acts_still_defaults_to_adaptive_polishing():
    """Narrowing the vocabulary must not have widened the default."""
    assert _series_selectors(scope_of("predict")) == ["adaptive"]


def test_the_help_of_a_command_that_acts_does_not_offer_other():
    """A word the help lists is a word the command takes."""
    for command in ACTING_COMMANDS:
        series = next(
            action for action in subcommands()[command]._actions
            if "--series" in action.option_strings
        )
        assert "other" not in series.help, command
