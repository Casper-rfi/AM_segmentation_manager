"""What an operation reports back, and how a path is shown in it.

The currency between the controllers that run batch operations and the window
that previews, confirms and applies them. Here rather than in either controller
because both produce one and neither owns it -- and because the window's runner
consumes it knowing about no controller at all.

Qt-free, like everything it is passed between.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..io.path import make_path_relative

#: How many filenames a report lists before leaving the rest to the count. A
#: copy of a whole collection runs to thousands of files, and a dialog listing
#: them is one nobody reads to the end.
MAX_REPORTED_FILES = 20


@dataclass(frozen=True)
class ConflictRow:
    """One thing a merge cannot decide, as a person has to see it.

    **One row per group, not one per kind.** A label's pixels and its workflow
    state are resolved under a single key, so they are one decision; showing
    two controls over one key would be a window lying about what a click did.

    Attributes:
        key: What names this decision to :func:`~..operations.merge.plan_merge`.
        title: The group or field, as the reader knows it.
        details: One line per kind in conflict -- the pixels, the state -- each
            saying what the two sides hold.
        corrected: Whether what is here is proofread work.
        ours: The label here, where this is a label. For looking at.
        theirs: The copy's label, where this is a label.
        image: The acquisition both labels describe, for looking at them over.
    """

    key: str
    title: str
    details: tuple[str, ...] = ()
    corrected: bool = False
    ours: Path | None = None
    theirs: Path | None = None
    image: Path | None = None

    @property
    def is_label(self) -> bool:
        """Return whether there is something here to look at.

        A state or a metadata row is two words against two words, both already
        on the row; there is no picture of it to show.
        """
        return self.ours is not None and self.theirs is not None


@dataclass(frozen=True)
class OperationReport:
    """What an operation did, or would do if it ran.

    Attributes:
        headline: The single line summarising the run.
        lines: One line per affected file or group, in the operation's order.
        count: How many things were affected, or would be. Zero means there is
            nothing to apply, which is what the window checks before asking.
        applied: False for a preview.
        blocked: Why the operation cannot run, or empty when it can. A blocked
            report carries no work; it is shown and dismissed.
        conflicts: What a merge cannot decide, for the dialog that asks. Empty
            for everything else, and for a merge with nothing outstanding.
        written: The files the run actually created, for an operation whose
            whole effect on the index is the paths it wrote. The window names
            them on the records that share them instead of reading the
            collection again; empty for a preview, and for a run whose effect
            is not expressible this way. **Only what was written**, never what
            was planned: a stopped run plans more than it creates, and a record
            claiming a prediction that is not there offers to convert nothing.
    """

    headline: str
    lines: tuple[str, ...] = ()
    count: int = 0
    applied: bool = False
    blocked: str = ""
    written: tuple[Path, ...] = ()
    conflicts: tuple[ConflictRow, ...] = ()

    def text(self) -> list[str]:
        """Return the report as displayable lines, headline first."""
        return [self.headline, *self.lines]


def capped(lines: Iterable[str]) -> tuple[str, ...]:
    """Return *lines* cut to :data:`MAX_REPORTED_FILES`, saying how many were left out.

    A list that stops without saying so reads as the whole of it.
    """
    lines = tuple(lines)
    if len(lines) <= MAX_REPORTED_FILES:
        return lines
    return (
        *lines[:MAX_REPORTED_FILES],
        f"... and {len(lines) - MAX_REPORTED_FILES} more",
    )


def shown_path(path: Path | None, root: Path | None) -> str:
    """Return *path* as a report should show it, against the scan root.

    An absolute path eight directories deep is unreadable in a dialog, and a
    report that fell back to one for some rows and not others would be read as
    saying something about those rows.
    """
    if path is None:
        return "(no label)"
    if root is None:
        return str(path)
    return make_path_relative(path, root)
