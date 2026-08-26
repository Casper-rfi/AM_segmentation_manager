"""Editing a project's ``AM_metadata.yaml``.

Creating that file is :mod:`~am_segmentation_manager.operations.init_project`'s
job; this module changes one that exists.

Only ``AM_metadata.yaml`` is ever written -- ``experiment.yaml`` belongs to
Autolamella. It is edited in place line by line rather than reparsed and dumped,
which preserves its comments, key order and any keys this package does not
model. Nothing is written until the whole batch validates.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..io.yaml import read_yaml
from ..metadata import (
    AM_METADATA_FILENAME,
    DESCRIPTIVE_FIELDS,
    EXPERIMENT_FILENAME,
    TODO_MARKER,
    build_experiment_metadata,
    field_named,
)
from .init_project import render_template

logger = logging.getLogger(__name__)

#: The block holding the fields this package models.
METADATA_KEY = "metadata"


class MetadataValidationError(ValueError):
    """Raised when a requested edit is not one that can be made.

    Carries every problem found, not just the first: a caller correcting a
    command line should see all of them at once.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class FieldChange:
    """One field's before and after.

    Attributes:
        field: The field name.
        old: The value in effect before the edit, resolved across both files.
        new: The value being written. ``None`` means the field is being removed
            from ``AM_metadata.yaml``, after which ``experiment.yaml``'s value
            (if any) applies again.
        old_source: Which file supplied *old*, so a surprising value can be
            traced without opening both files.
    """

    field: str
    old: str | None
    new: str | None
    old_source: Path | None = None

    @property
    def is_removal(self) -> bool:
        return self.new is None

    def describe(self) -> str:
        """Return a one-line description, for a preview or a log."""
        origin = f" (from {self.old_source.name})" if self.old_source else ""
        before = "unset" if self.old is None else repr(self.old)
        after = "unset" if self.new is None else repr(self.new)
        return f"{self.field}: {before}{origin} -> {after}"


@dataclass(frozen=True)
class MetadataChange:
    """What an edit would do, or did.

    Attributes:
        path: The ``AM_metadata.yaml`` affected.
        project: The project's identity, for reporting.
        changes: One entry per field that would actually change. A field asked
            to be set to the value it already holds does not appear.
        created: Whether the file had to be created.
        applied: Whether it was written, as opposed to previewed.
    """

    path: Path
    project: str
    changes: tuple[FieldChange, ...] = ()
    created: bool = False
    applied: bool = False

    @property
    def is_empty(self) -> bool:
        """Return whether nothing would change."""
        return not self.changes

    def describe(self) -> list[str]:
        """Return the change as lines, for a preview."""
        if self.is_empty:
            return [f"{self.project}: no change"]
        prefix = "would create" if self.created and not self.applied else None
        lines = [f"{self.project}:"]
        if prefix:
            lines.append(f"  ({prefix} {self.path.name})")
        lines.extend(f"  {change.describe()}" for change in self.changes)
        return lines


def _yaml_scalar(value: str) -> str:
    """Return *value* quoted as YAML requires, and no more than required.

    Delegated to the dumper rather than guessed: a sample containing a colon,
    a leading ``#``, or the word ``yes`` all need different handling, and
    getting one wrong writes a file that reads back as something else.
    """
    import yaml

    dumped = yaml.safe_dump({"k": value}, default_flow_style=False, allow_unicode=True)
    return dumped.split(":", 1)[1].strip()


def _validate(values: Mapping[str, str | None]) -> None:
    """Check every requested edit before any of them is made."""
    problems: list[str] = []
    for field, value in values.items():
        spec = field_named(field)
        if spec is None:
            problems.append(
                f"unknown field {field!r}; expected one of {', '.join(DESCRIPTIVE_FIELDS)}"
            )
            continue
        if value is None or value == TODO_MARKER:
            continue
        if not str(value).strip():
            problems.append(f"{field}: empty value; use --clear to remove it")
        if spec.choices and value not in spec.choices:
            problems.append(
                f"{field} {value!r} is not one of {', '.join(spec.choices)}"
            )
    if problems:
        raise MetadataValidationError(problems)


def _resolved_before(project_root: Path) -> tuple[dict[str, str | None], dict[str, Path]]:
    """Return the values in effect now, and which file supplied each."""
    experiment = read_yaml(project_root / EXPERIMENT_FILENAME)
    am_metadata = read_yaml(project_root / AM_METADATA_FILENAME)
    resolved = build_experiment_metadata(
        experiment,
        am_metadata,
        experiment_path=project_root / EXPERIMENT_FILENAME,
        am_metadata_path=project_root / AM_METADATA_FILENAME,
    )
    current = {field: getattr(resolved, field) for field in DESCRIPTIVE_FIELDS}
    return current, dict(resolved.sources)


def _edit_lines(text: str, values: Mapping[str, str | None]) -> str:
    """Apply edits to the ``metadata:`` block, leaving everything else alone.

    Fields already present are replaced where they sit, so comments above them
    keep applying to the right line. Fields not present are appended to the end
    of the block. Removals delete the line.
    """
    lines = text.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().rstrip(":") == METADATA_KEY and not line.startswith((" ", "\t"))
        ),
        None,
    )
    if start is None:
        # No block to edit: put one at the end rather than guessing where it
        # belongs among whatever else the file holds.
        lines.extend(["", f"{METADATA_KEY}:"])
        start = len(lines) - 1

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line.startswith((" ", "\t")):
            end = index
            break

    remaining = dict(values)
    edited: list[str] = []
    for index in range(start + 1, end):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            edited.append(line)
            continue
        field = stripped.split(":", 1)[0].strip()
        if field not in remaining:
            edited.append(line)
            continue
        value = remaining.pop(field)
        if value is None:
            continue  # removal: drop the line
        indent = line[: len(line) - len(line.lstrip())]
        edited.append(f"{indent}{field}: {_yaml_scalar(value)}")

    # Anything not already in the block is appended to it.
    additions = [
        f"  {field}: {_yaml_scalar(value)}"
        for field, value in remaining.items()
        if value is not None
    ]
    while edited and not edited[-1].strip():
        additions.append(edited.pop())

    result = lines[: start + 1] + edited + additions + lines[end:]
    return "\n".join(result) + "\n"


def plan_metadata_update(
    project_root: Path,
    values: Mapping[str, str | None],
    *,
    project_name: str | None = None,
) -> MetadataChange:
    """Return what updating *project_root*'s metadata would do, writing nothing.

    Args:
        project_root: The project directory.
        values: Field name to new value. ``None`` removes the field from
            ``AM_metadata.yaml``, so ``experiment.yaml``'s value applies again.
            ``"TODO"`` marks a field as outstanding work, which is different
            from removing it: an unfilled placeholder is reported by
            ``inspect-layout``.
        project_name: Identity to report, defaulting to the directory name.

    Raises:
        MetadataValidationError: If any requested edit is invalid. Nothing is
            written and no partial change is returned.
    """
    _validate(values)
    current, sources = _resolved_before(project_root)
    destination = project_root / AM_METADATA_FILENAME

    changes = []
    for field, new in values.items():
        old = current.get(field)
        # TODO is "outstanding", which resolves to None but is not the same as
        # absent, so a field already holding one is still a change to record.
        if old == new and not (new == TODO_MARKER and old is None):
            continue
        if new == TODO_MARKER and old is None and _holds_todo(destination, field):
            continue
        changes.append(FieldChange(field=field, old=old, new=new, old_source=sources.get(field)))

    return MetadataChange(
        path=destination,
        project=project_name or project_root.name,
        changes=tuple(changes),
        created=not destination.exists(),
    )


def _holds_todo(destination: Path, field: str) -> bool:
    """Return whether the file already records *field* as outstanding."""
    if not destination.exists():
        return False
    block = read_yaml(destination).get(METADATA_KEY)
    if not isinstance(block, Mapping):
        return False
    return str(block.get(field, "")).strip().upper() == TODO_MARKER


def update_project_metadata(
    project_root: Path,
    values: Mapping[str, str | None],
    *,
    dry_run: bool = False,
    project_name: str | None = None,
) -> MetadataChange:
    """Update a project's ``AM_metadata.yaml``.

    The file is created from the standard template if it does not exist, so
    that the guidance comments come with it; otherwise it is edited in place,
    preserving comments, key order and any keys this package does not model.

    ``experiment.yaml`` is never written.

    Args:
        project_root: The project directory.
        values: As :func:`plan_metadata_update`.
        dry_run: Compute and return the change without writing.
        project_name: Identity to report.

    Returns:
        What changed, with ``applied`` saying whether it was written.

    Raises:
        MetadataValidationError: If any requested edit is invalid.
    """
    change = plan_metadata_update(project_root, values, project_name=project_name)
    if dry_run or change.is_empty:
        return change

    destination = change.path
    if change.created:
        filled = {
            field: value for field, value in values.items()
            if value is not None and value != TODO_MARKER
        }
        text = render_template(project_root, filled)
        # The template writes TODO for anything not supplied, which is what an
        # explicit TODO asks for anyway.
        destination.write_text(text, encoding="utf-8")
    else:
        destination.write_text(
            _edit_lines(destination.read_text(encoding="utf-8"), values), encoding="utf-8"
        )

    logger.info(
        "Updated %s: %s", destination, "; ".join(c.describe() for c in change.changes)
    )
    return MetadataChange(
        path=change.path,
        project=change.project,
        changes=change.changes,
        created=change.created,
        applied=True,
    )
