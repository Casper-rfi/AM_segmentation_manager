"""Writing ``AM_metadata.yaml`` templates into projects.

Two jobs, which happen to want the same file:

* making a pre-Autolamella dataset indexable at all, since it has no
  ``experiment.yaml`` for the scanner to find;
* recording the descriptive metadata that Autolamella does not capture --
  sample, user, organisation and so on -- for projects that are already
  indexable.

The file is generated as text rather than dumped from a mapping, because the
comments explaining each field are most of its value to whoever fills it in, and
a YAML dump would discard them.

Fields that can be derived are pre-filled. Fields that vary between projects are
written as ``TODO`` placeholders and never guessed: an unfilled placeholder is
visible and reportable, whereas a plausible wrong sample is neither. This is why
``--defaults`` copies shared values *into* each file at generation time rather
than being resolved from a parent directory at scan time.

Existing files are never overwritten: they hold information a person typed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..io.yaml import read_yaml
from ..metadata import (
    AM_METADATA_FILENAME,
    DESCRIPTIVE_FIELDS,
    EXPERIMENT_FILENAME,
    LAMELLA_TYPES,
    TODO_MARKER,
    lamella_type_from_method,
)

logger = logging.getLogger(__name__)

#: Fields that must never be pre-filled from a defaults file, because they vary
#: between projects even when everything else is shared. Getting one of these
#: wrong is invisible in the output, so they stay placeholders until a person
#: fills them in.
NEVER_DEFAULTED: frozenset[str] = frozenset({"description", "sample"})

#: One-line explanation written above each field.
FIELD_COMMENTS: dict[str, str] = {
    "description": "What this experiment was for",
    "organisation": "Organisation that acquired the data",
    "project": "Name of the wider project this belongs to",
    "user": "Who acquired the data",
    "sample": "Cell type or organism",
    "lamella_type": f"One of: {', '.join(LAMELLA_TYPES)}",
}


@dataclass(frozen=True)
class InitResult:
    """Outcome of generating one project's metadata file.

    Attributes:
        path: Where the file is, or would be, written.
        written: Whether a file was created. False for a dry run and for a
            project that already had one.
        skipped_existing: Whether an existing file stopped this project.
        derived: Field names that were pre-filled.
        todo: Field names left as placeholders.
    """

    path: Path
    written: bool
    skipped_existing: bool
    derived: tuple[str, ...] = ()
    todo: tuple[str, ...] = ()


def read_defaults(path: Path) -> dict[str, str]:
    """Read shared field values from a defaults file.

    The file may either hold the fields at the top level or nest them under
    ``metadata:``, matching the shape of the file it feeds.

    Raises:
        FileNotFoundError: If the file does not exist. A defaults file named
            but missing is a mistake worth stopping for, not one to ignore.
        ValueError: If it holds a field this package does not recognise, or one
            that must not be defaulted.
    """
    if not path.exists():
        raise FileNotFoundError(f"Defaults file not found: {path}")

    document = read_yaml(path)
    section = document.get("metadata") if isinstance(document.get("metadata"), Mapping) else document

    unknown = sorted(set(section) - set(DESCRIPTIVE_FIELDS))
    if unknown:
        raise ValueError(
            f"{path}: unknown field(s) {', '.join(unknown)}; "
            f"expected any of {', '.join(DESCRIPTIVE_FIELDS)}"
        )
    forbidden = sorted(set(section) & NEVER_DEFAULTED)
    if forbidden:
        raise ValueError(
            f"{path}: {', '.join(forbidden)} varies between projects and cannot be "
            f"supplied as a default; fill it in per project"
        )
    return {name: str(value).strip() for name, value in section.items() if str(value).strip()}


def _derive_fields(project_root: Path) -> dict[str, str]:
    """Return field values derivable from a project's own files."""
    experiment_path = project_root / EXPERIMENT_FILENAME
    if not experiment_path.exists():
        return {}
    document = read_yaml(experiment_path)
    derived: dict[str, str] = {}
    lamella_type = lamella_type_from_method(document.get("method"))
    if lamella_type is not None:
        derived["lamella_type"] = lamella_type
    return derived


def render_template(
    project_root: Path,
    values: Mapping[str, str],
    derived: Sequence[str] = (),
) -> str:
    """Return the text of an ``AM_metadata.yaml`` for a project.

    Args:
        project_root: The project the file describes, named in a header comment
            so a file separated from its directory is still identifiable.
        values: Field values to write. Fields absent here are written as
            ``TODO`` placeholders.
        derived: Field names that were derived rather than supplied, noted in a
            trailing comment so a reader knows which values to double-check.
    """
    lines = [
        f"# Metadata for {project_root.name}",
        "#",
        "# Written by 'am-seg init-project'. Values here override experiment.yaml.",
        f"# Replace every {TODO_MARKER} with a real value, or delete the line if it",
        "# does not apply. Nothing is inherited from parent directories, so this",
        "# file is the whole story for this project.",
        "metadata:",
    ]
    for name in DESCRIPTIVE_FIELDS:
        comment = FIELD_COMMENTS.get(name)
        if comment:
            lines.append(f"  # {comment}")
        value = values.get(name, TODO_MARKER)
        suffix = "  # derived from experiment.yaml" if name in derived else ""
        lines.append(f"  {name}: {value}{suffix}")
    lines.extend(
        [
            "",
            "# Descriptions for imaging-condition variants, keyed by the suffix in the",
            "# image directory name (sem_5kv -> 5kv). Display only; delete if unused.",
            "# variants:",
            "#   5kv: 5 kV landing energy",
            "",
        ]
    )
    return "\n".join(lines)


def init_project(
    project_root: Path,
    defaults: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> InitResult:
    """Write an ``AM_metadata.yaml`` template into one project.

    An existing file is left untouched, and reported: it holds values a person
    typed, and regenerating it would discard them.

    Args:
        project_root: The project directory to write into.
        defaults: Shared values to pre-fill, as returned by :func:`read_defaults`.
        dry_run: Report what would be written without creating anything.

    Returns:
        An :class:`InitResult` describing what happened.
    """
    destination = project_root / AM_METADATA_FILENAME
    if destination.exists():
        logger.info("Leaving existing %s unchanged", destination)
        return InitResult(path=destination, written=False, skipped_existing=True)

    derived = _derive_fields(project_root)
    values = {**{k: v for k, v in (defaults or {}).items() if k not in NEVER_DEFAULTED}, **derived}
    todo = tuple(name for name in DESCRIPTIVE_FIELDS if name not in values)

    if not dry_run:
        destination.write_text(
            render_template(project_root, values, derived=tuple(derived)), encoding="utf-8"
        )
        logger.info("Wrote %s", destination)

    return InitResult(
        path=destination,
        written=not dry_run,
        skipped_existing=False,
        derived=tuple(sorted(derived)),
        todo=todo,
    )


def init_projects(
    project_roots: Sequence[Path],
    defaults: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> list[InitResult]:
    """Write a metadata template into each of *project_roots*."""
    return [init_project(root, defaults=defaults, dry_run=dry_run) for root in project_roots]
