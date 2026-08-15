"""Typed project metadata, resolved from the two YAML files.

Metadata comes from ``experiment.yaml``, written by Autolamella, and
``AM_metadata.yaml``, written by this package or by hand, which overrides it.
Values resolve **per key**, so a project can override one field without
restating the rest.

Nothing is inherited from parent directories: a directory can hold projects
prepared on different samples, so resolving a field from a parent would
attribute the wrong sample to a project invisibly.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Filename of this package's metadata file.
AM_METADATA_FILENAME = "AM_metadata.yaml"

#: Filename of Autolamella's experiment file.
EXPERIMENT_FILENAME = "experiment.yaml"

#: Placeholder written by ``init-project`` for a value only a human can supply.
#: Treated as "not filled in" rather than as a value.
TODO_MARKER = "TODO"

#: Autolamella's ``method`` value mapped to a lamella preparation type.
#: The 2026 schema drops ``method`` entirely, so this covers older projects
#: only; the rest need the value entered by hand.
METHOD_LAMELLA_TYPES: dict[str, str] = {
    "autolamella-ongrid": "cells-on-grid",
    "autoliftout": "lift-out",
    "autolamella-liftout": "lift-out",
    "autolamella-waffle": "waffle",
}

#: Accepted values for ``lamella_type``.
LAMELLA_TYPES: tuple[str, ...] = ("cells-on-grid", "waffle", "lift-out", "other")


def merge_metadata(base: dict, override: dict) -> dict:
    """Merge two metadata dictionaries, with `override` taking precedence.

    This performs a shallow merge: keys present in override replace base
    values. Deeper merging can be added when needed.
    """
    if not base:
        base = {}
    if not override:
        return dict(base)
    merged = dict(base)
    merged.update(override)
    logger.debug("Merged metadata keys: %s", list(merged.keys()))
    return merged


@dataclass(frozen=True)
class ExperimentMetadata:
    """Descriptive metadata for one project.

    Every field is optional: metadata is filled in progressively, and a project
    with none is still perfectly indexable. ``None`` means "not recorded",
    which includes a field still holding a ``TODO`` placeholder.

    Attributes:
        description: What the experiment was for.
        organisation: The organisation that acquired the data.
        project: The name of the wider project the experiment belongs to.
        user: Who acquired the data.
        sample: Cell type or organism.
        lamella_type: One of :data:`LAMELLA_TYPES`.
        variants: Human-readable descriptions of imaging-condition variants,
            keyed by the variant string in the image directory name. Variants
            are opaque to the package, so this is purely for display.
        sources: Which file supplied each populated field, for tracing a value
            back when it looks wrong.
        todo_fields: Fields present in a file but still holding a ``TODO``
            placeholder, so the remaining work can be reported as a checklist.
    """

    description: str | None = None
    organisation: str | None = None
    project: str | None = None
    user: str | None = None
    sample: str | None = None
    lamella_type: str | None = None
    variants: Mapping[str, str] = field(default_factory=dict)
    sources: Mapping[str, Path] = field(default_factory=dict)
    todo_fields: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return whether no descriptive field has been recorded."""
        return not any(getattr(self, name) for name in DESCRIPTIVE_FIELDS)

    def missing_fields(self) -> tuple[str, ...]:
        """Return the descriptive fields with no value, in declaration order."""
        return tuple(name for name in DESCRIPTIVE_FIELDS if getattr(self, name) is None)

    def variant_description(self, variant: str | None) -> str:
        """Return a display label for a variant, falling back to the raw string."""
        if variant is None:
            return "base"
        return self.variants.get(variant, variant)


#: The descriptive fields, in the order they are written to a template.
DESCRIPTIVE_FIELDS: tuple[str, ...] = (
    "description",
    "organisation",
    "project",
    "user",
    "sample",
    "lamella_type",
)

#: What a person is expected to enter in each field, in one line. Written above
#: the field in a generated template and shown as its tooltip in the GUI, which
#: is why it lives here rather than in either of them.
FIELD_COMMENTS: dict[str, str] = {
    "description": "What this experiment was for",
    "organisation": "Organisation that acquired the data",
    "project": "Name of the wider project this belongs to",
    "user": "Who acquired the data",
    "sample": "Cell type or organism",
    "lamella_type": f"One of: {', '.join(LAMELLA_TYPES)}",
}


def _metadata_section(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a document's ``metadata:`` block, or an empty mapping.

    Autolamella's 2026 ``experiment.yaml`` carries an empty top-level
    ``metadata: {}``, the same key this package uses, so the two layer without
    conflict as long as resolution is per key.
    """
    section = document.get("metadata")
    return section if isinstance(section, Mapping) else {}


def _clean(value: Any) -> str | None:
    """Return a trimmed string value, or None for blanks and TODO markers."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == TODO_MARKER:
        return None
    return text


def lamella_type_from_method(method: Any) -> str | None:
    """Return the lamella type implied by Autolamella's ``method`` value."""
    text = _clean(method)
    if text is None:
        return None
    resolved = METHOD_LAMELLA_TYPES.get(text.lower())
    if resolved is None:
        logger.debug("No lamella type known for Autolamella method %r", text)
    return resolved


def build_experiment_metadata(
    experiment: Mapping[str, Any] | None,
    am_metadata: Mapping[str, Any] | None,
    experiment_path: Path | None = None,
    am_metadata_path: Path | None = None,
) -> ExperimentMetadata:
    """Build the typed metadata view from the two source documents.

    Values are resolved per key, strongest source last:

    1. ``experiment.yaml``: its ``metadata:`` block, plus ``lamella_type``
       derived from ``method:`` when present
    2. ``AM_metadata.yaml``: its ``metadata:`` block

    Args:
        experiment: Parsed ``experiment.yaml``, or None.
        am_metadata: Parsed ``AM_metadata.yaml``, or None.
        experiment_path: Path to record as the source of experiment values.
        am_metadata_path: Path to record as the source of AM_metadata values.

    Returns:
        The resolved metadata, recording which file supplied each value and
        which fields are still placeholders.
    """
    experiment = experiment or {}
    am_metadata = am_metadata or {}

    values: dict[str, str] = {}
    sources: dict[str, Path] = {}
    todo: set[str] = set()

    def apply(section: Mapping[str, Any], path: Path | None) -> None:
        for name in DESCRIPTIVE_FIELDS:
            if name not in section:
                continue
            cleaned = _clean(section[name])
            if cleaned is None:
                if str(section[name] or "").strip().upper() == TODO_MARKER:
                    todo.add(name)
                continue
            todo.discard(name)
            values[name] = cleaned
            if path is not None:
                sources[name] = path

    apply(_metadata_section(experiment), experiment_path)

    derived = lamella_type_from_method(experiment.get("method"))
    if derived is not None and "lamella_type" not in values:
        values["lamella_type"] = derived
        if experiment_path is not None:
            sources["lamella_type"] = experiment_path

    apply(_metadata_section(am_metadata), am_metadata_path)

    unknown = {
        value for name, value in values.items()
        if name == "lamella_type" and value not in LAMELLA_TYPES
    }
    for value in unknown:
        logger.warning(
            "Unrecognised lamella_type %r; expected one of %s",
            value, ", ".join(LAMELLA_TYPES),
        )

    variants = am_metadata.get("variants")
    if not isinstance(variants, Mapping):
        variants = {}

    return ExperimentMetadata(
        **values,
        variants={str(key): str(text) for key, text in variants.items()},
        sources=sources,
        todo_fields=tuple(sorted(todo)),
    )


def parse_created_at(value: float | int | str | None) -> datetime | None:
    """Convert an Autolamella ``created_at`` Unix timestamp to a datetime.

    The *experiment.yaml* file stores creation time as a Unix timestamp
    (e.g. ``1785231172.267575``).  This helper converts the value to a
    timezone-aware UTC datetime.

    Args:
        value: Unix timestamp as a float, int, or numeric string.
            Returns ``None`` when *value* is ``None`` or cannot be parsed.
    """
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        logger.warning("Could not parse created_at timestamp: %r", value)
        return None
