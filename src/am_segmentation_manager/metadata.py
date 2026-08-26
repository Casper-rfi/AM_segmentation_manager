"""Typed project metadata, resolved from the two YAML files.

Metadata comes from ``experiment.yaml``, written by Autolamella, and
``AM_metadata.yaml``, written by this package or by hand, which overrides it.
Values resolve **per key**, so a project can override one field without
restating the rest.

Nothing is inherited from parent directories: a directory can hold projects
prepared on different samples, so resolving a field from a parent would
attribute the wrong sample to a project invisibly.

Which fields exist, and how each is edited, defaulted and counted by, is
declared once in :data:`FIELDS`.
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

#: Filename of the protocol Autolamella milled to. Nothing here reads it --
#: see `docs/decisions.md` on why it is not a project marker -- but it
#: describes how the data was produced, so a copy of a dataset takes it along.
PROTOCOL_FILENAME = "protocol.yaml"

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

#: What kind of value a field holds. Text is the default. A number is validated
#: as one when it is entered and carried as a float, so a value that is not one
#: is refused at the point somebody can still fix it rather than deposited in a
#: CSV -- and, being a real number, it can be summed and sorted by whoever
#: receives that file.
TEXT = "text"
NUMBER = "number"


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
        freezing_method: How the sample was vitrified, e.g.
            ``"plunge-frozen"``. Free text, for the reason ``grid_type`` is.
        grid_id: Identifier of the physical grid the lamellae were cut from.
            Several projects may name one grid; nothing here assumes they do
            not.
        grid_type: The grid the sample was prepared on, e.g.
            ``"Quantifoil R2/2 Cu 200 mesh"``. Free text, material, mesh and
            film combining in more ways than a fixed vocabulary would survive.
        gis_deposition_time_s: Seconds of GIS deposition. A number, so ``0.0``
            is a recorded value -- no deposition -- and not the same as unset.
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
    freezing_method: str | None = None
    grid_id: str | None = None
    grid_type: str | None = None
    gis_deposition_time_s: float | None = None
    lamella_type: str | None = None
    variants: Mapping[str, str] = field(default_factory=dict)
    sources: Mapping[str, Path] = field(default_factory=dict)
    todo_fields: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return whether no descriptive field has been recorded.

        Tested against None rather than for truth: a GIS deposition time of
        zero seconds is a recorded value, and a falsy one.
        """
        return all(getattr(self, name) is None for name in DESCRIPTIVE_FIELDS)

    def missing_fields(self) -> tuple[str, ...]:
        """Return the descriptive fields with no value, in declaration order."""
        return tuple(name for name in DESCRIPTIVE_FIELDS if getattr(self, name) is None)

    def variant_description(self, variant: str | None) -> str:
        """Return a display label for a variant, falling back to the raw string."""
        if variant is None:
            return "base"
        return self.variants.get(variant, variant)


@dataclass(frozen=True)
class ProjectField:
    """One descriptive field, and how each part of the package treats it.

    Attributes:
        name: The key in the ``metadata:`` block and the attribute on
            :class:`ExperimentMetadata`.
        label: What the field is called in a form. Written out rather than
            derived from *name*, which turns ``grid_id`` into "Grid Id" and
            ``gis_deposition_time_s`` into something nobody would write.
        comment: What a person is expected to enter, in one line. Written above
            the field in a generated template and shown as its tooltip.
        kind: :data:`TEXT` or :data:`NUMBER`. A number is rejected unless it is
            one, and reaches :class:`ExperimentMetadata` as a float.
        choices: The values accepted, or empty for free text. A field with a
            vocabulary is edited by picking from it and rejects anything else.
        suggested: Whether the editor offers the values other projects already
            use. For a field whose value repeats across projects, which is how
            a collection ends up with two spellings of one sample.
        defaultable: Whether a shared defaults file may pre-fill it. False
            where the value varies between projects even when everything else
            is shared, so getting it wrong would be invisible in the output.
        breakdown_label: What this field is called when records are counted by
            it, or None for a field nothing can usefully be grouped by.
    """

    name: str
    label: str
    comment: str
    kind: str = TEXT
    choices: tuple[str, ...] = ()
    suggested: bool = False
    defaultable: bool = True
    breakdown_label: str | None = None

    @property
    def breakdown_name(self) -> str:
        """Return the name a command line takes to group by this field."""
        return self.name.replace("_", "-")


#: The descriptive fields, in the order they are written to a template.
#:
#: **Adding a field is a row here.** The name, the guidance, the editor it gets,
#: whether it can be defaulted and whether records can be counted by it are all
#: read off this table; nothing outside it lists the fields or singles one out.
FIELDS: tuple[ProjectField, ...] = (
    ProjectField(
        "description",
        "Description",
        "What this experiment was for",
        defaultable=False,
    ),
    ProjectField(
        "organisation",
        "Organisation",
        "Organisation that acquired the data",
        suggested=True,
        breakdown_label="organisation",
    ),
    ProjectField(
        "project",
        "Project",
        "Name of the wider project this belongs to",
        suggested=True,
        breakdown_label="project",
    ),
    ProjectField(
        "user",
        "User",
        "Who acquired the data",
        suggested=True,
        breakdown_label="user",
    ),
    ProjectField(
        "sample",
        "Sample",
        "Cell type or organism",
        suggested=True,
        defaultable=False,
        breakdown_label="sample",
    ),
    ProjectField(
        "freezing_method",
        "Freezing method",
        "How the sample was vitrified, e.g. plunge-frozen",
        suggested=True,
        breakdown_label="freezing method",
    ),
    ProjectField(
        "grid_id",
        "Grid ID",
        "Identifier of the physical grid this was prepared on",
        suggested=True,
        defaultable=False,
        breakdown_label="grid",
    ),
    ProjectField(
        "grid_type",
        "Grid type",
        "Grid the sample was prepared on, e.g. Quantifoil R2/2 Cu 200 mesh",
        suggested=True,
        defaultable=False,
        breakdown_label="grid type",
    ),
    ProjectField(
        "gis_deposition_time_s",
        "GIS deposition time (s)",
        "Seconds of GIS deposition, as a number",
        kind=NUMBER,
        breakdown_label="GIS deposition time (s)",
    ),
    ProjectField(
        "lamella_type",
        "Lamella type",
        f"One of: {', '.join(LAMELLA_TYPES)}",
        choices=LAMELLA_TYPES,
        breakdown_label="lamella preparation type",
    ),
)

#: The fields by name, for the lookups that have one in hand.
FIELDS_BY_NAME: dict[str, ProjectField] = {field.name: field for field in FIELDS}

#: The field names, in declaration order.
DESCRIPTIVE_FIELDS: tuple[str, ...] = tuple(FIELDS_BY_NAME)

#: What a person is expected to enter in each field, in one line.
FIELD_COMMENTS: dict[str, str] = {field.name: field.comment for field in FIELDS}


def field_named(name: str) -> ProjectField | None:
    """Return the field called *name*, or None if this package has no such field."""
    return FIELDS_BY_NAME.get(name)


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


def parse_field_value(name: str, text: str) -> str | float | None:
    """Return a cleaned string as the value the field called *name* holds.

    A number that is not one is dropped with a warning rather than kept as
    text: the attribute is typed, and a value that cannot be summed sitting in
    a column of ones that can is worse than an admitted gap.
    """
    spec = FIELDS_BY_NAME.get(name)
    if spec is None or spec.kind != NUMBER:
        return text
    try:
        return float(text)
    except ValueError:
        logger.warning("%s is %r, which is not a number; ignoring it", name, text)
        return None


def format_field_value(value: object) -> str:
    """Return a resolved field value as the text that stands for it.

    One formatter for every consumer -- an editor, a CSV cell, the name of a
    group in a breakdown -- so a value cannot be shown one way and matched
    another. A whole number loses its decimal point: these are entered by hand,
    and a deposition time typed as ``30`` should read back as ``30``.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


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

    values: dict[str, str | float] = {}
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
            value = parse_field_value(name, cleaned)
            if value is None:
                continue
            todo.discard(name)
            values[name] = value
            if path is not None:
                sources[name] = path

    apply(_metadata_section(experiment), experiment_path)

    derived = lamella_type_from_method(experiment.get("method"))
    if derived is not None and "lamella_type" not in values:
        values["lamella_type"] = derived
        if experiment_path is not None:
            sources["lamella_type"] = experiment_path

    apply(_metadata_section(am_metadata), am_metadata_path)

    for name, value in values.items():
        field_spec = FIELDS_BY_NAME[name]
        if field_spec.choices and value not in field_spec.choices:
            logger.warning(
                "Unrecognised %s %r; expected one of %s",
                name, value, ", ".join(field_spec.choices),
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
