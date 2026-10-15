"""Typed project metadata, resolved from the two YAML files.

``experiment.yaml`` is Autolamella's and ``AM_metadata.yaml`` this package's or a
person's, which overrides it **per key**, so one field can be overridden without
restating the rest. Nothing is inherited from parent directories, which can hold
projects on different samples. Which fields exist, and how each is edited,
defaulted and counted by, is declared once in :data:`FIELDS`.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
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

#: The two freezing methods that cover almost every sample here, offered
#: whether or not a project has used one yet. Not a vocabulary: jet freezing
#: and self-pressurised freezing exist, and refusing one would be worse than
#: tolerating two spellings of the usual two.
FREEZING_METHODS: tuple[str, ...] = ("Plunge freezing", "High pressure freezing")

#: What kind of value a field holds. Text is the default. A number is validated
#: as one when it is entered and carried as a float, so a value that is not one
#: is refused at the point somebody can still fix it rather than deposited in a
#: CSV -- and, being a real number, it can be summed and sorted by whoever
#: receives that file.
TEXT = "text"
NUMBER = "number"

#: How several values of one field are written where only text will fit -- an
#: editor, a CSV cell, the name of a group. Not how they are *stored*: a file
#: holds a YAML list, so a value may contain a comma, a colon or a newline
#: without the file becoming ambiguous. A value may not contain this, which is
#: what splitting on it costs and the only thing it costs.
VALUE_SEPARATOR = "; "


def is_recorded(value: object) -> bool:
    """Return whether a resolved field value stands for anything.

    Not the truth of the value: a GIS deposition time of zero seconds is a
    recorded value and a falsy one, and an empty tuple is a multi-valued field
    holding nothing rather than a field holding an empty value.
    """
    if isinstance(value, tuple):
        return bool(value)
    return value is not None


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
    with none is still perfectly indexable. "Not recorded" is ``None`` for a
    field holding one value and an empty tuple for one holding several, which
    includes a field still holding a ``TODO`` placeholder. Ask
    :func:`is_recorded` rather than testing either, and :func:`field_values`
    for the values themselves.

    A multi-valued field is a tuple however many values it holds, so nothing
    reading one has to ask which case it is in.

    Attributes:
        description: What the experiment was for.
        organisation: The organisations that acquired the data.
        project: The wider projects the experiment belongs to.
        user: Who acquired the data.
        sample: Cell types or organisms. Several because one specimen can be a
            co-culture, or a host and the pathogen infecting it.
        sample_state: What was done to the sample, e.g. infected,
            differentiated. Separate values rather than one phrase, so an
            export can be narrowed to every infected sample however else it
            was treated.
        sample_medium: What the sample was in, e.g. culture medium,
            cryoprotectant, fiducials.
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
        sources: Which file supplied each populated field, for tracing a value
            back when it looks wrong.
        todo_fields: Fields present in a file but still holding a ``TODO``
            placeholder, so the remaining work can be reported as a checklist.
    """

    description: str | None = None
    organisation: tuple[str, ...] = ()
    project: tuple[str, ...] = ()
    user: tuple[str, ...] = ()
    sample: tuple[str, ...] = ()
    sample_state: tuple[str, ...] = ()
    sample_medium: tuple[str, ...] = ()
    freezing_method: str | None = None
    grid_id: str | None = None
    grid_type: str | None = None
    gis_deposition_time_s: float | None = None
    lamella_type: str | None = None
    sources: Mapping[str, Path] = field(default_factory=dict)
    todo_fields: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Return whether no descriptive field has been recorded."""
        return len(self.missing_fields()) == len(DESCRIPTIVE_FIELDS)

    def missing_fields(self) -> tuple[str, ...]:
        """Return the descriptive fields with no value, in declaration order."""
        return tuple(
            name for name in DESCRIPTIVE_FIELDS
            if not is_recorded(getattr(self, name))
        )


#: The headings the Add data tab gathers the fields under, in the order shown.
GENERAL = "General"
SAMPLE_PREPARATION = "Sample preparation"
LAMELLA_PREPARATION = "Lamella preparation"


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
        multi: Whether the field may hold several values at once. Stored as a
            YAML list, held as a tuple, and shown separated by
            :data:`VALUE_SEPARATOR`. A single value is read, written and
            displayed the same either way, so turning this on for a field
            leaves every file already written valid.
        choices: The values accepted, or empty for free text. A field with a
            vocabulary is edited by picking from it and rejects anything else.
        defaults: Values always offered, whether or not a project uses one yet.
            For a field with a handful of usual answers and a long tail --
            unlike *choices*, anything else is still accepted, so an unusual
            one is typed rather than refused.
        suggested: Whether the editor offers the values other projects already
            use. For a field whose value repeats across projects, which is how
            a collection ends up with two spellings of one sample.
        defaultable: Whether a shared defaults file may pre-fill it. False
            where the value varies between projects even when everything else
            is shared, so getting it wrong would be invisible in the output.
        breakdown_label: What this field is called when records are counted by
            it, or None for a field nothing can usefully be grouped by.
        section: The heading it sits under on the Add data tab. A section's
            fields are consecutive rows, so the table's order is the tab's.
        prose: Whether the value is a sentence or two rather than a name, and
            so edited in a box that wraps.
    """

    name: str
    label: str
    comment: str
    kind: str = TEXT
    multi: bool = False
    choices: tuple[str, ...] = ()
    defaults: tuple[str, ...] = ()
    suggested: bool = False
    defaultable: bool = True
    breakdown_label: str | None = None
    section: str = GENERAL
    prose: bool = False

    @property
    def breakdown_name(self) -> str:
        """Return the name a command line takes to group by this field."""
        return self.name.replace("_", "-")

    @property
    def guidance(self) -> str:
        """Return the comment, saying as well whether the field holds several.

        Appended rather than written into each comment: a field that becomes
        multi-valued should start saying so without anyone remembering to edit
        its line.
        """
        if not self.multi:
            return self.comment
        return f"{self.comment}. Holds several -- add them one at a time"


#: The descriptive fields, in the order they are written to a template and
#: shown on the Add data tab, a section's fields together.
#:
#: **Adding a field is a row here.** The name, the guidance, the editor it gets,
#: whether it can be defaulted and whether records can be counted by it are all
#: read off this table; nothing outside it lists the fields or singles one out.
FIELDS: tuple[ProjectField, ...] = (
    ProjectField(
        "description",
        "Description",
        "Goals and context of the experiment",
        defaultable=False,
        section=GENERAL,
        prose=True,
    ),
    ProjectField(
        "organisation",
        "Organisation(s)",
        "Organisation that acquired the data",
        multi=True,
        suggested=True,
        breakdown_label="organisation",
        section=GENERAL,
    ),
    ProjectField(
        "project",
        "Project(s)",
        "Name of the wider project this belongs to",
        multi=True,
        suggested=True,
        breakdown_label="project",
        section=GENERAL,
    ),
    ProjectField(
        "user",
        "User(s)",
        "Who acquired the data",
        multi=True,
        suggested=True,
        breakdown_label="user",
        section=GENERAL,
    ),
    ProjectField(
        "sample",
        "Sample(s)",
        "Cell type or organism",
        multi=True,
        suggested=True,
        defaultable=False,
        breakdown_label="sample",
        section=SAMPLE_PREPARATION,
    ),
    ProjectField(
        "sample_state",
        "Sample state(s)",
        "What was done to the sample, e.g. infected, differentiated",
        multi=True,
        suggested=True,
        defaultable=False,
        breakdown_label="sample state",
        section=SAMPLE_PREPARATION,
    ),
    ProjectField(
        "sample_medium",
        "Sample media",
        "What the sample was in, e.g. culture medium, cryoprotectant, fiducials",
        multi=True,
        suggested=True,
        defaultable=False,
        breakdown_label="sample medium",
        section=SAMPLE_PREPARATION,
    ),
    ProjectField(
        "grid_type",
        "Grid type",
        "Grid the sample was prepared on, e.g. Quantifoil R2/2 Cu 200 mesh",
        suggested=True,
        defaultable=False,
        breakdown_label="grid type",
        section=SAMPLE_PREPARATION,
    ),
    ProjectField(
        "grid_id",
        "Grid ID",
        "Identifier of the physical grid this was prepared on",
        suggested=True,
        defaultable=False,
        breakdown_label="grid ID",
        section=SAMPLE_PREPARATION,
    ),
    ProjectField(
        "freezing_method",
        "Freezing method",
        "How the sample was vitrified",
        defaults=FREEZING_METHODS,
        suggested=True,
        breakdown_label="freezing method",
        section=SAMPLE_PREPARATION,
    ),
    ProjectField(
        "gis_deposition_time_s",
        # Spelled out, where the breakdown heading keeps the symbol: beside
        # "Project(s)" a bracketed "(s)" reads as a plural, not as seconds.
        "GIS deposition time (seconds)",
        "Seconds of GIS deposition, as a number",
        kind=NUMBER,
        breakdown_label="GIS deposition time (s)",
        section=LAMELLA_PREPARATION,
    ),
    ProjectField(
        "lamella_type",
        "Lamella type",
        f"One of: {', '.join(LAMELLA_TYPES)}",
        choices=LAMELLA_TYPES,
        breakdown_label="lamella preparation type",
        section=LAMELLA_PREPARATION,
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


def _clean_items(value: Any) -> tuple[str, ...]:
    """Return a multi-valued field's values, as the file states them.

    A bare scalar is one value, which is what keeps every file written before a
    field became multi-valued readable, and what lets ``experiment.yaml`` carry
    one without knowing this package's conventions. Repeats are dropped and
    order is kept.

    The separator is not looked for here: storage is a YAML list, so a value
    may contain any character at all, and a scalar that happens to hold a
    semicolon is one value that reads as one value.
    """
    items = value if isinstance(value, (list, tuple)) else [value]
    cleaned = (_clean(item) for item in items)
    return tuple(dict.fromkeys(text for text in cleaned if text is not None))


def _is_placeholder(value: Any) -> bool:
    """Return whether a raw value is a ``TODO`` marker rather than a value."""
    return str(value or "").strip().upper() == TODO_MARKER


def yaml_field_value(field: str, value: str) -> str:
    """Return *value* written as YAML would write it for *field*.

    Delegated to the dumper rather than guessed: a sample containing a colon,
    a leading ``#``, or the word ``yes`` all need different handling, and
    getting one wrong writes a file that reads back as something else.

    A numeric field is written as a number, so the file says what it holds;
    quoted, it would read back as a string that happens to look like one. A
    field holding several values is written as a list, on one line so that the
    line-by-line editing above stays a matter of replacing a line -- and as a
    bare scalar when it holds one, which is what it has always looked like and
    what it still reads back as. Placeholders are exempt from both, ``TODO``
    being no more a number than it is a sample.
    """
    import yaml

    spec = FIELDS_BY_NAME.get(field)
    if spec is not None and value != TODO_MARKER:
        if spec.multi:
            items = list(split_values(value))
            if len(items) != 1:
                # The whole list through the dumper rather than the items
                # joined: a value holding a comma has to come back quoted, or
                # [a, b] reads as two values where one was meant.
                return yaml.safe_dump(
                    items, default_flow_style=True, allow_unicode=True
                ).strip()
            value = items[0]
        elif spec.kind == NUMBER:
            number = float(value)
            dumped = yaml.safe_dump(
                {"k": int(number) if number.is_integer() else number},
                default_flow_style=False,
            )
            return dumped.split(":", 1)[1].strip()

    dumped = yaml.safe_dump({"k": value}, default_flow_style=False, allow_unicode=True)
    return dumped.split(":", 1)[1].strip()



def parse_field_value(name: str, text: str) -> str | float | tuple[str, ...] | None:
    """Return the value the field called *name* holds, from text a person typed.

    The editor's and the command line's form, where several values arrive
    separated by :data:`VALUE_SEPARATOR`; a file states them as a list instead,
    which :func:`build_experiment_metadata` reads directly.

    A number that is not one is dropped with a warning rather than kept as
    text: the attribute is typed, and a value that cannot be summed sitting in
    a column of ones that can is worse than an admitted gap.
    """
    spec = FIELDS_BY_NAME.get(name)
    if spec is None:
        return text
    if spec.multi:
        return split_values(text)
    if spec.kind != NUMBER:
        return text
    try:
        return float(text)
    except ValueError:
        logger.warning("%s is %r, which is not a number; ignoring it", name, text)
        return None


def split_values(text: str) -> tuple[str, ...]:
    """Return the values *text* stands for, as a person typed them.

    Order is kept and repeats are dropped: a field naming one organisation
    twice says nothing the once did.
    """
    parts = (part.strip() for part in text.split(VALUE_SEPARATOR.strip()))
    return tuple(dict.fromkeys(part for part in parts if part))


def field_values(value: object) -> tuple[str, ...]:
    """Return a resolved value as the separate values it holds.

    One for a field holding one, none for a field holding nothing, and however
    many a multi-valued field holds. What anything counting, offering or
    listing values asks, so that a field holding several is not a special case
    at each of them.
    """
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    return (format_field_value(value),)


def format_field_value(value: object) -> str:
    """Return a resolved field value as the text that stands for it.

    One formatter for every consumer -- an editor, a CSV cell, the name of a
    group in a breakdown -- so a value cannot be shown one way and matched
    another. A whole number loses its decimal point: these are entered by hand,
    and a deposition time typed as ``30`` should read back as ``30``.
    """
    if value is None:
        return ""
    if isinstance(value, tuple):
        return VALUE_SEPARATOR.join(value)
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

    values: dict[str, Any] = {}
    sources: dict[str, Path] = {}
    todo: set[str] = set()

    def apply(section: Mapping[str, Any], path: Path | None) -> None:
        for name in DESCRIPTIVE_FIELDS:
            if name not in section:
                continue
            raw = section[name]
            if FIELDS_BY_NAME[name].multi:
                value: object = _clean_items(raw)
            else:
                cleaned = _clean(raw)
                value = None if cleaned is None else parse_field_value(name, cleaned)
            if not is_recorded(value):
                if _is_placeholder(raw):
                    todo.add(name)
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

    return ExperimentMetadata(
        **values,
        sources=sources,
        todo_fields=tuple(sorted(todo)),
    )
