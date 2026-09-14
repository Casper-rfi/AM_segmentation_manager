"""Which side of each merge conflict to keep, as a person works through them.

Arranges a merge's conflicts as the collection is arranged -- folder, project,
lamella, group in acquisition order; or folder, project, field in form order --
and holds one decision per conflict. **A decision belongs to a conflict, never
to a row above one**: deciding a folder decides every conflict under it, and
what a folder shows is read back from those. Nothing is decided in advance and
nothing here writes; Qt-free, like the controllers.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Final

from ..frames import group_by_acquisition
from ..metadata import FIELDS
from ..models import folder_of
from ..operations.merge import OURS, THEIRS
from .report import ConflictRow

__all__ = [
    "ALL",
    "LEVEL_CONFLICT",
    "LEVEL_FOLDER",
    "LEVEL_LAMELLA",
    "LEVEL_PROJECT",
    "NONE",
    "SIDES",
    "SOME",
    "ConflictNode",
    "ConflictResolution",
]

#: The two answers a conflict can be given.
SIDES: Final[tuple[str, ...]] = (OURS, THEIRS)

#: How much of what is under a row has been given one side.
ALL: Final[str] = "all"
SOME: Final[str] = "some"
NONE: Final[str] = "none"

#: What a row of the tree stands for.
LEVEL_FOLDER: Final[str] = "folder"
LEVEL_PROJECT: Final[str] = "project"
LEVEL_LAMELLA: Final[str] = "lamella"
LEVEL_CONFLICT: Final[str] = "conflict"

#: Where each metadata field comes in the form, and what the form calls it.
_FIELD_ORDER: Final[dict[str, int]] = {
    spec.name: index for index, spec in enumerate(FIELDS)
}
_FIELD_LABELS: Final[dict[str, str]] = {spec.name: spec.label for spec in FIELDS}


@dataclass(eq=False)
class ConflictNode:
    """One row of a conflict tree: a single conflict, or a heading over several.

    Compared by identity, so a widget can key its rows by node.

    Attributes:
        level: One of the ``LEVEL_`` values.
        caption: What the row says.
        keys: Every conflict under this row, in tree order -- only its own,
            for a conflict.
        children: The rows under this one. Empty for a conflict.
        row: The conflict itself, where this is one.
        parent: The row above, or None at the top of the tree.
    """

    level: str
    caption: str
    keys: tuple[str, ...] = ()
    children: tuple[ConflictNode, ...] = ()
    row: ConflictRow | None = None
    parent: ConflictNode | None = field(default=None, repr=False)

    @property
    def is_conflict(self) -> bool:
        """Return whether this row is one conflict rather than a heading."""
        return self.row is not None


class ConflictResolution:
    """The conflicts of one merge, arranged for deciding, and what was decided.

    Attributes:
        labels: The top rows of the tree of label groups.
        fields: The top rows of the tree of project metadata fields.
    """

    def __init__(self, rows: Iterable[ConflictRow]) -> None:
        rows = list(rows)
        self._rows: dict[str, ConflictRow] = {row.key: row for row in rows}
        self._decided: dict[str, str] = {}
        self.labels = _tree([row for row in rows if not row.is_field])
        self.fields = _tree([row for row in rows if row.is_field])
        self._conflicts: dict[str, ConflictNode] = {
            node.keys[0]: node
            for node in _walk((*self.labels, *self.fields))
            if node.is_conflict
        }
        self._label_keys = tuple(key for node in self.labels for key in node.keys)
        self._field_keys = tuple(key for node in self.fields for key in node.keys)
        self._label_holders = _holders(self.labels)
        self._field_holders = _holders(self.fields)

    # -- reading -------------------------------------------------------------

    def conflict(self, key: str) -> ConflictNode:
        """Return the row of the conflict *key*.

        Raises:
            KeyError: If *key* is not one of this merge's conflicts.
        """
        return self._conflicts[key]

    def side(self, key: str) -> str | None:
        """Return the side chosen for the conflict *key*, or None while undecided."""
        return self._decided.get(key)

    def coverage(self, node: ConflictNode, side: str) -> str:
        """Return how much of what is under *node* has been given *side*.

        :data:`ALL`, :data:`SOME` or :data:`NONE`, asked once per side, so a
        heading over conflicts decided both ways shows some of each.
        """
        given = sum(1 for key in node.keys if self._decided.get(key) == side)
        if not given:
            return NONE
        return ALL if given == len(node.keys) else SOME

    def outstanding(self, nodes: Iterable[ConflictNode] | None = None) -> int:
        """Return how many conflicts under *nodes*, or in the whole merge, are undecided."""
        keys = (
            self._rows
            if nodes is None
            else [key for node in nodes for key in node.keys]
        )
        return sum(1 for key in keys if key not in self._decided)

    def proofread(self, node: ConflictNode) -> int:
        """Return how many conflicts under *node* hold proofread work here."""
        return sum(1 for key in node.keys if self._rows[key].corrected)

    @property
    def resolutions(self) -> dict[str, str]:
        """Return what has been decided, keyed and valued as a merge reads it."""
        return dict(self._decided)

    # -- deciding ------------------------------------------------------------

    def decide(self, node: ConflictNode, side: str | None) -> None:
        """Give every conflict under *node* the side *side*, or None to undecide them.

        Raises:
            ValueError: If *side* is neither of :data:`SIDES`.
        """
        if side is not None and side not in SIDES:
            raise ValueError(
                f"{side!r} is not a side; expected one of {', '.join(SIDES)}"
            )
        for key in node.keys:
            if side is None:
                self._decided.pop(key, None)
            else:
                self._decided[key] = side

    def toggle(self, node: ConflictNode, side: str) -> None:
        """Give *side* to everything under *node*, or take it away where it holds it throughout.

        What clicking a mark does: a full mark empties, any other fills.
        """
        self.decide(node, None if self.coverage(node, side) == ALL else side)

    def decide_all(self, nodes: Iterable[ConflictNode], side: str | None) -> None:
        """Decide every row of *nodes* the same way, as :meth:`decide` does one."""
        for node in nodes:
            self.decide(node, side)

    # -- moving --------------------------------------------------------------

    def first_undecided(self, nodes: Iterable[ConflictNode]) -> str | None:
        """Return the first undecided conflict under *nodes*, or None if there is none."""
        return next(
            (key for node in nodes for key in node.keys if key not in self._decided),
            None,
        )

    def next_undecided(self, key: str) -> str | None:
        """Return the next undecided conflict after *key*, in the tree *key* is in.

        Wraps round and ends at *key* itself, so None means every conflict in
        that tree has been decided.
        """
        order = self._field_keys if self._rows[key].is_field else self._label_keys
        start = order.index(key) + 1
        return next(
            (other for other in (*order[start:], *order[:start])
             if other not in self._decided),
            None,
        )

    def step_within(self, key: str, offset: int) -> str | None:
        """Return the conflict *offset* places from *key* under the same row.

        None past either end, as stepping through a lamella stops at its ends.
        """
        siblings = _conflicts_under(self._conflicts[key].parent)
        index = siblings.index(key) + offset
        return siblings[index] if 0 <= index < len(siblings) else None

    def step_across(self, key: str, offset: int) -> str | None:
        """Return the first conflict of the row *offset* rows away that holds any.

        A lamella for a group, a project for a field, crossing into the next
        project or folder where one runs out. None past either end of the tree.
        """
        holders = (
            self._field_holders if self._rows[key].is_field else self._label_holders
        )
        parent = self._conflicts[key].parent
        assert parent is not None, "a conflict always sits under a row"
        index = holders.index(parent) + offset
        if not 0 <= index < len(holders):
            return None
        return _conflicts_under(holders[index])[0]


# ---------------------------------------------------------------------------
# Building the tree
# ---------------------------------------------------------------------------

def _tree(rows: Sequence[ConflictRow]) -> tuple[ConflictNode, ...]:
    """Return the top rows of the tree over *rows*.

    A level with nothing to say is left out rather than shown empty: a project
    at the scan root hangs from the top, as it does in the project list, and a
    group whose image names no lamella hangs from its project.
    """
    nested: dict[str, dict[str, dict[str, list[ConflictRow]]]] = {}
    for row in rows:
        by_project = nested.setdefault(folder_of(row.project), {})
        by_project.setdefault(row.project, {}).setdefault(row.lamella, []).append(row)

    top: list[ConflictNode] = []
    for folder, projects in sorted(nested.items()):
        heads = [_project(key, lamellae) for key, lamellae in sorted(projects.items())]
        if folder:
            top.append(_heading(LEVEL_FOLDER, folder, heads))
        else:
            top.extend(heads)
    return tuple(top)


def _project(key: str, lamellae: dict[str, list[ConflictRow]]) -> ConflictNode:
    """Return one project's row, over its lamellae or directly over its conflicts."""
    children: list[ConflictNode] = []
    for lamella, rows in sorted(lamellae.items()):
        leaves = [_leaf(row) for row in _in_order(rows)]
        if lamella:
            children.append(_heading(LEVEL_LAMELLA, lamella, leaves))
        else:
            children.extend(leaves)
    return _heading(LEVEL_PROJECT, PurePath(key).name, children)


def _in_order(rows: Sequence[ConflictRow]) -> list[ConflictRow]:
    """Return one row's conflicts in the order they are stepped through.

    Fields as the form lists them. Groups by the grouping the timeline uses,
    so SEM comes before FIB within one acquisition; a group that cannot be
    placed there -- no record, or no beam -- follows, by name.
    """
    if any(row.is_field for row in rows):
        return sorted(
            rows,
            key=lambda row: (
                _FIELD_ORDER.get(row.metadata_field, len(_FIELD_ORDER)),
                row.metadata_field,
            ),
        )
    unplaced = {id(row.record): row for row in rows if row.record is not None}
    acquisitions = group_by_acquisition(
        row.record for row in rows if row.record is not None
    )
    placed = [
        unplaced.pop(id(record))
        for acquisition in acquisitions
        for record in acquisition.records
        if id(record) in unplaced
    ]
    rest = [row for row in rows if row.record is None or id(row.record) in unplaced]
    return placed + sorted(rest, key=lambda row: row.key)


def _leaf(row: ConflictRow) -> ConflictNode:
    """Return the row of one conflict."""
    return ConflictNode(LEVEL_CONFLICT, _caption(row), keys=(row.key,), row=row)


def _caption(row: ConflictRow) -> str:
    """Return what one conflict's row says: the field, or the group's image."""
    if row.is_field:
        return _FIELD_LABELS.get(row.metadata_field, row.metadata_field)
    if row.record is not None:
        return row.record.image_path.name
    return PurePath(row.key).name


def _heading(
    level: str, caption: str, children: Sequence[ConflictNode]
) -> ConflictNode:
    """Return a row over *children*, holding every conflict they hold."""
    node = ConflictNode(
        level,
        caption,
        keys=tuple(key for child in children for key in child.keys),
        children=tuple(children),
    )
    for child in children:
        child.parent = node
    return node


def _walk(nodes: Iterable[ConflictNode]) -> Iterator[ConflictNode]:
    """Yield *nodes* and everything under them, each row before its children."""
    for node in nodes:
        yield node
        yield from _walk(node.children)


def _holders(nodes: Iterable[ConflictNode]) -> list[ConflictNode]:
    """Return the rows holding conflicts directly, in tree order."""
    return [
        node for node in _walk(nodes)
        if any(child.is_conflict for child in node.children)
    ]


def _conflicts_under(node: ConflictNode | None) -> list[str]:
    """Return the conflicts directly under *node*, in order."""
    if node is None:
        return []
    return [child.keys[0] for child in node.children if child.is_conflict]
