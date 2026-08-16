"""What a state change applies to: checked projects, lamellae or frames.

A set the user builds and can see -- checkmarks in the two lists, a band over
the frames picked out on the timeline. **The finest level present wins**, and
nothing coarser is consulted; see `docs/decisions.md` § Workflow state for why
that order. Qt-free, like the controllers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: What the selection currently rests on, finest first. Reported so the window
#: can say which, and so a confirmation can name it.
LEVEL_FRAMES = "frames"
LEVEL_LAMELLAE = "lamellae"
LEVEL_PROJECTS = "projects"
#: Nothing is selected, so a change applies to the frame being viewed.
LEVEL_CURRENT_FRAME = "frame"


def selection_key(path: Path) -> str:
    """Return the stable identity of a project or lamella directory."""
    return str(Path(path).resolve())


@dataclass
class Selection:
    """The set a state change applies to, and the state of the gesture building it.

    Attributes:
        projects: Resolved paths of the checked projects.
        lamellae: Resolved paths of the checked lamellae.
        frames: Indices of the selected frames within the current lamella.
        anchor: The frame a Shift-extension measures from. The last frame the
            user pointed at, whether or not that selected anything.
    """

    projects: set[str] = field(default_factory=set)
    lamellae: set[str] = field(default_factory=set)
    frames: set[int] = field(default_factory=set)
    anchor: int | None = None
    #: The frame selection as it stood when the current drag began. An
    #: extension is computed from this rather than added to what is already
    #: there, so dragging back over frames un-selects them again instead of
    #: leaving the widest reach of the drag behind.
    _base: set[int] = field(default_factory=set)

    # -- what is selected --------------------------------------------------

    @property
    def level(self) -> str:
        """Return which level the selection rests on."""
        if self.frames:
            return LEVEL_FRAMES
        if self.lamellae:
            return LEVEL_LAMELLAE
        if self.projects:
            return LEVEL_PROJECTS
        return LEVEL_CURRENT_FRAME

    @property
    def is_empty(self) -> bool:
        """Return whether nothing has been picked out."""
        return self.level == LEVEL_CURRENT_FRAME

    def clear(self) -> None:
        """Drop everything, at every level."""
        self.projects.clear()
        self.lamellae.clear()
        self.clear_frames()

    def clear_frames(self) -> None:
        """Drop the frame selection and the gesture building it."""
        self.frames.clear()
        self._base.clear()
        self.anchor = None

    # -- checking projects and lamellae ------------------------------------

    def check(self, kind: str, key: str, checked: bool) -> None:
        """Check or uncheck one project or lamella.

        Args:
            kind: ``"projects"`` or ``"lamellae"``.
            key: The directory's resolved path, from :func:`selection_key`.
            checked: Whether it is now checked.
        """
        chosen = self._set(kind)
        if checked:
            chosen.add(key)
        else:
            chosen.discard(key)

    def check_all(self, kind: str, keys: list[str], checked: bool) -> None:
        """Check or uncheck every one of *keys*.

        Only the keys named are touched, so checking all the lamellae of one
        project leaves those checked in another project alone.
        """
        for key in keys:
            self.check(kind, key, checked)

    def is_checked(self, kind: str, key: str) -> bool:
        """Return whether a project or lamella is checked."""
        return key in self._set(kind)

    def _set(self, kind: str) -> set[str]:
        if kind == LEVEL_PROJECTS:
            return self.projects
        if kind == LEVEL_LAMELLAE:
            return self.lamellae
        raise ValueError(f"Not a checkable level: {kind}")

    # -- selecting frames --------------------------------------------------

    def point_at(self, position: int) -> None:
        """Note that the user pointed at a frame, without selecting it.

        Plain navigation moves the anchor, so a Shift-extension afterwards
        measures from where the user actually is rather than from wherever they
        last happened to select something.
        """
        self.anchor = position
        self._base = set(self.frames)

    def toggle(self, position: int) -> None:
        """Add a frame to the selection, or take it out again."""
        if position in self.frames:
            self.frames.discard(position)
        else:
            self.frames.add(position)
        self.point_at(position)

    def begin_extend(self, position: int) -> None:
        """Start a Shift-extension, keeping what is already selected.

        With no anchor -- nothing pointed at yet -- the extension starts here,
        so Shift on a fresh lamella selects one frame rather than nothing.
        """
        self._base = set(self.frames)
        if self.anchor is None:
            self.anchor = position

    def extend_to(self, position: int) -> None:
        """Select everything between the anchor and *position*, inclusive."""
        if self.anchor is None:
            self.anchor = position
        low, high = sorted((self.anchor, position))
        self.frames = self._base | set(range(low, high + 1))

    def describe(self) -> str:
        """Return a short phrase naming what is selected, for a confirmation."""
        if self.frames:
            return _counted(len(self.frames), "frame", "frames")
        if self.lamellae:
            return _counted(len(self.lamellae), "lamella", "lamellae")
        if self.projects:
            return _counted(len(self.projects), "project", "projects")
        return "this frame"


def _counted(count: int, singular: str, plural: str) -> str:
    """Return ``"1 lamella"`` or ``"4 lamellae"``."""
    return f"{count} {singular if count == 1 else plural}"
