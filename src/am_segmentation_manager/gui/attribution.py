"""Who the window attributes its changes to, in one place for all of them.

Every controller that records must carry the same author: a name adopted in
Settings that reaches all but one of them has the next thing written credited
to the name the user has just replaced. So they share this object rather than
each holding a recorder of its own, and adopting a name is one assignment here.

A holder rather than a list of who to tell, because a list is kept by hand and
a controller can be added without joining it. Qt-free, like the controllers
that hold it.
"""
from __future__ import annotations

from ..history import Recorder


class Attribution:
    """The recorder every part of the window records through.

    One object shared by every controller that writes, so a controller is told
    about a new author by holding this rather than by being on a list.

    Attributes:
        recorder: Who what is recorded from now on is credited to. Replaced
            whole by :meth:`adopt`; never mutated in place, a
            :class:`~am_segmentation_manager.history.Recorder` being frozen.
    """

    def __init__(self, recorder: Recorder | None = None) -> None:
        """Hold *recorder*, or resolve the author the way a run would."""
        self.recorder = recorder if recorder is not None else Recorder.for_run()

    def adopt(self, name: str | None = None) -> Recorder:
        """Resolve the author again, and return who that turned out to be.

        Called after the name on disk has changed, so it is re-read rather than
        assumed: `resolve_author` decides between what was passed, the
        environment and the config file, and only it knows that order.
        """
        self.recorder = Recorder.for_run(name)
        return self.recorder
