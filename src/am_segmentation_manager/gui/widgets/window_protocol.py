"""What the window's mixin uses of the window it is mixed into, declared for mypy.

:class:`~.operation_actions.OperationActions` reads attributes that
:class:`~.main_window.CurationWindow` defines. **Only a type checker reads this**: the
mixin takes :class:`MixinBase` as a base under ``TYPE_CHECKING`` and ``object``
otherwise, so it adds nothing at run time. An attribute the mixin starts using is
declared here, or mypy reports it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qtpy.QtCore import (
        QSettings,
    )
    from qtpy.QtWidgets import QMainWindow

    from ..controller import CurationController
    from ..export_controller import ExportController
    from ..history_controller import HistoryController
    from ..metadata_controller import MetadataController
    from ..operations_controller import OperationsController
    from ..report import OperationReport
    from ..training_controller import TrainingController
    from .curate_panel import CuratePanel
    from .export_panel import ExportPanel
    from .metadata_panel import MetadataPanel
    from .operation_runner import OperationRunner


class CurationWindowParts(Protocol):
    """The window's controllers, panels, runner and methods the mixin calls."""

    controller: CurationController
    export_controller: ExportController
    metadata_controller: MetadataController
    operations_controller: OperationsController
    history_controller: HistoryController
    training_controller: TrainingController
    settings: QSettings

    curate_panel: CuratePanel
    export_panel: ExportPanel
    metadata_panel: MetadataPanel

    runner: OperationRunner

    def _flash(self, message: str, milliseconds: int = 2000) -> None: ...

    def _remember_model(self) -> None: ...

    def _remember_device(self) -> str: ...

    def _operation_applied(
        self, title: str, report: OperationReport, rescan: bool = False
    ) -> None: ...


if TYPE_CHECKING:  # pragma: no cover - typing only

    class MixinBase(CurationWindowParts, QMainWindow):
        """What the mixin is to mypy: the window's parts, on a window a dialog can be parented to."""
