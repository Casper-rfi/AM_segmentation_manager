"""What the window's mixins use of the window they are mixed into, declared for mypy.

:class:`~.main_window.CurationWindow` is assembled from
:class:`~.operation_actions.OperationActions` and :class:`~.label_tab.LabelTab`, each
reading attributes that the window or the other mixin defines. **Only a type checker
reads this**: each mixin takes :class:`MixinBase` as a base under ``TYPE_CHECKING`` and
``object`` otherwise, so it adds nothing at run time. An attribute a mixin starts using is declared here, or mypy reports it.
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
    from ..label_controller import LabelController
    from ..loader import BackgroundLoader
    from ..metadata_controller import MetadataController
    from ..operations_controller import OperationsController
    from ..report import OperationReport
    from ..training_controller import TrainingController
    from .curate_panel import CuratePanel
    from .export_panel import ExportPanel
    from .label_panel import LabelPanel
    from .metadata_panel import MetadataPanel
    from .operation_runner import OperationRunner


class CurationWindowParts(Protocol):
    """The window's controllers, panels, runner and methods the mixins call."""

    controller: CurationController
    export_controller: ExportController
    label_controller: LabelController
    metadata_controller: MetadataController
    operations_controller: OperationsController
    history_controller: HistoryController
    training_controller: TrainingController
    settings: QSettings

    curate_panel: CuratePanel
    label_panel: LabelPanel
    export_panel: ExportPanel
    metadata_panel: MetadataPanel

    runner: OperationRunner
    label_loader: BackgroundLoader

    def _flash(self, message: str, milliseconds: int = 2000) -> None: ...

    def rescan(self) -> None: ...

    def _remember_model(self) -> None: ...

    def _remember_device(self) -> str: ...

    def _operation_applied(
        self, title: str, report: OperationReport, rescan: bool = False
    ) -> None: ...


if TYPE_CHECKING:  # pragma: no cover - typing only

    class MixinBase(CurationWindowParts, QMainWindow):
        """What a mixin is to mypy: the window's parts, on a window a dialog can be parented to."""
