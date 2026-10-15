"""What the window's mixins use of the window they are mixed into, declared for mypy.

:class:`~.main_window.CurationWindow` is assembled from
:class:`~.operation_runner.OperationRunner`, :class:`~.operation_actions.OperationActions`
and :class:`~.label_tab.LabelTab`, each reading attributes that the window or another
mixin defines. **Only a type checker reads this**: each mixin takes :class:`MixinBase`
as a base under ``TYPE_CHECKING`` and ``object`` otherwise, so it adds nothing at run
time. An attribute a mixin starts using is declared here, or mypy reports it.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qtpy.QtCore import (
        QSettings,
    )
    from qtpy.QtWidgets import QMainWindow

    from ...progress import Cancellation
    from ..controller import CurationController
    from ..export_controller import ExportController
    from ..history_controller import HistoryController
    from ..label_controller import LabelController
    from ..loader import BackgroundLoader
    from ..metadata_controller import MetadataController
    from ..operations_controller import OperationsController
    from ..report import OperationReport
    from ..training_controller import TrainingController
    from ..worker import BackgroundOperation
    from .curate_panel import CuratePanel
    from .export_panel import ExportPanel
    from .label_panel import LabelPanel
    from .metadata_panel import MetadataPanel


class CurationWindowParts(Protocol):
    """The window's controllers, panels, signals and methods the mixins call."""

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

    operations: BackgroundOperation
    label_loader: BackgroundLoader

    def _flash(self, message: str, milliseconds: int = 2000) -> None: ...

    def rescan(self) -> None: ...

    def _remember_model(self) -> None: ...

    def _remember_device(self) -> str: ...

    def _run_operation(
        self,
        description: str,
        work: Callable[[], Any],
        on_done: Callable[[Any], object],
        *,
        cancellation: Cancellation | None = None,
    ) -> None: ...

    def _run_stoppable(
        self,
        work: Callable[[Callable[[int, int], None], Cancellation], Any],
        on_done: Callable[[Any], object],
        description: str,
    ) -> None: ...

    def _operation_running(self) -> bool: ...

    def _operate(
        self,
        title: str,
        run: Callable[..., OperationReport],
        *,
        cancellation: Cancellation | None = None,
        rescan: bool = False,
        confirm: Callable[[str, OperationReport, str], bool] | None = None,
        notes: tuple[str, ...] = (),
    ) -> None: ...

    def _operation_applied(
        self, title: str, report: OperationReport, rescan: bool = False
    ) -> None: ...

    def _confirm_report(self, title: str, report: OperationReport, action: str) -> bool: ...

    def _show_report(self, title: str, report: OperationReport) -> None: ...

    def _editable_records(self) -> list | None: ...

    def _scope_notes(self) -> tuple[str, ...]: ...


if TYPE_CHECKING:  # pragma: no cover - typing only

    class MixinBase(CurationWindowParts, QMainWindow):
        """What a mixin is to mypy: the window's parts, on a window a dialog can be parented to."""
