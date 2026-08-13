"""The curation window: layout, wiring and keyboard shortcuts.

This module connects the three panels and the two sliders to the controller.
It makes no decisions of its own -- it asks the controller what to show, and
tells the controller what the user did.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...operations.selection import RecordState
from ..controller import CurationController, EditingDisabled
from ..loader import BackgroundLoader
from .frame_strip import FrameStateStrip
from .image_panel import ImagePanel
from .project_panel import ProjectPanel
from .state_panel import StatePanel

logger = logging.getLogger(__name__)

try:  # Qt 6 moved QShortcut out of QtWidgets.
    from qtpy.QtGui import QKeySequence, QShortcut
except ImportError:  # pragma: no cover - depends on the installed binding
    from qtpy.QtGui import QKeySequence
    from qtpy.QtWidgets import QShortcut


class CurationWindow(QMainWindow):
    """Main window for reviewing and curating adaptive milling data."""

    #: Emitted to the background loader thread; see :class:`BackgroundLoader`.
    request_load = Signal(list, int)

    def __init__(self, controller: CurationController) -> None:
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Adaptive milling segmentation curation")
        self.resize(1600, 950)

        self.project_panel = ProjectPanel()
        self.image_panel = ImagePanel(controller.config)
        self.state_panel = StatePanel()

        self._build_layout()
        self._connect()
        self._add_shortcuts()

        self.loader = BackgroundLoader(controller.cache)
        self.request_load.connect(self.loader.worker.load_batch)
        self.loader.worker.loaded.connect(self._on_loaded)

        self.refresh()

    # -- construction ------------------------------------------------------

    def _build_layout(self) -> None:
        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.addWidget(self.image_panel, stretch=1)
        centre_layout.addLayout(self._build_sliders())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.project_panel)
        splitter.addWidget(centre)
        splitter.addWidget(self.state_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1040, 280])
        self.setCentralWidget(splitter)

    def _build_sliders(self) -> QVBoxLayout:
        layout = QVBoxLayout()

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self._frame_slider_moved)
        frame_row.addWidget(self.frame_slider, stretch=1)
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(90)
        frame_row.addWidget(self.frame_label)
        layout.addLayout(frame_row)

        self.strip = FrameStateStrip()
        self.strip.frame_clicked.connect(self._frame_slider_moved)
        layout.addWidget(self.strip)

        lamella_row = QHBoxLayout()
        lamella_row.addWidget(QLabel("Lamella"))
        self.lamella_slider = QSlider(Qt.Horizontal)
        self.lamella_slider.valueChanged.connect(self._lamella_slider_moved)
        lamella_row.addWidget(self.lamella_slider, stretch=1)
        self.lamella_label = QLabel("0 / 0")
        self.lamella_label.setMinimumWidth(90)
        lamella_row.addWidget(self.lamella_label)
        layout.addLayout(lamella_row)

        return layout

    def _connect(self) -> None:
        self.project_panel.scan_requested.connect(self.scan)
        self.project_panel.project_selected.connect(self._select_project)
        self.project_panel.lamella_selected.connect(self._select_lamella)

        self.image_panel.mode_changed.connect(self._set_mode)
        self.image_panel.overlay_toggled.connect(self._set_overlay)
        self.image_panel.alpha_changed.connect(lambda _alpha: self._render())

        self.state_panel.editing_toggled.connect(self.controller.set_editing_enabled)
        self.state_panel.target_changed.connect(self._set_target)
        self.state_panel.state_requested.connect(self.apply_state)

    def _add_shortcuts(self) -> None:
        """Bind the keys that make curating a long lamella bearable."""
        bindings = (
            ("Right", lambda: self._step_frame(1)),
            ("Left", lambda: self._step_frame(-1)),
            ("Down", lambda: self._step_lamella(1)),
            ("Up", lambda: self._step_lamella(-1)),
            ("1", lambda: self._set_mode("sem", from_user=True)),
            ("2", lambda: self._set_mode("fib", from_user=True)),
            ("3", lambda: self._set_mode("plot", from_user=True)),
            ("Space", self._toggle_overlay),
            ("S", lambda: self.apply_state("selected")),
            ("D", lambda: self.apply_state("deselected")),
            ("C", lambda: self.apply_state("corrected")),
            ("F", self.image_panel.pair.fit),
        )
        self._shortcuts = [
            QShortcut(QKeySequence(key), self, activated=action) for key, action in bindings
        ]

    # -- user actions ------------------------------------------------------

    def scan(self, path: str) -> None:
        """Scan *path* for projects and show what was found."""
        if not path:
            return
        directory = Path(path)
        if not directory.is_dir():
            QMessageBox.warning(self, "Scan", f"Not a folder: {directory}")
            return
        try:
            self.controller.scan(directory)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            logger.exception("Scan failed")
            QMessageBox.critical(self, "Scan failed", str(exc))
            return
        self.project_panel.set_path(str(directory))
        self.refresh(reload_lists=True)

    def _select_project(self, row: int) -> None:
        self.controller.select_project(row)
        self.refresh(reload_lists=True)

    def _select_lamella(self, row: int) -> None:
        self.controller.select_lamella(row)
        self.refresh()

    def _frame_slider_moved(self, value: int) -> None:
        if value != self.controller.frame_position:
            self.controller.select_frame(value)
            self.refresh()

    def _lamella_slider_moved(self, value: int) -> None:
        if value != self.controller.lamella_position:
            self.controller.select_lamella(value)
            self.refresh(reload_lists=True)

    def _step_frame(self, delta: int) -> None:
        self.controller.step_frame(delta)
        self.refresh()

    def _step_lamella(self, delta: int) -> None:
        self.controller.step_lamella(delta)
        self.refresh(reload_lists=True)

    def _set_mode(self, mode: str, from_user: bool = False) -> None:
        if from_user and mode not in self.controller.available_modes():
            return
        self.controller.set_mode(mode)
        self.image_panel.set_mode(mode)
        self.refresh()

    def _set_overlay(self, show: bool) -> None:
        self.controller.show_overlay = show
        self._render()

    def _toggle_overlay(self) -> None:
        self.image_panel.overlay_check.toggle()

    def _set_target(self, target: str) -> None:
        self.controller.set_target(target)
        self._refresh_state_panel()

    def apply_state(self, state: RecordState) -> None:
        """Move the targeted records into *state*, confirming where required."""
        try:
            changed = self.controller.apply_state(state)
        except EditingDisabled:
            QMessageBox.information(
                self, "Editing is off",
                "Tick 'Enable editing' before changing the state of an image.",
            )
            return
        except PermissionError:
            if not self._confirm_corrected(state):
                self._refresh_state_panel()
                return
            try:
                changed = self.controller.apply_state(state, confirm_corrected=True)
            except OSError as exc:
                self._report_failure(exc)
                return
        except OSError as exc:
            self._report_failure(exc)
            return

        if changed:
            frame = self.controller.frame
            name = frame.display_name if frame is not None else "frame"
            self.state_panel.append_log(
                f"{name}: {changed} file set -> {state} ({self.controller.target})"
            )
        self.refresh()

    def _confirm_corrected(self, state: str) -> bool:
        """Ask before moving proofread data out of the corrected state."""
        files = "\n".join(
            str(record.image_path.name) for record in self.controller.target_records()
        )
        answer = QMessageBox.warning(
            self,
            "Move corrected data?",
            "These files are in the corrected state, which holds manually "
            f"proofread work:\n\n{files}\n\nMove them to '{state}'?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Yes

    def _report_failure(self, exc: Exception) -> None:
        logger.warning("State change refused: %s", exc)
        QMessageBox.critical(self, "Could not change state", str(exc))

    # -- rendering ---------------------------------------------------------

    def refresh(self, *, reload_lists: bool = False) -> None:
        """Bring the whole window back in step with the controller."""
        if reload_lists:
            self._refresh_lists()
        self._refresh_sliders()
        self._render()
        self._refresh_state_panel()
        self._schedule_loads()

    def _refresh_lists(self) -> None:
        self.project_panel.show_projects(
            [project.display_name for project in self.controller.projects],
            self.controller.project_position,
        )
        self.project_panel.show_lamellae(
            [lamella.display_name for lamella in self.controller.lamellae],
            self.controller.lamella_position,
        )
        self.project_panel.show_summary(self._summary_text())

    def _summary_text(self) -> str:
        if self.controller.project is None:
            return "No project loaded"
        counts: dict[str, int] = {}
        for state in self.controller.frame_states():
            counts[str(state)] = counts.get(str(state), 0) + 1
        breakdown = ", ".join(f"{count} {state}" for state, count in sorted(counts.items()))
        return f"{len(self.controller.frames)} frames\n{breakdown}"

    def _refresh_sliders(self) -> None:
        self._set_slider(
            self.frame_slider, self.frame_label,
            self.controller.frame_position, len(self.controller.frames),
        )
        self._set_slider(
            self.lamella_slider, self.lamella_label,
            self.controller.lamella_position, len(self.controller.lamellae),
        )
        self.strip.set_states(
            self.controller.frame_states(), self.controller.frame_position
        )

    @staticmethod
    def _set_slider(slider: QSlider, label: QLabel, position: int, total: int) -> None:
        slider.blockSignals(True)
        slider.setEnabled(total > 1)
        slider.setMaximum(max(0, total - 1))
        slider.setValue(position)
        slider.blockSignals(False)
        label.setText(f"{position + 1} / {total}" if total else "0 / 0")

    def _render(self) -> None:
        """Draw the current frame, or say why it cannot be drawn yet."""
        frame = self.controller.frame
        if frame is None:
            self.image_panel.show_empty()
            return

        self.image_panel.set_available_modes(self.controller.available_modes())
        self.image_panel.set_status(self._status_text())

        if self.controller.mode == "plot":
            if frame.plot_path is None:
                self.image_panel.show_plot(None)
            elif self.controller.cache.cached_plot(self.controller.frame_position) is None:
                self.image_panel.show_loading()
            else:
                self.image_panel.show_plot(self.controller.current_plot())
            return

        modality = "SEM" if self.controller.mode == "sem" else "FIB"
        if frame.record(modality) is None:
            self.image_panel.show_image(None, self.controller.show_overlay)
            return
        # Only read what is already prepared: a miss means the loader is still
        # working, and blocking here would freeze the window on a slow read.
        cached = self.controller.cache.cached_image(self.controller.frame_position, modality)
        if cached is None:
            self.image_panel.show_loading()
        else:
            self.image_panel.show_image(cached, self.controller.show_overlay)

    def _status_text(self) -> str:
        frame = self.controller.frame
        project = self.controller.project
        lamella = self.controller.lamella
        if frame is None or project is None or lamella is None:
            return ""
        return (
            f"{project.display_name}  |  {lamella.display_name}  |  "
            f"{frame.display_name}  |  {frame.stage}"
        )

    def _refresh_state_panel(self) -> None:
        self.state_panel.show_state(
            self.controller.target_state(),
            self.controller.state_of("SEM"),
            self.controller.state_of("FIB"),
        )
        self.state_panel.set_target(self.controller.target)
        self.state_panel.set_editing_enabled(self.controller.editing_enabled)

    def _schedule_loads(self) -> None:
        """Ask the loader thread to prepare the frames around this one.

        Taking a new token first abandons any batch still running for a
        position the user has already left.
        """
        token = self.loader.next_token()
        requests = self.controller.pending_loads()
        if requests:
            self.request_load.emit(requests, token)

    def _on_loaded(self, token: int, request) -> None:
        """Redraw when the frame on screen finishes preparing."""
        if token != self.loader.token:
            return
        if request.frame_index == self.controller.frame_position:
            self._render()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.loader.stop()
        super().closeEvent(event)
