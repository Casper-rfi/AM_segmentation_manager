"""Tests for the curation view model.

The controller is driven with a stub scanner and stub loaders, so these tests
exercise navigation, targeting and the editing guard without a display, without
the optional image dependencies, and (except where a real move is made) without
touching the filesystem.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from am_segmentation_manager.config import SegmentationConfig
from am_segmentation_manager.gui.cache import FrameCache, LoadRequest
from am_segmentation_manager.gui.controller import (
    MIXED,
    CurationController,
    EditingDisabled,
)
from am_segmentation_manager.models import (
    AdaptiveMillingStage,
    ImageRecord,
    Lamella,
    Project,
)
from am_segmentation_manager.scanner import ProjectIndex

CONFIG = SegmentationConfig(
    class_colors={0: (0, 0, 0), 1: (255, 0, 0)}, class_labels={0: "background", 1: "GIS"}
)


def make_stage(name: str, number: str, count: int, *, fib: bool = True, plots: bool = True):
    images = []
    for n in range(count):
        images.append(ImageRecord(image_path=Path(f"{name}/sem/img_{n:03d}_SEM.tif"), modality="SEM"))
        if fib:
            images.append(
                ImageRecord(image_path=Path(f"{name}/fib/img_{n:03d}_FIB.tif"), modality="FIB")
            )
    return AdaptiveMillingStage(
        name=name,
        path=Path(name),
        number=number,
        images=images,
        plots=[Path(f"{name}/plots/img_{n:03d}_plot.png") for n in range(count)] if plots else [],
    )


def make_project(name: str, lamella_specs: list[tuple[str, int]]) -> Project:
    lamellae = []
    for position, (lamella_name, frame_count) in enumerate(lamella_specs, start=1):
        lamellae.append(
            Lamella(
                name=lamella_name,
                path=Path(lamella_name),
                number=f"{position:02d}",
                stages=[make_stage(f"{lamella_name}_stage", "01", frame_count)],
            )
        )
    return Project(name=name, path=Path(name), lamellae=lamellae, number="001")


def make_controller(projects: list[Project] | None = None) -> CurationController:
    projects = projects if projects is not None else [
        make_project("project-a", [("01-alpha", 4), ("02-beta", 2)]),
        make_project("project-b", [("01-gamma", 3)]),
    ]

    def scan(path: Path) -> ProjectIndex:
        index = ProjectIndex(start_path=path)
        for project in projects:
            index.add(project)
        return index

    cache = FrameCache(lambda image, label=None: f"img:{image.name}", lambda plot: f"plot:{plot.name}")
    controller = CurationController(cache=cache, config=CONFIG, scan=scan)
    controller.scan(Path("/scan-root"))
    return controller


# ---------------------------------------------------------------------------
# Scanning and selection
# ---------------------------------------------------------------------------

def test_scan_selects_the_first_project_and_lamella():
    controller = make_controller()

    assert len(controller.projects) == 2
    assert controller.project.name == "project-a"
    assert controller.lamella.name == "01-alpha"
    assert len(controller.frames) == 4
    assert controller.frame_position == 0


def test_selecting_a_project_moves_to_its_first_lamella():
    controller = make_controller()
    controller.step_frame(2)

    controller.select_project(1)

    assert controller.project.name == "project-b"
    assert controller.lamella.name == "01-gamma"
    assert controller.frame_position == 0


def test_scan_with_no_projects_leaves_everything_empty():
    controller = make_controller(projects=[])

    assert controller.projects == []
    assert controller.project is None
    assert controller.frame is None
    assert controller.frames == []


def test_lamella_with_no_frames_is_handled():
    controller = make_controller([make_project("p", [("01-empty", 0)])])

    assert controller.frames == []
    assert controller.frame is None


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def test_stepping_through_frames():
    controller = make_controller()

    controller.step_frame(1)
    assert controller.frame_position == 1
    assert controller.frame.image_number == 1

    controller.step_frame(-1)
    assert controller.frame_position == 0


def test_frame_navigation_stops_at_the_ends():
    controller = make_controller()

    controller.step_frame(-5)
    assert controller.frame_position == 0

    controller.step_frame(99)
    assert controller.frame_position == 3


def test_selecting_a_frame_out_of_range_is_clamped():
    controller = make_controller()

    controller.select_frame(99)

    assert controller.frame_position == 3


def test_lamella_navigation_stops_at_the_ends():
    controller = make_controller()

    controller.step_lamella(99)
    assert controller.lamella.name == "02-beta"

    controller.step_lamella(-99)
    assert controller.lamella.name == "01-alpha"


def test_changing_lamella_returns_to_the_first_frame():
    controller = make_controller()
    controller.step_frame(3)

    controller.step_lamella(1)

    assert controller.frame_position == 0
    assert len(controller.frames) == 2


def test_changing_lamella_discards_the_previous_cache():
    """Nothing prepared for the old lamella is worth keeping."""
    controller = make_controller()
    controller.current_image()
    assert controller.cache.cached_image(0, "SEM") is not None

    controller.step_lamella(1)

    assert controller.cache.cached_image(0, "SEM") is None
    assert controller.cache.frames == tuple(controller.frames)


# ---------------------------------------------------------------------------
# Display mode
# ---------------------------------------------------------------------------

def test_current_image_follows_the_mode():
    controller = make_controller()

    assert controller.current_image().display == "img:img_000_SEM.tif"

    controller.set_mode("fib")
    assert controller.current_image().display == "img:img_000_FIB.tif"


def test_plot_mode_has_no_acquisition():
    controller = make_controller()
    controller.set_mode("plot")

    assert controller.current_image() is None
    assert controller.current_plot().plot == "plot:img_000_plot.png"


def test_unknown_mode_raises():
    controller = make_controller()

    with pytest.raises(ValueError):
        controller.set_mode("edx")


def test_available_modes_reflect_what_the_frame_has():
    controller = make_controller([
        make_project("p", [("01-sem-only", 2)])
    ])
    assert controller.available_modes() == ["sem", "fib", "plot"]

    controller_without = make_controller([
        Project(
            name="p", path=Path("p"), number="001",
            lamellae=[Lamella(
                name="01", path=Path("01"), number="01",
                stages=[make_stage("s", "01", 1, fib=False, plots=False)],
            )],
        )
    ])
    assert controller_without.available_modes() == ["sem"]


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------

def test_pending_loads_prioritise_the_visible_modality():
    controller = make_controller()
    controller.set_mode("fib")

    pending = controller.pending_loads(radius=0)

    assert pending == [LoadRequest(0, "FIB"), LoadRequest(0, "SEM")]


def test_pending_loads_exclude_plots_unless_the_plot_view_is_on():
    controller = make_controller()

    assert all(request.modality is not None for request in controller.pending_loads())

    controller.set_mode("plot")
    assert any(request.modality is None for request in controller.pending_loads())


def test_pending_loads_follow_the_current_frame():
    controller = make_controller()
    controller.select_frame(2)

    pending = controller.pending_loads(radius=1)

    assert [request.frame_index for request in pending][:2] == [2, 2]
    assert {request.frame_index for request in pending} == {1, 2, 3}


# ---------------------------------------------------------------------------
# Workflow state reporting
# ---------------------------------------------------------------------------

def test_state_of_each_modality():
    controller = make_controller()

    assert controller.state_of("SEM") == "selected"
    assert controller.state_of("FIB") == "selected"
    assert controller.combined_state() == "selected"


def test_mixed_state_is_reported_rather_than_picking_a_side():
    """Curation deselects one modality at a time, so this is a normal state."""
    controller = make_controller()
    controller.frame.record("SEM").image_path = Path("sem/deselected/img_000_SEM.tif")

    assert controller.state_of("SEM") == "deselected"
    assert controller.combined_state() == MIXED


def test_target_state_follows_the_target():
    controller = make_controller()
    controller.frame.record("SEM").image_path = Path("sem/corrected/img_000_SEM.tif")

    controller.set_target("SEM")
    assert controller.target_state() == "corrected"

    controller.set_target("FIB")
    assert controller.target_state() == "selected"

    controller.set_target("both")
    assert controller.target_state() == MIXED


def test_frame_states_cover_the_whole_lamella():
    controller = make_controller()
    controller.frames[2].record("SEM").image_path = Path("sem/deselected/img_002_SEM.tif")
    controller.frames[2].record("FIB").image_path = Path("fib/deselected/img_002_FIB.tif")

    assert controller.frame_states() == ["selected", "selected", "deselected", "selected"]


def test_state_of_a_missing_modality_is_none():
    controller = make_controller([
        Project(
            name="p", path=Path("p"), number="001",
            lamellae=[Lamella(
                name="01", path=Path("01"), number="01",
                stages=[make_stage("s", "01", 1, fib=False)],
            )],
        )
    ])

    assert controller.state_of("FIB") is None
    assert controller.combined_state() == "selected"


# ---------------------------------------------------------------------------
# Edit targeting
# ---------------------------------------------------------------------------

def test_target_records_for_each_target():
    controller = make_controller()

    controller.set_target("SEM")
    assert [r.modality for r in controller.target_records()] == ["SEM"]

    controller.set_target("FIB")
    assert [r.modality for r in controller.target_records()] == ["FIB"]

    controller.set_target("both")
    assert [r.modality for r in controller.target_records()] == ["SEM", "FIB"]


def test_unknown_target_raises():
    controller = make_controller()

    with pytest.raises(ValueError):
        controller.set_target("BOTH")


# ---------------------------------------------------------------------------
# The editing guard
# ---------------------------------------------------------------------------

def test_state_changes_are_refused_until_editing_is_switched_on():
    """Guards against changing data on disk while only looking at it."""
    controller = make_controller()

    with pytest.raises(EditingDisabled):
        controller.apply_state("deselected")


def test_enabling_editing_allows_changes(tmp_path):
    controller = make_controller()
    controller.set_editing_enabled(True)
    frame = controller.frame
    _place_on_disk(tmp_path, frame)

    controller.set_target("SEM")
    changed = controller.apply_state("deselected")

    assert changed == 1
    assert controller.state_of("SEM") == "deselected"
    assert controller.state_of("FIB") == "selected"


def test_applying_to_both_moves_both_modalities(tmp_path):
    controller = make_controller()
    controller.set_editing_enabled(True)
    controller.set_target("both")
    _place_on_disk(tmp_path, controller.frame)

    changed = controller.apply_state("deselected")

    assert changed == 2
    assert controller.combined_state() == "deselected"


def test_moving_out_of_corrected_still_needs_confirmation(tmp_path):
    controller = make_controller()
    controller.set_editing_enabled(True)
    controller.set_target("both")
    _place_on_disk(tmp_path, controller.frame)
    controller.apply_state("corrected")

    with pytest.raises(PermissionError):
        controller.apply_state("selected")

    assert controller.combined_state() == "corrected"
    assert controller.apply_state("selected", confirm_corrected=True) == 2


def test_applying_to_an_absent_modality_changes_nothing():
    controller = make_controller([
        Project(
            name="p", path=Path("p"), number="001",
            lamellae=[Lamella(
                name="01", path=Path("01"), number="01",
                stages=[make_stage("s", "01", 1, fib=False)],
            )],
        )
    ])
    controller.set_editing_enabled(True)
    controller.set_target("FIB")

    assert controller.apply_state("deselected") == 0


def _place_on_disk(tmp_path: Path, frame) -> None:
    """Give a frame's records real files, so transitions can actually move them."""
    for record in frame.records:
        path = tmp_path / record.image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("image")
        record.image_path = path


# ---------------------------------------------------------------------------
# The edit target follows the view
# ---------------------------------------------------------------------------

def test_target_follows_the_view_mode():
    """Changing the state of the image you cannot see is almost always a slip."""
    controller = make_controller()

    controller.set_mode("fib")
    assert controller.target == "FIB"

    controller.set_mode("sem")
    assert controller.target == "SEM"


def test_plot_view_leaves_the_target_alone():
    controller = make_controller()
    controller.set_target("both")

    controller.set_mode("plot")

    assert controller.target == "both"


def test_following_can_be_switched_off():
    controller = make_controller()
    controller.follow_mode = False
    controller.set_target("both")

    controller.set_mode("fib")

    assert controller.target == "both"


def test_target_can_still_be_set_explicitly_after_a_mode_change():
    controller = make_controller()
    controller.set_mode("sem")

    controller.set_target("both")

    assert controller.target == "both"


# ---------------------------------------------------------------------------
# Edit scope
# ---------------------------------------------------------------------------

def test_frame_scope_covers_only_the_current_frame():
    controller = make_controller()
    controller.set_target("both")

    assert len(controller.scoped_records()) == 2


def test_lamella_scope_covers_every_frame_of_the_lamella():
    controller = make_controller()
    controller.set_target("both")
    controller.set_scope("lamella")

    # 4 frames x 2 modalities
    assert len(controller.scoped_records()) == 8


def test_project_scope_covers_every_lamella():
    controller = make_controller()
    controller.set_target("both")
    controller.set_scope("project")

    # (4 + 2) frames x 2 modalities
    assert len(controller.scoped_records()) == 12


def test_scope_respects_the_modality_target():
    controller = make_controller()
    controller.set_scope("lamella")
    controller.set_target("SEM")

    records = controller.scoped_records()

    assert len(records) == 4
    assert {record.modality for record in records} == {"SEM"}


def test_unknown_scope_raises():
    controller = make_controller()

    with pytest.raises(ValueError):
        controller.set_scope("everything")


def test_bulk_change_applies_across_the_lamella(tmp_path):
    controller = make_controller()
    controller.set_editing_enabled(True)
    controller.set_target("both")
    controller.set_scope("lamella")
    for frame in controller.frames:
        _place_on_disk(tmp_path, frame)

    changed = controller.apply_state("deselected")

    assert changed == 8
    assert all(state == "deselected" for state in controller.frame_states())


# ---------------------------------------------------------------------------
# Which records actually hold proofread work
# ---------------------------------------------------------------------------

def test_corrected_records_lists_only_corrected_ones(tmp_path):
    """A frame with a corrected SEM and a deselected FIB risks only the SEM."""
    controller = make_controller()
    controller.set_editing_enabled(True)
    frame = controller.frame
    _place_on_disk(tmp_path, frame)

    controller.set_target("SEM")
    controller.apply_state("corrected")
    controller.set_target("FIB")
    controller.apply_state("deselected")

    controller.set_target("both")
    corrected = controller.corrected_records()

    assert [record.modality for record in corrected] == ["SEM"]


def test_corrected_records_is_empty_when_nothing_is_corrected():
    controller = make_controller()

    assert controller.corrected_records() == []


def test_the_target_starts_on_the_modality_first_shown():
    """The window opens in SEM view, so SEM is what a state change means."""
    controller = make_controller()

    assert controller.mode == "sem"
    assert controller.target == "SEM"


# ---------------------------------------------------------------------------
# Plot navigation keeps up
# ---------------------------------------------------------------------------

def test_plot_view_prepares_the_whole_lamella():
    """Plots decode slower than a held-down arrow key, so all are prepared."""
    controller = make_controller()
    controller.set_mode("plot")

    plot_requests = [r for r in controller.pending_loads() if r.modality is None]

    assert len(plot_requests) == len(controller.frames)


def test_plot_preparation_reaches_the_far_end_of_the_lamella():
    controller = make_controller()
    controller.set_mode("plot")
    controller.select_frame(0)

    indices = {r.frame_index for r in controller.pending_loads() if r.modality is None}

    assert max(indices) == len(controller.frames) - 1


def test_acquisitions_still_use_the_narrow_radius():
    """Images are fast enough that preparing the neighbourhood is enough."""
    controller = make_controller()
    controller.set_mode("sem")

    indices = {r.frame_index for r in controller.pending_loads(radius=1)}

    assert indices == {0, 1}


# ---------------------------------------------------------------------------
# Paths, for copying out of the window
# ---------------------------------------------------------------------------

def test_project_and_lamella_paths_are_absolute():
    """Copied paths must be pasteable into a terminal, so never relative."""
    controller = make_controller()

    project = controller.project_path()
    lamella = controller.lamella_path()

    assert project is not None and project.is_absolute()
    assert lamella is not None and lamella.is_absolute()
    assert project.name == "project-a"
    assert lamella.name == "01-alpha"


def test_paths_can_be_asked_for_by_row_rather_than_selection():
    """The right-click menu acts on the row under the cursor, not the selected
    one, so the accessors take a position."""
    controller = make_controller()

    assert controller.project_path(1).name == "project-b"
    assert controller.lamella_path(1).name == "02-beta"
    # The selection is unchanged by asking about another row.
    assert controller.project_position == 0
    assert controller.lamella_position == 0


def test_paths_are_none_when_the_position_does_not_exist():
    controller = make_controller()

    assert controller.project_path(99) is None
    assert controller.lamella_path(99) is None
    assert controller.project_path(-1) is None


def test_paths_are_none_before_anything_is_scanned():
    cache = FrameCache(lambda image, label=None: "", lambda plot: "")
    controller = CurationController(
        cache=cache, config=CONFIG, scan=lambda path: ProjectIndex(start_path=path)
    )

    assert controller.project_path() is None
    assert controller.lamella_path() is None
    assert controller.current_image_path() is None


def test_current_image_path_follows_the_mode():
    controller = make_controller()

    controller.set_mode("sem")
    sem = controller.current_image_path()
    controller.set_mode("fib")
    fib = controller.current_image_path()

    assert sem is not None and sem.name.endswith("_SEM.tif")
    assert fib is not None and fib.name.endswith("_FIB.tif")


def test_current_image_path_follows_the_frame():
    controller = make_controller()

    controller.set_mode("sem")
    first = controller.current_image_path()
    controller.step_frame(1)
    second = controller.current_image_path()

    assert first != second
    assert second.name == "img_001_SEM.tif"


def test_current_image_path_is_none_in_plot_mode():
    """Plot mode has no single acquisition behind it."""
    controller = make_controller()

    controller.set_mode("plot")

    assert controller.current_image_path() is None
    assert controller.current_plot_path() is not None
