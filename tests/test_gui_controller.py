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
    lamella_status,
    project_status,
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
# Rescanning
# ---------------------------------------------------------------------------
#
# The stub scanner reads the list it was given on every call, so a test changes
# what the next scan finds by changing that list.

def test_rescanning_stays_where_the_view_was():
    controller = make_controller()
    controller.select_project(1)
    controller.step_frame(2)

    controller.rescan()

    assert controller.project.name == "project-b"
    assert controller.lamella.name == "01-gamma"
    assert controller.frame_position == 2


def test_rescanning_follows_a_project_that_has_been_renumbered():
    projects = [
        make_project("project-a", [("01-alpha", 4)]),
        make_project("project-b", [("01-gamma", 3)]),
    ]
    controller = make_controller(projects)
    controller.select_project(1)
    projects.insert(0, make_project("project-new", [("01-delta", 2)]))

    controller.rescan()

    assert controller.project_position == 2, "the index moved; the project did not"
    assert controller.project.name == "project-b"


def test_rescanning_starts_over_when_what_was_selected_has_gone():
    projects = [
        make_project("project-a", [("01-alpha", 4)]),
        make_project("project-b", [("01-gamma", 3)]),
    ]
    controller = make_controller(projects)
    controller.select_project(1)
    del projects[1]

    controller.rescan()

    assert controller.project.name == "project-a", (
        "a lamella that has gone is one the reviewer has to be shown they left"
    )
    assert controller.lamella.name == "01-alpha"


def test_rescanning_before_anything_was_scanned_is_refused():
    controller = CurationController(cache=FrameCache(), config=CONFIG)

    with pytest.raises(RuntimeError, match="Nothing has been scanned"):
        controller.rescan()


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
    declare(controller.frame.record("SEM"), "deselected")

    assert controller.state_of("SEM") == "deselected"
    assert controller.combined_state() == MIXED


def test_target_state_follows_the_target():
    controller = make_controller()
    declare(controller.frame.record("SEM"), "corrected")

    controller.set_target("SEM")
    assert controller.target_state() == "corrected"

    controller.set_target("FIB")
    assert controller.target_state() == "selected"

    controller.set_target("both")
    assert controller.target_state() == MIXED


def test_frame_states_cover_the_whole_lamella():
    controller = make_controller()
    declare(controller.frames[2].record("SEM"), "deselected")
    declare(controller.frames[2].record("FIB"), "deselected")

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
# What a state change applies to
#
# The precedence is the whole of the model: frames beat ticked lamellae, which
# beat ticked projects, and with none of them it is the frame on screen.
# ---------------------------------------------------------------------------

def test_nothing_ticked_covers_only_the_current_frame():
    controller = make_controller()
    controller.set_target("both")

    assert len(controller.scoped_records()) == 2
    assert controller.describe_selection() == "this frame"


def test_a_ticked_lamella_covers_every_frame_of_it():
    controller = make_controller()
    controller.set_target("both")
    controller.set_lamella_checked(0, True)

    # 4 frames x 2 modalities
    assert len(controller.scoped_records()) == 8


def test_a_ticked_project_covers_every_lamella():
    controller = make_controller()
    controller.set_target("both")
    controller.set_project_checked(0, True)

    # (4 + 2) frames x 2 modalities
    assert len(controller.scoped_records()) == 12


def test_unticking_puts_it_back(tmp_path):
    controller = make_controller()
    controller.set_target("both")
    controller.set_lamella_checked(0, True)
    controller.set_lamella_checked(0, False)

    assert len(controller.scoped_records()) == 2


def test_a_selection_respects_the_modality_target():
    controller = make_controller()
    controller.set_lamella_checked(0, True)
    controller.set_target("SEM")

    records = controller.scoped_records()

    assert len(records) == 4
    assert {record.modality for record in records} == {"SEM"}


def test_picked_frames_beat_a_ticked_lamella():
    """Finest-first, because it is the fail-safe direction: a forgotten frame
    selection makes a change smaller than intended, not larger."""
    controller = make_controller()
    controller.set_target("both")
    controller.set_lamella_checked(0, True)
    controller.selection.toggle(0)

    assert len(controller.scoped_records()) == 2


def test_a_ticked_lamella_beats_a_ticked_project():
    controller = make_controller()
    controller.set_target("both")
    controller.set_project_checked(0, True)
    controller.set_lamella_checked(0, True)

    assert len(controller.scoped_records()) == 8


def test_picked_frames_are_dropped_when_the_lamella_changes():
    """A frame index only means anything within one lamella."""
    controller = make_controller()
    controller.selection.toggle(2)

    controller.select_lamella(1)

    assert controller.selected_frames() == set()


def test_a_tick_survives_changing_project():
    """Ticks are keyed by path, so they outlive the row numbers they were made
    on -- a lamella ticked in one project is still ticked after visiting
    another."""
    controller = make_controller()
    controller.set_target("both")
    controller.set_lamella_checked(1, True)
    ticked = controller.lamellae[1]

    controller.select_lamella(0)

    assert controller.lamella_checks() == [False, True]
    records = controller.scoped_records()
    assert len(records) == 4  # the second lamella has 2 frames x 2 modalities
    assert all(
        str(ticked.path) in str(record.image_path) for record in records
    )


def test_a_change_applies_across_a_ticked_lamella(tmp_path):
    controller = make_controller()
    controller.set_editing_enabled(True)
    controller.set_target("both")
    controller.set_lamella_checked(0, True)
    for frame in controller.frames:
        _place_on_disk(tmp_path, frame)

    changed = controller.apply_state("deselected")

    assert changed == 8
    assert all(state == "deselected" for state in controller.frame_states())


def test_a_change_applies_to_only_the_picked_frames(tmp_path):
    controller = make_controller()
    controller.set_editing_enabled(True)
    controller.set_target("both")
    for frame in controller.frames:
        _place_on_disk(tmp_path, frame)
    controller.selection.toggle(0)
    controller.selection.toggle(2)

    controller.apply_state("deselected")

    states = controller.frame_states()
    assert [states[0], states[2]] == ["deselected", "deselected"]
    assert [states[1], states[3]] == ["selected", "selected"]


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


# ---------------------------------------------------------------------------
# Curation status, for the project and lamella lists
# ---------------------------------------------------------------------------

def status_lamella(name: str, records: list[ImageRecord]) -> Lamella:
    """A one-stage lamella holding exactly the given records."""
    return Lamella(
        name=name,
        path=Path(name),
        number="01",
        stages=[AdaptiveMillingStage(name="stage", path=Path(name) / "stage", images=records)],
    )


def declare(record: ImageRecord, state: str) -> ImageRecord:
    """Put *record* into *state*, as the scanner does from the state file."""
    record.selected = state == "selected"
    record.corrected = state == "corrected"
    return record


def image(state: str, modality: str = "SEM", *, name: str = "img_000") -> ImageRecord:
    """A record declared to be in *state*."""
    directory = "sem" if modality == "SEM" else "fib"
    labels = "sem_labels" if modality == "SEM" else "fib_labels"
    return ImageRecord(
        image_path=Path(f"lam/stage/{directory}/{name}.tif"),
        labels=[Path(f"lam/stage/{labels}/{name}.tif")],
        modality=modality,
        selected=state == "selected",
        corrected=state == "corrected",
    )


def test_a_modality_with_no_images_is_empty_not_done():
    """The regression most likely to slip through: "nothing left awaiting a
    decision" is vacuously true for a modality that has no images at all, and
    many older datasets have no FIB whatsoever."""
    status = lamella_status(status_lamella("lam", [image("corrected", "SEM")]))

    assert status.sem == "done"
    assert status.fib == "empty"


def test_all_selected_is_not_started():
    status = lamella_status(status_lamella("lam", [image("selected"), image("selected", name="b")]))
    assert status.sem == "not started"


def test_deselected_and_corrected_together_count_as_done():
    """Done means nothing is left awaiting a decision, not that all were kept."""
    records = [image("deselected"), image("corrected", name="b")]
    assert lamella_status(status_lamella("lam", records)).sem == "done"


def test_a_mix_of_decided_and_undecided_is_in_progress():
    records = [image("corrected"), image("selected", name="b")]
    assert lamella_status(status_lamella("lam", records)).sem == "in progress"


def test_modalities_are_reported_separately():
    records = [image("corrected", "SEM"), image("selected", "FIB")]
    status = lamella_status(status_lamella("lam", records))

    assert status.sem == "done"
    assert status.fib == "not started"
    assert status.summary == "SEM: done · FIB: not started"


def test_a_project_is_only_done_when_every_lamella_is():
    project = Project(
        name="p",
        path=Path("p"),
        lamellae=[
            status_lamella("a", [image("corrected")]),
            status_lamella("b", [image("selected")]),
        ],
    )

    assert project_status(project).sem == "in progress"


def test_a_project_of_finished_lamellae_is_done():
    project = Project(
        name="p",
        path=Path("p"),
        lamellae=[
            status_lamella("a", [image("corrected")]),
            status_lamella("b", [image("deselected")]),
        ],
    )

    assert project_status(project).sem == "done"


def test_lamellae_without_a_modality_do_not_drag_a_project_down():
    """A project is not part-finished because half its lamellae never had FIB."""
    project = Project(
        name="p",
        path=Path("p"),
        lamellae=[
            status_lamella("a", [image("corrected", "FIB")]),
            status_lamella("b", [image("corrected", "SEM")]),
        ],
    )

    assert project_status(project).fib == "done"


def test_a_project_with_no_lamellae_is_empty():
    project = Project(name="p", path=Path("p"), lamellae=[])
    status = project_status(project)

    assert status.sem == "empty"
    assert status.fib == "empty"


def test_inconsistent_images_need_repair_rather_than_reading_as_done(tmp_path: Path):
    """A group whose images disagree with their shared label cannot read as done.

    Until repair-state runs, the state shown for the group is the image's rather
    than the label's, so any other answer would be a claim the files do not
    support.

    Built on disk because build_group finds an image's siblings by scanning the
    stage directory -- a record pointed at a file that does not exist forms a
    group of one, and a group of one is always consistent.
    """
    stage = tmp_path / "lam" / "stage"
    shared_label = stage / "sem_labels" / "corrected" / "img_000.tif"
    # sem/ moved into corrected/ with the label; sem_5kv/ was left behind. There
    # is no sem_5kv_labels/, so both image sets resolve to the shared sem_labels.
    for path in (shared_label, stage / "sem" / "corrected" / "img_000.tif",
                 stage / "sem_5kv" / "img_000.tif"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    records = [
        ImageRecord(
            image_path=stage / "sem" / "corrected" / "img_000.tif",
            labels=[shared_label], modality="SEM", variant=None,
        ),
        ImageRecord(
            image_path=stage / "sem_5kv" / "img_000.tif",
            labels=[shared_label], modality="SEM", variant="5kv",
        ),
    ]
    lamella = Lamella(
        name="lam", path=stage.parent, number="01",
        stages=[AdaptiveMillingStage(name="stage", path=stage, images=records)],
    )

    assert lamella_status(lamella).sem == "needs repair"


def test_needs_repair_overrides_done_for_the_whole_project(tmp_path: Path):
    """One unrepaired lamella must not be averaged away by finished ones."""
    stage = tmp_path / "lam" / "stage"
    shared_label = stage / "sem_labels" / "corrected" / "img_000.tif"
    for path in (shared_label, stage / "sem" / "corrected" / "img_000.tif",
                 stage / "sem_5kv" / "img_000.tif"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    broken = Lamella(
        name="lam", path=stage.parent, number="01",
        stages=[AdaptiveMillingStage(name="stage", path=stage, images=[
            ImageRecord(image_path=stage / "sem" / "corrected" / "img_000.tif",
                        labels=[shared_label], modality="SEM"),
            ImageRecord(image_path=stage / "sem_5kv" / "img_000.tif",
                        labels=[shared_label], modality="SEM", variant="5kv"),
        ])],
    )
    project = Project(
        name="p", path=tmp_path,
        lamellae=[status_lamella("done-one", [image("corrected")]), broken],
    )

    assert project_status(project).sem == "needs repair"


def test_statuses_line_up_with_the_lists_they_annotate():
    controller = make_controller()

    assert len(controller.project_statuses()) == len(controller.projects)
    assert len(controller.lamella_statuses()) == len(controller.lamellae)
