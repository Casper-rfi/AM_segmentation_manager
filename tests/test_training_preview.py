"""Choosing validation images to look at models on, and which checkpoint of a run."""
from __future__ import annotations

import shutil
from pathlib import Path

from am_segmentation_manager.operations.training_run import read_run_record
from am_segmentation_manager.training_preview import (
    PreviewImage,
    PreviewRun,
    describe_not_found,
    offered_values,
    pick_images,
    preview_checkpoint,
    preview_images,
    read_preview_run,
    split_found,
)
from am_segmentation_manager.training_runs import read_training_run
from test_training_controller import _write_epochs
from test_training_run import _described_export, _prepared


def _image(name: str, **described: str) -> PreviewImage:
    return PreviewImage(Path(f"/data/{name}.tif"), Path(f"/data/{name}_label.tif"), described)


ARCTIS_WAFFLE = _image("a", microscope_model="Arctis", project_lamella_type="waffle",
                       acceleration_voltage_kv="2.0")
ARCTIS_GRID = _image("b", microscope_model="Arctis", project_lamella_type="cells-on-grid",
                     acceleration_voltage_kv="5.0")
HYDRA_WAFFLE = _image("c", microscope_model="Hydra", project_lamella_type="waffle",
                      acceleration_voltage_kv="2.0")
EVERY = [ARCTIS_WAFFLE, ARCTIS_GRID, HYDRA_WAFFLE]


def _names(images) -> list[str]:
    return sorted(image.image.stem for image in images)


def test_an_image_one_of_the_runs_trained_on_is_not_offered():
    """A model looks good on what it has seen, and two exports split differently."""
    first = PreviewRun("first", (ARCTIS_WAFFLE, ARCTIS_GRID), frozenset({HYDRA_WAFFLE.image}))
    second = PreviewRun("second", (ARCTIS_GRID, HYDRA_WAFFLE), frozenset({ARCTIS_WAFFLE.image}))

    assert _names(preview_images([first, second])) == ["b"]
    assert _names(preview_images([first, second], unseen_only=False)) == ["a", "b", "c"]
    assert second.trained_on(ARCTIS_WAFFLE) is True and second.trained_on(ARCTIS_GRID) is False


def test_an_image_only_one_run_validated_on_is_offered_where_the_other_never_saw_it():
    first = PreviewRun("first", (ARCTIS_WAFFLE,), frozenset())
    second = PreviewRun("second", (HYDRA_WAFFLE,), frozenset())

    assert _names(preview_images([first, second])) == ["a", "c"]


def test_a_run_am_seg_did_not_start_offers_nothing_and_is_not_known_to_have_trained():
    foreign = read_preview_run("foreign", None)

    assert foreign.validation == () and foreign.trained_on(ARCTIS_WAFFLE) is None
    assert _names(preview_images([foreign, PreviewRun("ours", (ARCTIS_WAFFLE,), frozenset())])) == ["a"]


def test_an_image_two_runs_validated_on_is_described_by_the_one_that_says():
    bare = PreviewImage(ARCTIS_WAFFLE.image, ARCTIS_WAFFLE.label)
    runs = [PreviewRun("old", (bare,), frozenset()), PreviewRun("new", (ARCTIS_WAFFLE,), frozenset())]

    assert preview_images(runs)[0].describe() == "Arctis · waffle · 2.0 kV"


def test_each_thing_an_image_can_be_chosen_by_offers_the_values_there_are():
    assert offered_values(EVERY, "microscope_model") == ("Arctis", "Hydra")
    assert offered_values(EVERY, "acceleration_voltage_kv") == ("2.0", "5.0")
    assert offered_values([PreviewImage(Path("x"), Path("y"))], "microscope_model") == ()


def test_images_are_drawn_among_those_that_are_what_was_asked():
    waffles = pick_images(EVERY, {"project_lamella_type": "waffle"}, count=5)
    one = pick_images(EVERY, {"project_lamella_type": "waffle", "microscope_model": "Hydra"})

    assert _names(waffles) == ["a", "c"]
    assert _names(one) == ["c"]
    assert pick_images(EVERY, {"microscope_model": "Krios"}) == []


def test_with_everything_set_to_any_the_draw_is_over_every_image():
    assert _names(pick_images(EVERY, {"microscope_model": ""}, count=3)) == ["a", "b", "c"]
    assert len(pick_images(EVERY, None, count=2)) == 2


def test_a_seed_repeats_a_draw_whatever_order_the_images_came_in():
    many = [_image(f"image{number:02}") for number in range(40)]

    again = pick_images(list(reversed(many)), count=3, seed=7)

    assert pick_images(many, count=3, seed=7) == again
    assert pick_images(many, count=3, seed=8) != again


def test_a_run_s_images_are_read_from_what_it_keeps_with_what_each_is_of(tmp_path: Path):
    run = _prepared(tmp_path, _described_export(tmp_path))

    read = read_preview_run("round01", read_run_record(run))

    assert [image.image.name for image in read.validation] == ["b.tif"]
    assert read.validation[0].describe() == "Hydra"
    assert {path.name for path in read.training} == {"a.tif"}


def _described_beside_scores(run: Path, rows: list[tuple[str, ...]]) -> None:
    """Write what the validation images of *run* are of beside its scores, as a validation does."""
    from am_segmentation_manager.io.tables import write_csv
    from am_segmentation_manager.operations.training_run import own_scores_directory
    from am_segmentation_manager.operations.validation_run import IMAGES_FILENAME

    own_scores_directory(run).mkdir()
    write_csv(
        own_scores_directory(run) / IMAGES_FILENAME,
        ("image", "microscope_model", "acceleration_voltage_kv", "lamella"), rows,
    )


def test_a_run_with_no_images_file_is_described_by_the_one_beside_its_scores(tmp_path: Path):
    """A run prepared from an export that described nothing, and scored since."""
    from test_training_process import export_of_placeholders

    run = _prepared(tmp_path, export_of_placeholders(tmp_path))
    record = read_run_record(run)
    assert record.validation_images is None
    _described_beside_scores(run, [("b.tif", "Hydra", "2.0", "lam_01")])

    read = read_preview_run("round01", record)

    assert read.validation[0].describe() == "Hydra · 2.0 kV"
    assert offered_values(read.validation, "microscope_model") == ("Hydra",)


def test_an_images_file_out_of_step_with_the_list_describes_nothing(tmp_path: Path):
    """Rows are matched by their order, so a file of another length is no account of these."""
    from test_training_process import export_of_placeholders

    run = _prepared(tmp_path, export_of_placeholders(tmp_path))
    _described_beside_scores(run, [("b.tif", "Hydra", "2.0", ""), ("c.tif", "Arctis", "5.0", "")])

    read = read_preview_run("round01", read_run_record(run))

    assert [image.image.name for image in read.validation] == ["b.tif"]
    assert read.validation[0].describe() == ""


def test_an_image_says_more_of_itself_than_its_line_where_it_is_asked():
    image = _image(
        "a", microscope_model="Arctis", lamella="lam_01", pixel_size_nm="13.0", detector_type="ETD"
    )

    assert image.details() == ["Lamella: lam_01", "Microscope: Arctis", "Pixel size (nm): 13.0"]
    assert _image("b").details() == []


def test_a_run_is_looked_at_with_the_epoch_marked_else_its_best(tmp_path: Path):
    written = _write_epochs(tmp_path, 3, run_id="run_val")
    for epoch in (2, 3):
        (written / f"run_val_epoch{epoch:03}.pth").write_bytes(b"not a whole checkpoint")

    run = read_training_run(written)

    assert preview_checkpoint(run, None) == (None, "epoch 3, run's best: its weights are incomplete")
    assert preview_checkpoint(run, 1) == (None, "epoch 1, marked: its weights are not on disk")
    assert preview_checkpoint(None, None) == (None, "no epoch trained")


def test_a_whole_checkpoint_is_returned_with_what_it_is(tmp_path: Path):
    import zipfile

    written = _write_epochs(tmp_path, 2, run_id="run_val")
    with zipfile.ZipFile(written / "run_val_epoch001.pth", "w") as archive:
        archive.writestr("weights/data.pkl", b"weights")

    checkpoint, which = preview_checkpoint(read_training_run(written), 1)

    assert checkpoint is not None and checkpoint.epoch == 1
    assert which == "epoch 1, marked"


def _moved(tmp_path: Path) -> tuple[Path, Path]:
    """Return a prepared run and where its collection is now, the one it names being gone."""
    run = _prepared(tmp_path, _described_export(tmp_path))
    was = read_run_record(run).source
    here = tmp_path / "moved" / "collection"
    here.parent.mkdir()
    shutil.move(was, here)
    return run, here


def test_a_run_whose_collection_is_gone_is_read_as_under_the_folder_given(tmp_path: Path):
    run, here = _moved(tmp_path)

    read = read_preview_run("round01", read_run_record(run), here)

    assert [image.image.is_file() for image in read.validation] == [True]
    assert [image.label.is_file() for image in read.validation] == [True]
    assert all(path.is_relative_to(here) for path in read.training)
    assert read.validation[0].describe() == "Hydra"


def test_a_run_whose_collection_is_still_there_is_read_as_written(tmp_path: Path):
    run = _prepared(tmp_path, _described_export(tmp_path))
    record = read_run_record(run)
    scanned = tmp_path / "another_collection"
    scanned.mkdir()

    assert read_preview_run("round01", record, scanned) == read_preview_run("round01", record)
    assert read_preview_run("round01", record).validation[0].image.is_relative_to(record.source)


def test_with_no_folder_given_a_run_s_lists_are_read_as_written(tmp_path: Path):
    run, _ = _moved(tmp_path)
    record = read_run_record(run)

    assert read_preview_run("round01", record).validation[0].image.is_relative_to(record.source)


def test_an_image_trained_on_another_machine_is_still_not_offered(tmp_path: Path):
    """Two runs name one image by two paths, and it is one image once both are read as here."""
    run, here = _moved(tmp_path)
    record = read_run_record(run)
    ours = read_preview_run("ours", record, here)
    seen = next(iter(ours.training))
    theirs = PreviewRun("theirs", (PreviewImage(seen, seen),), frozenset())

    assert preview_images([ours, theirs]) == list(ours.validation)
    as_written = read_preview_run("ours", record)
    assert seen in [image.image for image in preview_images([as_written, theirs])]


def test_images_are_told_apart_by_whether_their_file_is_there():
    found, not_found = split_found(EVERY, exists=lambda path: path.stem != "b")

    assert (_names(found), _names(not_found)) == (["a", "c"], ["b"])


def test_images_not_found_are_said_once_with_the_folder_they_were_looked_for_in():
    """One line whatever the number of images, naming a folder and never a file."""
    runs = [
        PreviewRun("first", (ARCTIS_WAFFLE,), looked_in=Path("/ceph/labelled")),
        PreviewRun("second", (ARCTIS_WAFFLE,), looked_in=Path("/ceph/labelled")),
        PreviewRun("foreign", looked_in=Path("/never/read")),
    ]

    none = describe_not_found(EVERY, 3, runs, "Scan it.")
    some = describe_not_found(EVERY[:1], 3, runs, "Scan it.")

    assert none == (
        f"None of the 3 image(s) drawn is on this machine. They were looked for under "
        f"{Path('/ceph/labelled')}. Scan it."
    )
    assert some.startswith("1 of the 3 image(s) drawn are not on this machine and are left out.")
    assert ".tif" not in none + some and "Scan it." not in some
    assert describe_not_found([], 3, runs, "Scan it.") == ""


def test_a_run_says_which_folder_its_images_are_read_as_under(tmp_path: Path):
    run, here = _moved(tmp_path)
    record = read_run_record(run)

    assert read_preview_run("round01", record, here).looked_in == here
    assert read_preview_run("round01", record).looked_in == record.source
