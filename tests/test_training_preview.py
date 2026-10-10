"""Choosing validation images to look at models on, and which checkpoint of a run."""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.operations.training_run import read_run_record
from am_segmentation_manager.training_preview import (
    PreviewImage,
    PreviewRun,
    offered_values,
    pick_images,
    preview_checkpoint,
    preview_images,
    read_preview_run,
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
