"""Tests for CSV training data export functionality."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from am_segmentation_manager.cli import main
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.csv import export_training_csv, find_matching_label
from am_segmentation_manager.operations.training_split import Grouping
from am_segmentation_manager.state import read_state, write_state
from conftest import write_label


def _declare_corrected(stage: Path, names: list[str]) -> None:
    """Declare a stage's groups corrected, as the window would."""
    write_state(read_state(stage), {f"labels/{name}": "corrected" for name in names})


def test_find_matching_label(tmp_path: Path):
    """A group has one label in one place, so there is nothing to choose."""
    stage = tmp_path / "stage"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    img = sem / "img1.tif"
    label = labels / "img1.tif"
    label.write_text("lbl")

    record = ImageRecord(image_path=img, labels=[label], state="corrected")
    assert find_matching_label(record) == label

    absent = ImageRecord(image_path=img, labels=[labels / "gone.tif"])
    assert find_matching_label(absent) is None


def test_export_training_csv_full_split(tmp_path: Path):
    root = tmp_path / "dataset"
    stage = root / "01-lamella" / "Adaptive_polish_1"
    sem = stage / "sem" / "corrected"
    labels = stage / "labels" / "corrected"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    records = []
    for i in range(10):
        img = sem / f"AP_img_{i:03d}_SEM.tif"
        lbl = labels / f"AP_img_{i:03d}_SEM.tif"
        img.write_text("img")
        write_label(lbl, "annotated")
        records.append(
            ImageRecord(
                image_path=img,
                labels=[lbl],
                state="corrected",
                modality="SEM",
            )
        )

    out_dir = tmp_path / "output"
    train_path, val_path = export_training_csv(
        records,
        start_path=root,
        output_dir=out_dir,
        prefix="test_dataset",
        train_ratio=0.8,
        seed=42,
    )

    assert train_path.exists()
    assert val_path is not None and val_path.exists()

    # Verify CSV format and row count (8 train, 2 val)
    with train_path.open("r", encoding="utf-8") as fh:
        reader = list(csv.reader(fh))
        assert reader[0] == ["image", "label"]
        assert len(reader) == 9  # 1 header + 8 rows
        # Check relative paths format
        assert reader[1][0].startswith("01-lamella/Adaptive_polish_1/sem/")

    with val_path.open("r", encoding="utf-8") as fh:
        reader = list(csv.reader(fh))
        assert reader[0] == ["image", "label"]
        assert len(reader) == 3  # 1 header + 2 rows


def test_export_training_csv_reproducibility(tmp_path: Path):
    root = tmp_path / "dataset"
    stage = root / "01-lamella" / "Adaptive_polish_1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    records = []
    for i in range(20):
        img = sem / f"AP_img_{i:03d}_SEM.tif"
        lbl = labels / f"AP_img_{i:03d}_SEM.tif"
        img.write_text("img")
        write_label(lbl, "annotated")
        records.append(
            ImageRecord(
                image_path=img,
                labels=[lbl],
                state="selected",
                modality="SEM",
            )
        )

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    train1, val1 = export_training_csv(records, root, out1, seed=123)
    train2, val2 = export_training_csv(records, root, out2, seed=123)

    assert train1.read_text() == train2.read_text()
    assert val1.read_text() == val2.read_text()


def test_cli_create_training_csv_integration(tmp_path: Path):
    project = tmp_path / "projX"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: projX")

    stage = project / "L1" / "stage1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)

    for i in range(5):
        (sem / f"img{i}.tif").write_text("img")
        write_label(labels / f"img{i}.tif", "annotated")
    _declare_corrected(stage, [f"img{i}.tif" for i in range(5)])

    out_dir = tmp_path / "csv_output"

    rc = main([
        "create-training-csv",
        "--start-path", str(tmp_path),
        "--project", "projX",
        "--lamella", "all",
        "--stage", "all",
        "--image", "all",
        "--state", "corrected",
        # This project has SEM only, so a paired export would leave every FIB
        # cell blank; the shape under test here is the single-beam one.
        "--modality", "SEM",
        "-o", str(out_dir),
        "--output-prefix", "test_run",
    ])

    assert rc == 0
    train_csv = out_dir / "test_run_train.csv"
    val_csv = out_dir / "test_run_val.csv"
    assert train_csv.exists()
    assert val_csv.exists()

    content_train = train_csv.read_text()
    content_val = val_csv.read_text()
    assert "image,label" in content_train
    assert "image,label" in content_val
    combined = content_train + content_val
    assert "projX/L1/stage1/sem/img0.tif" in combined

    with (out_dir / "test_run_images.csv").open(newline="", encoding="utf-8") as handle:
        listed = list(csv.DictReader(handle))
    assert sorted(row["image"] for row in listed) == [
        f"projX/L1/stage1/sem/img{i}.tif" for i in range(5)
    ]
    assert {row["split"] for row in listed} == {"train"}, "one stage is one lamella, kept together"
    assert {"autolamella_project", "lamella", "modality"} <= set(listed[0])
    assert _manifest(out_dir, "test_run")["files"]["images"] == "test_run_images.csv"


def test_cli_create_training_csv_positional_scan_dir(tmp_path: Path):
    scan_dir = tmp_path / "20260728_CB_AM_bitmap_overnight"
    project = scan_dir / "AutoLamella-2026-07-28-10-31"
    stage = project / "01-ruling-ibex" / "Adaptive_polish_1"
    sem = stage / "sem"
    labels = stage / "labels"
    sem.mkdir(parents=True)
    labels.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: AutoLamella-2026-07-28-10-31")

    name = "01-ruling-ibex_AP_img_013_SEM.tif"
    (sem / name).write_text("img")
    write_label(labels / name, "annotated")
    _declare_corrected(stage, [name])

    out_dir = tmp_path / "csv_out"

    rc = main([
        "create-training-csv",
        str(scan_dir),
        "--project", "all",
        "--lamella", "all",
        "--stage", "all",
        "--image", "all",
        "--state", "corrected",
        "-o", str(out_dir),
        "--split", "1.0",
    ])

    assert rc == 0
    train_csv = out_dir / "dataset_train.csv"
    assert train_csv.exists()
    content = train_csv.read_text()
    # Paths MUST start with AutoLamella-2026-07-28-10-31 relative to the scan directory 20260728_CB_AM_bitmap_overnight
    assert "AutoLamella-2026-07-28-10-31/01-ruling-ibex/Adaptive_polish_1/sem/01-ruling-ibex_AP_img_013_SEM.tif" in content
    assert "AutoLamella-2026-07-28-10-31/01-ruling-ibex/Adaptive_polish_1/labels/01-ruling-ibex_AP_img_013_SEM.tif" in content


def test_cli_create_training_csv_requires_explicit_state(tmp_path: Path):
    project = tmp_path / "projX"
    project.mkdir()
    (project / "experiment.yaml").write_text("name: projX")

    # Call CLI without --state flag -> should fail (argparse SystemExit)
    with pytest.raises(SystemExit):
        main([
            "create-training-csv",
            "--start-path", str(tmp_path),
            "--project", "projX",
            "--lamella", "all",
            "--stage", "all",
            "--image", "all",
        ])


# ---------------------------------------------------------------------------
# Replacing an earlier export, and the manifest
# ---------------------------------------------------------------------------

def _labelled_records(root: Path, count: int = 10) -> list[ImageRecord]:
    """Return *count* SEM images of one stage, each with a label of its own."""
    stage = root / "01-lamella" / "Adaptive_polish_1"
    (stage / "sem").mkdir(parents=True)
    (stage / "labels").mkdir(parents=True)
    records = []
    for i in range(count):
        image = stage / "sem" / f"AP_img_{i:03d}_SEM.tif"
        label = stage / "labels" / f"AP_img_{i:03d}_SEM.tif"
        image.write_text("img")
        write_label(label, "annotated")
        records.append(
            ImageRecord(image_path=image, labels=[label], state="corrected", modality="SEM")
        )
    return records


def _manifest(out_dir: Path, prefix: str = "dataset") -> dict:
    import yaml

    return yaml.safe_load((out_dir / f"{prefix}_manifest.yaml").read_text())


def test_an_existing_validation_file_stops_the_export_before_anything_is_written(
    tmp_path: Path,
):
    """A split replaced by accident is gone with the seed that made it, so the
    refusal comes before a label is opened, not after the train file."""
    root = tmp_path / "dataset"
    records = _labelled_records(root)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "dataset_val.csv").write_text("earlier")
    opened = []

    with pytest.raises(FileExistsError, match="dataset_val.csv"):
        export_training_csv(
            records, root, out_dir, is_empty=lambda path: opened.append(path) or False
        )

    assert opened == []
    assert (out_dir / "dataset_val.csv").read_text() == "earlier"
    assert sorted(path.name for path in out_dir.iterdir()) == ["dataset_val.csv"]


def test_overwriting_replaces_an_earlier_training_set(tmp_path: Path):
    root = tmp_path / "dataset"
    records = _labelled_records(root)
    out_dir = tmp_path / "output"
    export_training_csv(records, root, out_dir, seed=1)

    export_training_csv(records, root, out_dir, seed=2, overwrite=True)

    assert _manifest(out_dir)["split"]["seed"] == 2


def test_the_manifest_counts_what_the_files_hold(tmp_path: Path):
    root = tmp_path / "dataset"
    records = _labelled_records(root)
    out_dir = tmp_path / "output"

    train, val = export_training_csv(
        records, root, out_dir, prefix="run", train_ratio=0.8, seed=42,
        scope="modality SEM", user="Ada Lovelace",
    )

    manifest = _manifest(out_dir, "run")
    rows = {
        "train": len(train.read_text().splitlines()) - 1,
        "validation": len(val.read_text().splitlines()) - 1,
    }
    assert manifest["files"] == {
        "train": "run_train.csv", "validation": "run_val.csv", "images": None,
    }
    assert manifest["counts"]["train"] == rows["train"] == 8
    assert manifest["counts"]["validation"] == rows["validation"] == 2
    assert manifest["counts"]["rows"] == manifest["counts"]["images"] == 10
    assert manifest["counts"]["images_left_out"] == 0
    assert manifest["split"] == {"train_ratio": 0.8, "seed": 42, "keep_together": "acquisition"}
    assert manifest["columns"] == ["image", "label"]
    assert manifest["scope"] == "modality SEM"
    assert manifest["written_by"] == "Ada Lovelace"
    assert manifest["collection"] == "../dataset"
    assert "source" not in manifest


def test_an_unseeded_split_records_no_seed(tmp_path: Path):
    root = tmp_path / "dataset"
    out_dir = tmp_path / "output"

    export_training_csv(_labelled_records(root), root, out_dir, seed=None)

    assert _manifest(out_dir)["split"]["seed"] is None


def test_a_full_training_share_is_not_stopped_by_a_validation_file_it_would_not_write(
    tmp_path: Path,
):
    root = tmp_path / "dataset"
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "dataset_val.csv").write_text("earlier")

    _, val = export_training_csv(_labelled_records(root), root, out_dir, train_ratio=1.0)

    assert val is None
    assert _manifest(out_dir)["files"]["validation"] is None
    assert (out_dir / "dataset_val.csv").read_text() == "earlier"


def test_cli_refuses_to_replace_a_training_set_without_overwrite(tmp_path: Path):
    project = tmp_path / "projX"
    stage = project / "L1" / "stage1"
    (stage / "sem").mkdir(parents=True)
    (stage / "labels").mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: projX")
    for i in range(5):
        (stage / "sem" / f"img{i}.tif").write_text("img")
        write_label(stage / "labels" / f"img{i}.tif", "annotated")
    _declare_corrected(stage, [f"img{i}.tif" for i in range(5)])
    out_dir = tmp_path / "csv_output"
    command = [
        "create-training-csv", "--start-path", str(tmp_path), "--project", "projX",
        "--lamella", "all", "--stage", "all", "--image", "all",
        "--state", "corrected", "--modality", "SEM", "-o", str(out_dir),
    ]

    assert main(command) == 0
    first = (out_dir / "dataset_train.csv").read_text()

    assert main([*command, "--seed", "7"]) == 2
    assert (out_dir / "dataset_train.csv").read_text() == first

    assert main([*command, "--seed", "7", "--overwrite"]) == 0
    manifest = _manifest(out_dir)
    assert manifest["split"]["seed"] == 7
    assert "--state corrected" in manifest["scope"]


# -- the images file ----------------------------------------------------------


def _describe(root: Path):
    from am_segmentation_manager.operations.metadata_csv import describe_training_images

    return describe_training_images(root)


def _listed(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_every_image_is_listed_once_with_the_side_it_is_on(tmp_path: Path):
    from am_segmentation_manager.operations.training_manifest import read_training_manifest

    root = tmp_path / "dataset"
    out_dir = tmp_path / "output"

    train, val = export_training_csv(
        _labelled_records(root), root, out_dir, train_ratio=0.8, describe_images=_describe(root)
    )

    manifest = read_training_manifest(out_dir / "dataset_manifest.yaml")
    listed = _listed(manifest.images)
    sides = {row["image"]: row["split"] for row in listed}
    assert len(listed) == len(sides) == 10
    assert {row["image"] for row in _listed(train)} == {i for i, side in sides.items() if side == "train"}
    assert {row["image"] for row in _listed(val)} == {i for i, side in sides.items() if side == "validation"}
    assert all(row["label"] and row["modality"] == "SEM" for row in listed)
    assert str(tmp_path) not in manifest.images.read_text(encoding="utf-8")


def test_an_export_asked_for_no_images_file_writes_none(tmp_path: Path):
    from am_segmentation_manager.operations.training_manifest import read_training_manifest

    root = tmp_path / "dataset"
    export_training_csv(_labelled_records(root), root, tmp_path / "output")

    assert not (tmp_path / "output" / "dataset_images.csv").exists()
    assert read_training_manifest(tmp_path / "output" / "dataset_manifest.yaml").images is None


def test_an_images_file_in_the_way_stops_the_export_before_anything_is_written(tmp_path: Path):
    root = tmp_path / "dataset"
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "dataset_images.csv").write_text("mine")

    with pytest.raises(FileExistsError, match="dataset_images.csv"):
        export_training_csv(_labelled_records(root), root, out_dir, describe_images=_describe(root))

    assert [path.name for path in out_dir.iterdir()] == ["dataset_images.csv"]


def test_images_that_cannot_be_described_stop_the_export_before_anything_is_written(tmp_path: Path):
    root = tmp_path / "dataset"
    out_dir = tmp_path / "output"

    def describe(records):
        raise ValueError("a project's metadata will not read")

    with pytest.raises(ValueError, match="will not read"):
        export_training_csv(_labelled_records(root), root, out_dir, describe_images=describe)

    assert list(out_dir.iterdir()) == []


def test_in_folds_an_image_s_side_is_the_fold_it_validates_in(tmp_path: Path):
    from am_segmentation_manager.operations.csv import export_kfold_csv

    root = tmp_path / "dataset"
    out_dir = tmp_path / "output"

    paths = export_kfold_csv(
        _labelled_records(root), root, out_dir, folds=2, describe_images=_describe(root)
    )

    sides = {row["image"]: row["split"] for row in _listed(out_dir / "dataset_images.csv")}
    assert len(sides) == 10
    for number, (_, validation) in enumerate(paths, start=1):
        assert {row["image"] for row in _listed(validation)} == {
            image for image, side in sides.items() if side == f"fold{number}"
        }
    assert _manifest(out_dir)["files"]["images"] == "dataset_images.csv"


#: Keeps a lamella whole, read off where :func:`_mixed_records` puts an image.
BY_LAMELLA = Grouping("lamella", lambda record: record.image_path.parents[2].name)


def _mixed_records(root: Path, lamellae: int = 10) -> list[ImageRecord]:
    """Return four SEM images of each lamella: two corrected, two selected, all annotated."""
    records = []
    for number in range(lamellae):
        stage = root / f"{number:02d}-lamella" / "Adaptive_polish_1"
        (stage / "sem").mkdir(parents=True)
        (stage / "labels").mkdir(parents=True)
        for i in range(4):
            image = stage / "sem" / f"AP_img_{i:03d}_SEM.tif"
            label = stage / "labels" / f"AP_img_{i:03d}_SEM.tif"
            image.write_text("img")
            write_label(label, "annotated")
            records.append(ImageRecord(
                image_path=image, labels=[label], modality="SEM",
                state="corrected" if i < 2 else "selected",
            ))
    return records


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _lamella(cell: str) -> str:
    return cell.split("/")[0]


def test_adding_selected_labels_leaves_the_validation_list_as_it_was(tmp_path: Path):
    records = _mixed_records(tmp_path / "data")
    corrected = [record for record in records if record.state == "corrected"]

    export_training_csv(
        corrected, tmp_path / "data", tmp_path / "proofread", grouping=BY_LAMELLA
    )
    export_training_csv(
        records, tmp_path / "data", tmp_path / "mixed",
        grouping=BY_LAMELLA, validate_on="corrected",
    )

    validation = _rows(tmp_path / "mixed" / "dataset_val.csv")
    assert validation == _rows(tmp_path / "proofread" / "dataset_val.csv")
    assert validation


def test_only_corrected_labels_are_validated_on_and_no_lamella_is_on_both_sides(
    tmp_path: Path,
):
    records = _mixed_records(tmp_path / "data")
    state_of = {
        record.image_path.relative_to(tmp_path / "data").as_posix(): record.state
        for record in records
    }

    export_training_csv(
        records, tmp_path / "data", tmp_path / "out",
        grouping=BY_LAMELLA, validate_on="corrected",
    )

    training = _rows(tmp_path / "out" / "dataset_train.csv")
    validation = _rows(tmp_path / "out" / "dataset_val.csv")
    assert {state_of[row["image"]] for row in validation} == {"corrected"}
    assert {state_of[row["image"]] for row in training} == {"corrected", "selected"}
    validated = {_lamella(row["image"]) for row in validation}
    assert not validated & {_lamella(row["image"]) for row in training}

    manifest = _manifest(tmp_path / "out")
    assert manifest["split"]["validate_on"] == "corrected"
    counts = manifest["counts"]
    assert counts["left_out_beside_validation"] == 2 * len(validated)
    assert counts["train_corrected"] + counts["train_not_corrected"] == len(training)
    assert counts["train"] + counts["validation"] + counts["left_out_beside_validation"] == 40


def test_nothing_corrected_to_validate_on_stops_the_export_before_anything_is_written(
    tmp_path: Path,
):
    records = [
        record for record in _mixed_records(tmp_path / "data") if record.state == "selected"
    ]

    with pytest.raises(ValueError, match="nothing to validate on"):
        export_training_csv(
            records, tmp_path / "data", tmp_path / "out",
            grouping=BY_LAMELLA, validate_on="corrected",
        )

    assert not (tmp_path / "out").exists()


def test_an_export_validating_on_any_state_says_nothing_of_it(tmp_path: Path):
    export_training_csv(_labelled_records(tmp_path / "data"), tmp_path / "data", tmp_path / "out")

    manifest = _manifest(tmp_path / "out")
    assert "validate_on" not in manifest["split"]
    assert "left_out_beside_validation" not in manifest["counts"]


def _mixed_project(root: Path, lamellae: int = 10) -> None:
    """Write a project of *lamellae*, each of two corrected and two selected SEM images."""
    project = root / "projX"
    project.mkdir(parents=True)
    (project / "experiment.yaml").write_text("name: projX")
    for number in range(lamellae):
        stage = project / f"L{number:02d}" / "stage1"
        (stage / "sem").mkdir(parents=True)
        (stage / "labels").mkdir(parents=True)
        for i in range(4):
            (stage / "sem" / f"img{i}.tif").write_text("img")
            write_label(stage / "labels" / f"img{i}.tif", "annotated")
        _declare_corrected(stage, ["img0.tif", "img1.tif"])


def _training_arguments(root: Path, out_dir: Path) -> list[str]:
    return [
        "create-training-csv", "--start-path", str(root), "--project", "all",
        "--lamella", "all", "--stage", "all", "--image", "all", "--modality", "SEM",
        "-o", str(out_dir),
    ]


def test_cli_trains_on_two_states_and_validates_on_one(tmp_path: Path):
    _mixed_project(tmp_path / "data")
    out_dir = tmp_path / "out"

    rc = main([
        *_training_arguments(tmp_path / "data", out_dir),
        "--state", "corrected", "selected", "--validate-on", "corrected",
    ])

    assert rc == 0
    listed = _rows(out_dir / "dataset_images.csv")
    assert {row["state"] for row in listed if row["split"] == "validation"} == {"corrected"}
    assert {row["state"] for row in listed if row["split"] == "train"} == {"corrected", "selected"}
    manifest = _manifest(out_dir)
    assert manifest["split"]["validate_on"] == "corrected"
    assert "--state corrected selected" in manifest["scope"]
    assert "--validate-on corrected" in manifest["scope"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--state", "selected", "--validate-on", "corrected"],
        ["--state", "corrected", "selected", "--validate-on", "corrected", "--folds", "2"],
    ],
)
def test_cli_refuses_a_validate_on_it_could_not_keep(tmp_path: Path, extra: list[str]):
    _mixed_project(tmp_path / "data", lamellae=3)
    out_dir = tmp_path / "out"

    assert main([*_training_arguments(tmp_path / "data", out_dir), *extra]) == 2
    assert not out_dir.exists()


def _add_lamellae(root: Path, first: int, count: int) -> list[ImageRecord]:
    """Return *count* more lamellae as :func:`_mixed_records` builds them, numbered from *first*."""
    records = []
    for number in range(first, first + count):
        stage = root / f"{number:02d}-lamella" / "Adaptive_polish_1"
        (stage / "sem").mkdir(parents=True)
        (stage / "labels").mkdir(parents=True)
        for i in range(4):
            image = stage / "sem" / f"AP_img_{i:03d}_SEM.tif"
            label = stage / "labels" / f"AP_img_{i:03d}_SEM.tif"
            image.write_text("img")
            write_label(label, "annotated")
            records.append(ImageRecord(
                image_path=image, labels=[label], modality="SEM",
                state="corrected" if i < 2 else "selected",
            ))
    return records


def test_a_later_export_keeps_every_lamella_on_the_side_it_was(tmp_path: Path):
    data = tmp_path / "data"
    records = _mixed_records(data)
    shared = {"grouping": BY_LAMELLA, "validate_on": "corrected"}
    export_training_csv(records, data, tmp_path / "first", **shared)
    was_training = {_lamella(row["image"]) for row in _rows(tmp_path / "first" / "dataset_train.csv")}
    was_validation = {_lamella(row["image"]) for row in _rows(tmp_path / "first" / "dataset_val.csv")}

    # A round of proofreading, and ten lamellae nobody had then.
    for record in records:
        record.state = "corrected"
    records += _add_lamellae(data, first=10, count=10)
    export_training_csv(
        records, data, tmp_path / "second", seed=7,
        validation_from=tmp_path / "first" / "dataset_manifest.yaml", **shared,
    )

    training = _rows(tmp_path / "second" / "dataset_train.csv")
    validation = _rows(tmp_path / "second" / "dataset_val.csv")
    now_training = {_lamella(row["image"]) for row in training}
    now_validation = {_lamella(row["image"]) for row in validation}
    assert was_training <= now_training and was_validation <= now_validation
    assert not now_training & now_validation
    proofread_since = [row for row in validation if _lamella(row["image"]) in was_validation]
    assert len(proofread_since) == 4 * len(was_validation), "all four images of each, now"
    assert now_validation - was_validation, "the new lamellae fill both sides"
    assert _manifest(tmp_path / "second")["split"]["validation_from"] == (
        "../first/dataset_manifest.yaml"
    )


def test_an_earlier_export_of_other_images_stops_the_export_before_anything_is_written(
    tmp_path: Path,
):
    export_training_csv(
        _mixed_records(tmp_path / "other"), tmp_path / "other", tmp_path / "first",
        grouping=BY_LAMELLA,
    )

    with pytest.raises(ValueError, match="names none of these images"):
        export_training_csv(
            _add_lamellae(tmp_path / "data", first=50, count=3), tmp_path / "data",
            tmp_path / "out", grouping=BY_LAMELLA,
            validation_from=tmp_path / "first" / "dataset_manifest.yaml",
        )

    assert not (tmp_path / "out").exists()


def test_cli_keeps_to_the_split_of_an_earlier_export(tmp_path: Path):
    _mixed_project(tmp_path / "data")
    arguments = ["--state", "corrected", "selected", "--validate-on", "corrected"]
    assert main([*_training_arguments(tmp_path / "data", tmp_path / "first"), *arguments]) == 0

    rc = main([
        *_training_arguments(tmp_path / "data", tmp_path / "second"), *arguments,
        "--seed", "7", "--validation-from", str(tmp_path / "first" / "dataset_manifest.yaml"),
    ])

    assert rc == 0
    for name in ("dataset_train.csv", "dataset_val.csv"):
        assert sorted(map(str, _rows(tmp_path / "second" / name))) == sorted(
            map(str, _rows(tmp_path / "first" / name))
        )
    assert "--validation-from dataset_manifest.yaml" in _manifest(tmp_path / "second")["scope"]
