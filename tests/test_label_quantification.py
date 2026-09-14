"""Rolling label measurements up, exporting them, and the command.

The behaviour under test is mostly about *not counting the same thing twice*.
Several imaging conditions of one acquisition share one label, because they are
one segmentation; a crack in that label is one crack however many conditions
reference it, and however many frames of the milling sequence it appears in.

A lamella counts as cracked as soon as any one of its labels holds a crack.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from am_segmentation_manager.aggregate import (
    by_has_crack,
    count_by,
    project_lookup,
    summarise_lamellae,
)
from am_segmentation_manager.cli import main
from am_segmentation_manager.config import load_segmentation_config
from am_segmentation_manager.filtering import iter_matching_records
from am_segmentation_manager.operations.label_measure import (
    clear_cache,
    measure_records,
)
from am_segmentation_manager.operations.metadata_csv import (
    export_label_measurements_csv,
)
from am_segmentation_manager.scanner import Scanner

HAS_IMAGE_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "tifffile", "cv2")
)
pytestmark = pytest.mark.skipif(
    not HAS_IMAGE_STACK, reason="optional image libs not installed"
)

if HAS_IMAGE_STACK:
    import numpy as np
    import tifffile

CRACK_CLASS = 3
LAMELLA_CLASS = 2

#: 100 nm pixels over a 100 px label: one pixel is 0.01 um^2, so a 20x20 square
#: is 4 um^2 and comfortably over the 0.1 um^2 default threshold.
PIXEL_SIZE_M = 100e-9
SIDE = 100


@pytest.fixture(autouse=True)
def _clear_measurement_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def config():
    return load_segmentation_config()


def header(width: int = SIDE, height: int = SIDE) -> str:
    """A tag-270 document carrying what the scale is derived from."""
    return json.dumps(
        {
            "image": {"resolution": [width, height], "beam_type": "ELECTRON"},
            "pixel_size": {"x": PIXEL_SIZE_M, "y": PIXEL_SIZE_M},
            "microscope_state": {"electron_beam": {"voltage": 2000.0}},
        }
    )


def write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        np.full((SIDE, SIDE), 100, dtype=np.uint8),
        description=header(),
    )
    return path


def write_label(path: Path, *, cracks: int = 0, lamella: bool = True) -> Path:
    """Write a label holding *cracks* separated 20x20 squares of class 3."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((SIDE, SIDE), dtype=np.uint8)
    if lamella:
        array[:10] = LAMELLA_CLASS
    for index in range(cracks):
        row = 20 + index * 25
        array[row : row + 20, 20:40] = CRACK_CLASS
    tifffile.imwrite(str(path), array)
    return path


def build_project(
    root: Path,
    *,
    name: str = "project",
    lamellae: tuple[str, ...] = ("01-lamella",),
    cracks_per_lamella: tuple[int, ...] = (1,),
    variants: tuple[str, ...] = (),
    frames: int = 1,
) -> Path:
    """Build a project whose SEM conditions share one label per acquisition.

    Args:
        cracks_per_lamella: How many cracks the *first* frame of each lamella's
            label holds. Later frames hold none, which is what makes the
            per-lamella roll-up meaningful.
        variants: Extra SEM imaging conditions sharing the label directory.
        frames: Acquisitions per lamella.
    """
    project = root / name
    project.mkdir(parents=True, exist_ok=True)
    (project / "experiment.yaml").write_text(f"name: {name}\n", encoding="utf-8")

    for index, lamella in enumerate(lamellae):
        stage = project / lamella / "Adaptive_polish"
        cracks = cracks_per_lamella[index] if index < len(cracks_per_lamella) else 0
        for frame in range(frames):
            filename = f"img_{frame:03d}_electron.tif"
            write_image(stage / "sem" / filename)
            for variant in variants:
                write_image(stage / variant / filename)
            write_label(
                stage / "sem_labels" / filename,
                cracks=cracks if frame == 0 else 0,
            )
    return project


def records_of(root: Path, state: str = "all") -> list:
    index = Scanner(root).build_index()
    return list(iter_matching_records(index, modality="all", state=state))


# ---------------------------------------------------------------------------
# One label, measured once
# ---------------------------------------------------------------------------

def test_conditions_sharing_a_label_share_one_measurement(tmp_path: Path, config):
    """Five conditions of one scene are one segmentation, not five."""
    build_project(tmp_path, variants=("sem_fi1", "sem_fi2"))
    records = records_of(tmp_path)

    measurements = measure_records(records, config)

    assert len(records) == 3  # three conditions
    assert len(measurements) == 3  # every image is answered for
    distinct = {id(measurement) for measurement in measurements.values()}
    assert len(distinct) == 1  # by the same measurement object


def test_a_record_without_a_label_is_absent(tmp_path: Path, config):
    """Distinct from an unreadable label, which is present with no areas."""
    project = build_project(tmp_path)
    for label in (project / "01-lamella" / "Adaptive_polish" / "sem_labels").iterdir():
        label.unlink()

    measurements = measure_records(records_of(tmp_path), config)

    assert measurements == {}


def test_scale_comes_from_the_image_header(tmp_path: Path, config):
    build_project(tmp_path)

    measurements = measure_records(records_of(tmp_path), config)
    measurement = next(iter(measurements.values()))

    assert measurement.pixel_size_nm == pytest.approx(100.0)
    assert measurement.areas_um2[CRACK_CLASS] == pytest.approx(4.0)
    assert measurement.crack_count == 1


# ---------------------------------------------------------------------------
# The per-lamella roll-up
# ---------------------------------------------------------------------------

def test_one_crack_in_one_frame_makes_the_lamella_cracked(tmp_path: Path, config):
    """Three frames, a crack in the first: the lamella has a crack."""
    build_project(tmp_path, frames=3, cracks_per_lamella=(1,))
    records = records_of(tmp_path)

    summaries = summarise_lamellae(records, measure_records(records, config))

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.has_crack
    assert summary.measured == 3  # three labels
    assert summary.cracked == 1  # one of them holds the crack
    assert summary.cracks == 1


def test_a_shared_label_is_counted_once(tmp_path: Path, config):
    """The same crack seen under three conditions is still one crack."""
    build_project(tmp_path, variants=("sem_fi1", "sem_fi2"), cracks_per_lamella=(2,))
    records = records_of(tmp_path)

    summary = summarise_lamellae(records, measure_records(records, config))[0]

    assert len(records) == 3
    assert summary.labels == 1
    assert summary.cracks == 2  # not 6
    assert summary.crack_area_um2 == pytest.approx(8.0)


def test_lamellae_are_summarised_separately(tmp_path: Path, config):
    build_project(
        tmp_path,
        lamellae=("01-lamella", "02-lamella"),
        cracks_per_lamella=(1, 0),
    )
    records = records_of(tmp_path)

    summaries = summarise_lamellae(records, measure_records(records, config))

    assert [summary.lamella for summary in summaries] == ["01-lamella", "02-lamella"]
    assert [summary.has_crack for summary in summaries] == [True, False]


def test_empty_labels_are_excluded_from_the_measured_total(tmp_path: Path, config):
    """A placeholder the segmentation software wrote is not a proofread label."""
    project = build_project(tmp_path, frames=2, cracks_per_lamella=(1,))
    labels = sorted((project / "01-lamella" / "Adaptive_polish" / "sem_labels").iterdir())
    write_label(labels[1], cracks=0, lamella=False)  # blank the second

    records = records_of(tmp_path)
    summary = summarise_lamellae(records, measure_records(records, config))[0]

    assert summary.labels == 2
    assert summary.empty == 1
    assert summary.measured == 1  # the empty one is not a measurement
    assert summary.cracked_share == pytest.approx(1.0)


def test_cracked_share_is_none_when_nothing_could_be_measured(tmp_path: Path, config):
    """No data and no cracks are different answers."""
    project = build_project(tmp_path)
    for label in (project / "01-lamella" / "Adaptive_polish" / "sem_labels").iterdir():
        write_label(label, cracks=0, lamella=False)

    records = records_of(tmp_path)
    summary = summarise_lamellae(records, measure_records(records, config))[0]

    assert summary.measured == 0
    assert summary.cracked_share is None
    assert not summary.has_crack


def test_projects_sharing_a_name_stay_apart(tmp_path: Path, config):
    """Project names are not unique in the reference collection; paths are."""
    build_project(tmp_path / "a", name="project", cracks_per_lamella=(1,))
    build_project(tmp_path / "b", name="project", cracks_per_lamella=(0,))

    records = records_of(tmp_path)
    lookup = project_lookup(Scanner(tmp_path).build_index().all_projects())
    summaries = summarise_lamellae(records, measure_records(records, config), lookup)

    assert len(summaries) == 2
    assert len({summary.project for summary in summaries}) == 2


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------

def test_breakdown_by_crack(tmp_path: Path, config):
    build_project(
        tmp_path,
        lamellae=("01-lamella", "02-lamella"),
        cracks_per_lamella=(1, 0),
    )
    records = records_of(tmp_path)

    counts = count_by(records, by_has_crack(measure_records(records, config)))

    assert counts == {"cracked": 1, "intact": 1}


def test_a_label_with_no_scale_is_not_counted_as_intact(tmp_path: Path, config):
    """Its own group, so an unread label never understates the cracking."""
    project = build_project(tmp_path)
    # Rewrite the image with no header, so no pixel size can be derived.
    for image in (project / "01-lamella" / "Adaptive_polish" / "sem").iterdir():
        tifffile.imwrite(str(image), np.zeros((SIDE, SIDE), dtype=np.uint8))

    records = records_of(tmp_path)
    counts = count_by(records, by_has_crack(measure_records(records, config)))

    assert counts == {"no-pixel-size": 1}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_measurement_csv_has_a_row_per_image_and_named_area_columns(
    tmp_path: Path, config
):
    build_project(tmp_path, variants=("sem_fi1",))
    records = records_of(tmp_path)
    output = tmp_path / "out" / "measurements.csv"

    export_label_measurements_csv(
        records, measure_records(records, config), config, tmp_path, output
    )
    rows = read_csv(output)

    assert len(rows) == 2  # one per condition, both pointing at one label
    assert len({row["label"] for row in rows}) == 1
    assert rows[0]["area_um2_crack_hole"] == "4"
    assert rows[0]["area_um2_lamella"] == "10"
    assert rows[0]["crack_count"] == "1"
    assert rows[0]["label_status"] == "measured"
    assert rows[0]["label_pixel_size_nm"] == "100"


def test_measurement_csv_omits_images_with_no_label(tmp_path: Path, config):
    """It is a file of measurements; 'which images lack labels' has its own export."""
    project = build_project(tmp_path, frames=2)
    labels = sorted((project / "01-lamella" / "Adaptive_polish" / "sem_labels").iterdir())
    labels[1].unlink()

    records = records_of(tmp_path)
    output = tmp_path / "measurements.csv"
    export_label_measurements_csv(
        records, measure_records(records, config), config, tmp_path, output
    )

    assert len(records) == 2
    assert len(read_csv(output)) == 1


def test_measurement_csv_records_the_size_distribution(tmp_path: Path, config):
    """Each qualifying component, so a script can plot the distribution."""
    build_project(tmp_path, cracks_per_lamella=(3,))
    records = records_of(tmp_path)
    output = tmp_path / "measurements.csv"

    export_label_measurements_csv(
        records, measure_records(records, config), config, tmp_path, output
    )
    row = read_csv(output)[0]

    assert row["crack_areas_um2"].split() == ["4", "4", "4"]
    assert row["crack_area_um2_total"] == "12"
    assert row["crack_area_um2_largest"] == "4"
    assert row["min_crack_area_um2"] == "0.1"


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def quantify(root: Path, *extra: str) -> int:
    return main([
        "quantify-labels",
        "--start-path", str(root),
        "--project", "all", "--lamella", "all", "--stage", "all", "--image", "all",
        "--state", "all",
        *extra,
    ])


def test_command_reports_and_exports(tmp_path: Path, capsys):
    build_project(
        tmp_path,
        lamellae=("01-lamella", "02-lamella"),
        cracks_per_lamella=(1, 0),
    )
    per_image = tmp_path / "images.csv"
    per_lamella = tmp_path / "lamellae.csv"

    assert quantify(tmp_path, "-o", str(per_image), "--lamella-output", str(per_lamella)) == 0

    printed = capsys.readouterr().out
    assert "1 of 2 lamellae have a crack" in printed
    assert "01-lamella" in printed

    lamella_rows = read_csv(per_lamella)
    assert [row["has_crack"] for row in lamella_rows] == ["True", "False"]
    assert len(read_csv(per_image)) == 2


def test_command_defaults_to_corrected_labels(tmp_path: Path, capsys):
    """A prediction-derived label measures the model, not the sample.

    Both halves matter: the default must skip this uncorrected dataset, and
    --state all must then find it. Asserting only the first would pass just as
    well if the command measured nothing at all.
    """
    build_project(tmp_path)
    scope = [
        "--start-path", str(tmp_path),
        "--project", "all", "--lamella", "all", "--stage", "all", "--image", "all",
    ]

    assert main(["quantify-labels", *scope]) == 0
    assert "have a crack" not in capsys.readouterr().out

    assert main(["quantify-labels", *scope, "--state", "all"]) == 0
    assert "1 of 1 lamellae have a crack" in capsys.readouterr().out


def test_command_threshold_override(tmp_path: Path, capsys):
    build_project(tmp_path, cracks_per_lamella=(1,))

    # Each crack is 4 um^2, so a 10 um^2 floor discards it.
    assert quantify(tmp_path, "--min-crack-area", "10") == 0

    assert "0 of 1 lamellae have a crack" in capsys.readouterr().out


def test_the_command_refuses_to_replace_an_export_before_measuring(
    tmp_path: Path, monkeypatch
):
    """Measuring reads every label, so a refusal has to come before it, not after."""
    import am_segmentation_manager.cli.quantifying as quantifying

    build_project(tmp_path)
    per_image, per_lamella = tmp_path / "images.csv", tmp_path / "lamellae.csv"
    per_lamella.write_text("kept\n", encoding="utf-8")

    def measured(*args, **kwargs):
        raise AssertionError("labels were measured for an export that was refused")

    monkeypatch.setattr(quantifying, "measure_records", measured)

    assert quantify(tmp_path, "-o", str(per_image), "--lamella-output", str(per_lamella)) == 2
    assert per_lamella.read_text(encoding="utf-8") == "kept\n"
    assert not per_image.exists()
