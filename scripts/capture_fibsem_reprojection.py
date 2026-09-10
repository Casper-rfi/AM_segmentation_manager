#!/usr/bin/env python3
"""Capture fibsemOS's own reprojections, which ``tests/test_overview.py`` holds to.

**This is a standalone script, not an ``am-seg`` command**, and it runs in an
environment with fibsem 0.5.2rc1 or later installed -- fibsemOS's own -- which
neither of this package's environments is. Run it when the fibsemOS function
``overview.py`` restates changes, and commit what it writes.

It reads a collection without changing it and writes two fixtures into
``tests/data``: ``fibsem_reprojection.json``, where fibsemOS places every
recorded lamella on every Autolamella overview it can read, and
``fibsem_reprojection_branches.json``, geometry no overview in the collection
reaches -- the ion beam, the stage facing the ion column, a pre-tilt below 45
degrees, a compustage either way up, no scan rotation -- made by changing one
real header. Each case records the header values the projection reads, so the
suite needs no fibsem to compare against them.

Nothing existing is overwritten unless ``--overwrite`` says so.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import yaml
from fibsem.imaging import tiled
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition

REPOSITORY = Path(__file__).resolve().parent.parent
FUNCTION = "calculate_reprojected_stage_position2"
#: A position recorded further than this from the overview's rotation is one
#: ``overview.reproject`` refuses, so it is not captured.
ROTATION_TOLERANCE_DEG = 2.0


def geometry(image: FibsemImage) -> dict:
    """Return what the projection reads from *image*'s header."""
    metadata = image.metadata
    state = metadata.microscope_state
    beam = metadata.image_settings.beam_type
    scan = (state.electron_beam if beam is BeamType.ELECTRON else state.ion_beam).scan_rotation
    centre = state.stage_position
    hardware = metadata.hardware_geometry
    return {
        "width": int(image.data.shape[1]),
        "height": int(image.data.shape[0]),
        "pixel_size_m": metadata.pixel_size.x,
        "scan_rotation_rad": scan,
        "beam_type": beam.name,
        "centre": {axis: getattr(centre, axis) for axis in "xyzrt"},
        "system": {
            "model": metadata.system_info.model,
            "shuttle_pre_tilt_deg": hardware.shuttle_pre_tilt,
            "rotation_reference_deg": hardware.rotation_reference,
            "rotation_180_deg": hardware.rotation_180,
            "electron_column_tilt_deg": hardware.column_tilt,
            "ion_column_tilt_deg": hardware.fib_column_tilt,
        },
    }


def placed(image: FibsemImage, position: FibsemStagePosition) -> dict:
    """Return where fibsemOS places *position* on *image*."""
    point = getattr(tiled, FUNCTION)(image, position)
    return {
        "name": position.name,
        "position": {axis: getattr(position, axis) for axis in "xyzrt"},
        "fibsem_x": float(point.x),
        "fibsem_y": float(point.y),
    }


def lamella_name(entry: dict) -> str | None:
    """Return the directory a position's lamella was written into, as overview.py does."""
    petname = entry.get("petname")
    if isinstance(petname, str) and petname.strip():
        return petname.strip()
    path = entry.get("path")
    if isinstance(path, str) and path.strip():
        return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or None
    return None


def recorded_positions(experiment: Path) -> list[FibsemStagePosition]:
    """Return every position ``experiment.yaml`` records coordinates for."""
    document = yaml.safe_load(experiment.read_text(encoding="utf-8")) or {}
    found = []
    for entry in document.get("positions") or []:
        if not isinstance(entry, dict):
            continue
        stage = (((entry.get("state") or {}).get("microscope_state") or {}).get("stage_position")) or {}
        values = [stage.get(axis) for axis in "xyzrt"]
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            continue
        x, y, z, r, t = (float(value) for value in values)
        found.append(FibsemStagePosition(x=x, y=y, z=z, r=r, t=t, name=lamella_name(entry)))
    return found


def is_overview(path: Path) -> bool:
    stem = path.stem.lower()
    return "overview" in stem and not stem.endswith("-autogamma")


def real_cases(collection: Path) -> list[dict]:
    """Return every overview fibsemOS can read, with its lamellae placed on it."""
    cases = []
    for experiment in sorted(collection.rglob("experiment.yaml")):
        positions = recorded_positions(experiment)
        if not positions:
            continue
        for path in sorted(experiment.parent.glob("*.tif")):
            if not is_overview(path):
                continue
            try:
                image = FibsemImage.load(str(path))
                case = geometry(image)
            except Exception as exc:  # noqa: BLE001 - a header fibsemOS cannot read
                print(f"skipped {path.relative_to(collection)}: {exc}", file=sys.stderr)
                continue
            rotation = image.metadata.microscope_state.stage_position.r
            entries = [
                placed(image, position)
                for position in positions
                if position.name
                and abs(math.degrees(rotation - position.r)) <= ROTATION_TOLERANCE_DEG
            ]
            if entries:
                cases.append({"overview": path.relative_to(collection).as_posix(), **case,
                              "positions": entries})
    return cases


def around(image: FibsemImage) -> list[FibsemStagePosition]:
    """Return positions a little way from *image*'s centre, one along each axis."""
    centre = image.metadata.microscope_state.stage_position
    offsets = {
        "east": (200e-6, 0.0, 0.0),
        "north": (0.0, 150e-6, 0.0),
        "raised": (0.0, 0.0, 150e-6),
        "along the grid": (0.0, 100e-6, -100e-6),
    }
    return [
        FibsemStagePosition(
            x=centre.x + dx, y=centre.y + dy, z=centre.z + dz, r=centre.r, t=centre.t, name=name,
        )
        for name, (dx, dy, dz) in offsets.items()
    ]


def branch_cases(image: FibsemImage) -> list[dict]:
    """Return geometry no real overview reaches, each made by changing *image*'s header."""
    def changed(label: str, change) -> dict:
        variant = copy.deepcopy(image)
        change(variant.metadata)
        return {"label": label, **geometry(variant),
                "positions": [placed(variant, position) for position in around(variant)]}

    def no_scan_rotation(metadata):
        metadata.microscope_state.electron_beam.scan_rotation = 0.0

    def ion_beam(metadata):
        metadata.image_settings.beam_type = BeamType.ION
        metadata.microscope_state.ion_beam.scan_rotation = (
            metadata.microscope_state.electron_beam.scan_rotation
        )

    def facing_the_ion_column(metadata):
        rotation = math.degrees(metadata.microscope_state.stage_position.r)
        metadata.hardware_geometry.rotation_180 = rotation
        metadata.hardware_geometry.rotation_reference = rotation + 180

    def shallow_pretilt(metadata):
        metadata.hardware_geometry.shuttle_pre_tilt = 35.0

    def compustage(tilt_deg):
        def change(metadata):
            metadata.system_info.model = "Arctis"
            metadata.hardware_geometry.is_compustage = True
            metadata.hardware_geometry.shuttle_pre_tilt = 0.0
            metadata.microscope_state.stage_position.t = math.radians(tilt_deg)
        return change

    return [
        changed("the header as written", lambda metadata: None),
        changed("no scan rotation", no_scan_rotation),
        changed("an ion-beam overview", ion_beam),
        changed("the stage facing the ion column", facing_the_ion_column),
        changed("a pre-tilt below 45 degrees", shallow_pretilt),
        changed("a compustage at no tilt", compustage(0.0)),
        changed("a compustage upside down", compustage(-180.0)),
    ]


def write(path: Path, document: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite to replace it")
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("collection", type=Path, help="Folder holding the Autolamella projects")
    parser.add_argument(
        "--branch-overview", type=Path, required=True,
        help="An overview, relative to the collection, whose header the branch cases change",
    )
    parser.add_argument("--out", type=Path, default=REPOSITORY / "tests" / "data")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing fixtures")
    args = parser.parse_args()

    source = f"fibsem {version('fibsem')}, {FUNCTION}"
    cases = real_cases(args.collection)
    write(args.out / "fibsem_reprojection.json", {"source": source, "cases": cases}, args.overwrite)
    branches = branch_cases(FibsemImage.load(str(args.collection / args.branch_overview)))
    write(
        args.out / "fibsem_reprojection_branches.json",
        {
            "source": source,
            "note": "Synthetic geometry made by changing one real header, for the "
                    "branches no overview in the collection reaches.",
            "cases": branches,
        },
        args.overwrite,
    )
    print(f"{len(cases)} overviews, {sum(len(case['positions']) for case in cases)} positions; "
          f"{len(branches)} branch cases")


if __name__ == "__main__":
    main()
