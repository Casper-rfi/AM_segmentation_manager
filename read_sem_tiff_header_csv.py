#!/usr/bin/env python3
"""Read TIFF paths from a CSV file and write extracted metadata into a new CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

try:
    import tifffile
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("tifffile is required. Install it with: pip install tifffile") from exc


METADATA_COLUMNS = [
    "image_width_pixels",
    "image_height_pixels",
    "pixel_size_nm",
    "dwell_time_ns",
    "image_integrations",
    "line_integrations",
    "microscope_model",
    "acceleration_voltage",
    "beam_current_pa",
]
DEFAULT_BASE_DIR = Path("<collection>").expanduser()
DEFAULT_SESSION_METADATA_CSV = Path(__file__).with_name("sessions_metadata_manual.csv")


def _get_tag_value(page: Any, tag_code: int) -> Any:
    tag = page.tags.get(tag_code)
    return None if tag is None else tag.value


def _parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    if isinstance(value, dict):
        return value
    return {}


def _find_key_value(node: Any, *keys: str) -> Any:
    if not isinstance(node, dict):
        return None

    normalized_keys = {key.lower().replace("-", "").replace("_", ""): key for key in keys}
    for key, value in node.items():
        normalized_key = key.lower().replace("-", "").replace("_", "")
        if normalized_key in normalized_keys:
            return value

    for value in node.values():
        if isinstance(value, dict):
            found = _find_key_value(value, *keys)
            if found is not None:
                return found
    return None


def _format_float(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def extract_tiff_metadata(path: Path) -> dict[str, str]:
    """Extract a compact set of TIFF metadata values for CSV output."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        width = _get_tag_value(page, 256) or page.shape[1]
        height = _get_tag_value(page, 257) or page.shape[0]

        image_description = _parse_metadata(_get_tag_value(page, 270))
        fei_data = _parse_metadata(_get_tag_value(page, 34682))
        metadata_source = image_description if image_description else fei_data

        pixel_size_nm: float | None = None
        if isinstance(metadata_source, dict):
            pixel_size_data = _find_key_value(metadata_source, "pixel_size")
            if isinstance(pixel_size_data, dict):
                pixel_size_x = _find_key_value(pixel_size_data, "x")
                pixel_size_y = _find_key_value(pixel_size_data, "y")
                if pixel_size_x is not None and pixel_size_y is not None:
                    pixel_size_nm = ((float(pixel_size_x) + float(pixel_size_y)) / 2.0) * 1e9
                elif pixel_size_x is not None:
                    pixel_size_nm = float(pixel_size_x) * 1e9
                elif pixel_size_y is not None:
                    pixel_size_nm = float(pixel_size_y) * 1e9
            if pixel_size_nm is None:
                pixel_width = _find_key_value(metadata_source, "pixelwidth")
                pixel_height = _find_key_value(metadata_source, "pixelheight")
                if pixel_width is not None and pixel_height is not None:
                    pixel_size_nm = ((float(pixel_width) + float(pixel_height)) / 2.0) * 1e9
                elif pixel_width is not None:
                    pixel_size_nm = float(pixel_width) * 1e9
                elif pixel_height is not None:
                    pixel_size_nm = float(pixel_height) * 1e9

        dwell_time_ns: float | None = None
        if isinstance(metadata_source, dict):
            dwell_time_seconds = _find_key_value(metadata_source, "dwell_time", "dwell", "dwelltime")
            if dwell_time_seconds is None:
                dwell_time_seconds = _find_key_value(metadata_source, "dwell")
            if dwell_time_seconds is not None:
                dwell_time_ns = float(dwell_time_seconds) * 1e9

        image_integrations = 1
        if isinstance(metadata_source, dict):
            value = _find_key_value( metadata_source, "frame_integration", "frameintegration", "integrate",)
            if value is not None:
                image_integrations = value

        line_integrations = 1
        if isinstance(metadata_source, dict):
            value = _find_key_value( metadata_source, "line_integration", "lineintegration",)
            if value is not None:
                line_integrations = value

        microscope_model: Any = None
        if isinstance(metadata_source, dict):
            microscope_model = _find_key_value(metadata_source, "model")
            if microscope_model is None:
                microscope_model = _find_key_value(metadata_source, "systemtype")

        acceleration_voltage: Any = None
        beam_current: Any = None
        if isinstance(metadata_source, dict):
            acceleration_voltage = _find_key_value(
                metadata_source,
                "voltage",
                "hv",
                "accelerating_voltage",
                "acceleration_voltage",
                "acceleratingvoltage",
                "accelerationvoltage",
            )
            beam_current = _find_key_value(metadata_source, "beam_current", "beamcurrent")

        beam_current_pa = None
        if beam_current is not None:
            beam_current_pa = float(beam_current) * 1e12

    return {
        "image_width_pixels": str(width),
        "image_height_pixels": str(height),
        "pixel_size_nm": _format_float(pixel_size_nm),
        "dwell_time_ns": _format_float(dwell_time_ns),
        "image_integrations": "" if image_integrations is None else str(image_integrations),
        "line_integrations": "" if line_integrations is None else str(line_integrations),
        "microscope_model": "" if microscope_model is None else str(microscope_model),
        "acceleration_voltage": _format_float(acceleration_voltage),
        "beam_current_pa": _format_float(beam_current_pa),
    }


def read_input_rows(input_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV is empty or missing a header row.")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def get_path_column(fieldnames: list[str]) -> str:
    for name in ("sem_image", "image_path", "path", "filepath"):
        if name in fieldnames:
            return name
    return fieldnames[0] if fieldnames else ""


def read_session_metadata(metadata_csv: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("Session metadata CSV is empty or missing a header row.")

        metadata_rows = [dict(row) for row in reader]
        metadata_fieldnames = [name for name in reader.fieldnames if name != "session_id"]

    metadata_by_session_id = {
        row.get("session_id", ""): {name: (row.get(name, "") or "") for name in metadata_fieldnames}
        for row in metadata_rows
        if row.get("session_id")
    }
    return metadata_fieldnames, metadata_by_session_id


def resolve_path(raw_path: str, input_csv: Path, base_dir: Path | None = None) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate

    if base_dir is None:
        base_dir = input_csv.parent

    resolved_candidate = (base_dir / candidate).resolve()
    if resolved_candidate.exists():
        return resolved_candidate

    fallback_candidate = (input_csv.parent / candidate).resolve()
    return fallback_candidate


def write_output_csv(
    input_csv: Path,
    output_csv: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    base_dir: Path,
    session_metadata_fieldnames: list[str],
    session_metadata_by_id: dict[str, dict[str, str]],
) -> None:
    path_column = get_path_column(fieldnames)
    output_fieldnames = fieldnames + session_metadata_fieldnames + METADATA_COLUMNS
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()

        for row in rows:
            path_value = (row.get(path_column, "") or "").strip()
            session_id = (row.get("session_id", "") or "").strip()
            session_metadata = session_metadata_by_id.get(session_id, {})

            if not path_value:
                writer.writerow({**row, **{column: "" for column in session_metadata_fieldnames + METADATA_COLUMNS}})
                continue

            path = resolve_path(path_value, input_csv, base_dir)
            if not path.exists() or path.suffix.lower() not in {".tif", ".tiff"}:
                print(f"Skipping unavailable TIFF path: {path}", file=sys.stderr)
                writer.writerow({**row, **session_metadata, **{column: "" for column in METADATA_COLUMNS}})
                continue

            try:
                metadata = extract_tiff_metadata(path)
            except Exception as exc:  # pragma: no cover - runtime error handling
                print(f"Failed to read metadata for {path}: {exc}", file=sys.stderr)
                metadata = {column: "" for column in METADATA_COLUMNS}

            writer.writerow({**row, **session_metadata, **metadata})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="CSV file containing TIFF paths in the sem_image column")
    parser.add_argument("-o", "--output", help="Path for the output CSV (defaults to <input>_with_metadata.csv)")
    parser.add_argument(
        "-b",
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Directory that relative TIFF paths are resolved from",
    )
    parser.add_argument(
        "--session-metadata-csv",
        default=str(DEFAULT_SESSION_METADATA_CSV),
        help="CSV file containing session_id-based metadata to join",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser()
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    base_dir = Path(args.base_dir).expanduser()
    if not base_dir.is_absolute():
        base_dir = (input_csv.parent / base_dir).resolve()

    session_metadata_csv = Path(args.session_metadata_csv).expanduser()
    if not session_metadata_csv.is_absolute():
        session_metadata_csv = (input_csv.parent / session_metadata_csv).resolve()
    if not session_metadata_csv.exists():
        raise SystemExit(f"Session metadata CSV not found: {session_metadata_csv}")

    output_csv = Path(args.output).expanduser() if args.output else input_csv.with_name(f"{input_csv.stem}_with_metadata.csv")
    if not output_csv.is_absolute():
        output_csv = (input_csv.parent / output_csv).resolve()

    rows, fieldnames = read_input_rows(input_csv)
    session_metadata_fieldnames, session_metadata_by_id = read_session_metadata(session_metadata_csv)
    write_output_csv(
        input_csv,
        output_csv,
        rows,
        fieldnames,
        base_dir,
        session_metadata_fieldnames,
        session_metadata_by_id,
    )
    print(f"Wrote metadata CSV to {output_csv}")


if __name__ == "__main__":
    main()
