"""Tests for reading acquisition metadata from image headers.

Header documents are built as plain dictionaries and passed to
:func:`extract_metadata`, rather than written into TIFFs and read back. The
mapping from header keys to fields is what is under test; going through a file
would test tifffile as well and make each case slower and harder to read. One
round-trip test covers the file reading itself.

The structures below are transcribed from real headers -- an Arctis writing the
fibsem JSON, and the same instrument four years earlier writing the FEI tag.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from am_segmentation_manager.image_metadata import (
    ELECTRON,
    ION,
    ImageMetadata,
    extract_metadata,
    modality_disagreements,
    read_image_metadata,
    read_records_metadata,
)
from am_segmentation_manager.models import ImageRecord


def fibsem_header(beam: str = ELECTRON) -> dict:
    """A tag-270 document, with **both** beams present as the real ones are."""
    electron_state = {
        "beam_type": "ELECTRON", "voltage": 2000.0, "beam_current": 2.5e-11,
        "dwell_time": 2e-07, "hfw": 4e-05,
    }
    ion_state = {
        "beam_type": "ION", "voltage": 30000.0, "beam_current": 2e-11,
        "dwell_time": 3e-07, "hfw": 4e-05,
    }
    return {
        "image": {
            "beam_type": beam, "resolution": [3072, 2048],
            # The image block records this acquisition, so it follows the beam.
            "dwell_time": 2e-07 if beam == ELECTRON else 3e-07,
            "hfw": 4e-05,
        },
        "pixel_size": {"x": 1.3020833333333335e-08, "y": 1.3020833333333333e-08},
        "shape": [2048, 3072],
        "microscope_state": {
            "timestamp": "04/15/2025 05:01:10",
            # Only the beam that took the image is populated. The other is null.
            "electron_beam": electron_state if beam == ELECTRON else None,
            "ion_beam": ion_state if beam == ION else None,
            "electron_detector": (
                {"type": "ETD", "mode": "SecondaryElectrons"} if beam == ELECTRON else None
            ),
            "ion_detector": {"type": "ETD", "mode": "SecondaryElectrons"} if beam == ION else None,
        },
        # system holds BOTH beams' configured defaults whichever took the image.
        "system": {
            "electron": {"voltage": 2000, "plasma": False, "plasma_gas": None},
            "ion": {"voltage": 30000, "plasma": True, "plasma_gas": "Argon", "current": 2e-11},
            "info": {"model": "Arctis", "serial_number": "9952468", "manufacturer": "Thermo"},
        },
    }


def fei_header(beam: str = ELECTRON) -> dict:
    """A tag-34682 mapping, with both beam columns present as the real ones are."""
    return {
        "User": {"Date": "01/28/2024", "Time": "02:53:08 PM"},
        "System": {"SystemType": "Arctis", "Dnumber": 9952468, "Type": "DualBeam"},
        "Beam": {"HV": 2000 if beam == ELECTRON else 30000,
                 "Beam": "EBeam" if beam == ELECTRON else "IBeam"},
        "EBeam": {"HV": 2000, "BeamCurrent": 1e-10, "HFW": 4.144e-05, "Source": "FEG"},
        "IBeam": {"HV": 30000, "BeamCurrent": 2e-11, "HFW": 2.59e-05, "Source": "Argon"},
        "Scan": {"Dwelltime": 5e-08, "PixelWidth": 1.34896e-08, "Integrate": 16},
        "EScan": {"Dwell": 5e-08, "PixelWidth": 1.34896e-08, "LineIntegration": 1},
        "IScan": {"Dwell": 5e-08, "PixelWidth": 8.43099e-09, "LineIntegration": 4},
        "Image": {"ResolutionX": 3072, "ResolutionY": 2048, "Integrate": 16},
        "Detectors": {"Name": "ETD", "Mode": "SE"},
    }


# ---------------------------------------------------------------------------
# Only the beam that took the image
#
# Both dialects carry settings for both beams in every image. Reading the wrong
# column yields a number that looks entirely plausible, which is the failure
# mode worth the most tests.
# ---------------------------------------------------------------------------

def test_an_electron_image_reads_the_electron_beam():
    metadata = extract_metadata(fibsem_header(ELECTRON), "fibsem")

    assert metadata.beam_type == ELECTRON
    assert metadata.modality == "SEM"
    assert metadata.acceleration_voltage_v == 2000.0
    assert metadata.beam_current_pa == pytest.approx(25.0)
    assert metadata.dwell_time_ns == pytest.approx(200.0)


def test_an_ion_image_reads_the_ion_beam():
    metadata = extract_metadata(fibsem_header(ION), "fibsem")

    assert metadata.beam_type == ION
    assert metadata.modality == "FIB"
    assert metadata.acceleration_voltage_v == 30000.0
    assert metadata.beam_current_pa == pytest.approx(20.0)
    assert metadata.dwell_time_ns == pytest.approx(300.0)


def test_an_electron_image_never_reports_the_ion_beams_source():
    """Regression: beam_source was read from system.ion regardless of beam, so
    every SEM image reported the FIB's Argon."""
    assert extract_metadata(fibsem_header(ELECTRON), "fibsem").beam_source is None
    assert extract_metadata(fibsem_header(ION), "fibsem").beam_source == "Argon"


def test_each_beam_reports_its_own_source_in_the_legacy_dialect():
    """Regression: the field was called plasma_gas, so an SEM reported its FEG
    emitter as a plasma gas."""
    assert extract_metadata(fei_header(ELECTRON), "fei").beam_source == "FEG"
    assert extract_metadata(fei_header(ION), "fei").beam_source == "Argon"


def test_the_legacy_dialect_reads_the_matching_beam_column():
    electron = extract_metadata(fei_header(ELECTRON), "fei")
    ion = extract_metadata(fei_header(ION), "fei")

    assert (electron.acceleration_voltage_v, ion.acceleration_voltage_v) == (2000.0, 30000.0)
    assert electron.beam_current_pa == pytest.approx(100.0)
    assert ion.beam_current_pa == pytest.approx(20.0)
    # EScan vs IScan, which differ here precisely so a mix-up is visible.
    assert electron.line_integrations == 1
    assert ion.line_integrations == 4


def test_a_beam_specific_search_does_not_cross_to_the_other_beam():
    """The recursive fallback must stay inside the beam that took the image.

    Here the declared path is missing, so the search runs -- and the only
    voltage outside the electron subtree is the ion beam's 30 kV. Reporting
    that for an SEM image would be a plausible, wrong number.
    """
    document = fibsem_header(ELECTRON)
    del document["microscope_state"]["electron_beam"]["voltage"]

    assert extract_metadata(document, "fibsem").acceleration_voltage_v is None


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def test_units_are_converted_from_the_si_values_headers_store():
    metadata = extract_metadata(fibsem_header(ION), "fibsem")

    assert metadata.pixel_size_nm == pytest.approx(13.02, abs=0.01)   # m  -> nm
    assert metadata.dwell_time_ns == pytest.approx(300.0)             # s  -> ns
    assert metadata.beam_current_pa == pytest.approx(20.0)            # A  -> pA
    assert metadata.hfw_um == pytest.approx(40.0)                     # m  -> um


def test_beam_current_needs_no_special_case_for_the_ion_beam():
    """Both dialects and both beams record amps, so one conversion covers all."""
    values = [
        extract_metadata(fibsem_header(ELECTRON), "fibsem").beam_current_pa,
        extract_metadata(fibsem_header(ION), "fibsem").beam_current_pa,
        extract_metadata(fei_header(ELECTRON), "fei").beam_current_pa,
        extract_metadata(fei_header(ION), "fei").beam_current_pa,
    ]
    assert values == pytest.approx([25.0, 20.0, 100.0, 20.0])


def test_dimensions_are_integers_not_floats():
    metadata = extract_metadata(fibsem_header(), "fibsem")

    assert (metadata.image_width, metadata.image_height) == (3072, 2048)
    assert isinstance(metadata.image_width, int)


# ---------------------------------------------------------------------------
# Absent is None, never a default
# ---------------------------------------------------------------------------

def test_integrations_are_read_from_the_fibsem_image_block():
    """The 2024 fibsem headers do record these, in image.frame_integration.

    An earlier draft asserted the schema had no integration fields at all,
    generalised from a single 2025 sample that omitted them. Comparing against
    the standalone extraction script on real data found the miss.
    """
    document = fibsem_header()
    document["image"]["frame_integration"] = 8
    document["image"]["line_integration"] = 2
    metadata = extract_metadata(document, "fibsem")

    assert metadata.image_integrations == 8
    assert metadata.line_integrations == 2


def test_a_null_integration_reads_as_one_pass_not_as_unknown():
    """A null here is the field saying integration was not enabled, and not
    enabled is a single pass -- an image exists, so the probe crossed each line
    at least once. This is the one place absence has a defined physical
    meaning; everything else stays None."""
    document = fibsem_header()
    document["image"]["frame_integration"] = None
    document["image"]["line_integration"] = None
    metadata = extract_metadata(document, "fibsem")

    assert metadata.image_integrations == 1
    assert metadata.line_integrations == 1


def test_only_the_integration_counts_have_a_floor():
    """A missing voltage must not become a plausible number."""
    document = fibsem_header(ELECTRON)
    del document["microscope_state"]["electron_beam"]["voltage"]
    del document["microscope_state"]["electron_beam"]["beam_current"]
    metadata = extract_metadata(document, "fibsem")

    assert metadata.acceleration_voltage_v is None
    assert metadata.beam_current_pa is None


def test_integrations_absent_from_the_document_read_as_one():
    """Absence means integration was not enabled, which is one pass."""
    metadata = extract_metadata(fibsem_header(), "fibsem")

    assert metadata.image_integrations == 1
    assert metadata.line_integrations == 1


def test_the_image_block_wins_over_the_beam_state_and_the_system_default():
    """One real header carries three different dwell times: 200ns for the
    image, 300ns for the beam state, 1000ns as the system default. The image
    block describes this acquisition, so it is the one that counts."""
    document = fibsem_header(ELECTRON)
    document["image"]["dwell_time"] = 2e-07
    document["microscope_state"]["electron_beam"]["dwell_time"] = 3e-07
    document["system"]["electron"]["dwell_time"] = 1e-06

    assert extract_metadata(document, "fibsem").dwell_time_ns == pytest.approx(200.0)


def test_a_serialised_null_string_is_not_a_value():
    """The fibsem writer emits Python None as the string "None" in places --
    141 FIB images in the reference collection carry plasma_gas: "None"."""
    document = fibsem_header(ION)
    document["system"]["ion"]["plasma_gas"] = "None"

    assert extract_metadata(document, "fibsem").beam_source is None


def test_a_bare_shape_document_is_not_an_acquisition_header():
    """tifffile writes {"shape": [...]} into the description of any array it
    saves, and Autolamella's autofocus images carry exactly that. Treating it
    as a header would report a width and height for a file recording no
    acquisition settings at all."""
    from am_segmentation_manager.image_metadata import FIBSEM_MARKER_KEYS

    assert not FIBSEM_MARKER_KEYS & {"shape"}


def test_integrations_are_read_where_the_dialect_does_record_them():
    metadata = extract_metadata(fei_header(ELECTRON), "fei")

    assert metadata.image_integrations == 16
    assert metadata.line_integrations == 1


def test_an_empty_document_yields_no_invented_values():
    """Not even the integration floor: with no evidence of an acquisition there
    is nothing to infer one pass *of*."""
    metadata = extract_metadata({}, "fibsem")

    assert metadata.is_empty
    assert metadata.beam_type is None
    assert metadata.modality is None
    assert metadata.image_integrations is None


def test_blank_strings_are_not_values():
    """The FEI dialect writes '' for settings that do not apply."""
    document = fei_header(ELECTRON)
    document["EBeam"]["BeamCurrent"] = ""

    assert extract_metadata(document, "fei").beam_current_pa is None


# ---------------------------------------------------------------------------
# Everything else
# ---------------------------------------------------------------------------

def test_the_instrument_is_identified_the_same_way_in_both_dialects():
    """The same Arctis, four years and two software generations apart."""
    modern = extract_metadata(fibsem_header(), "fibsem")
    legacy = extract_metadata(fei_header(), "fei")

    assert modern.microscope_model == legacy.microscope_model == "Arctis"
    assert modern.microscope_serial == legacy.microscope_serial == "9952468"


def test_acquisition_time_is_parsed_from_both_dialects():
    assert extract_metadata(fibsem_header(), "fibsem").acquired_at == datetime(2025, 4, 15, 5, 1, 10)
    assert extract_metadata(fei_header(), "fei").acquired_at == datetime(2024, 1, 28, 14, 53, 8)


def test_the_dialect_is_recorded():
    assert extract_metadata(fibsem_header(), "fibsem").source == "fibsem"
    assert extract_metadata(fei_header(), "fei").source == "fei"


def test_beam_is_inferred_when_the_header_does_not_state_it():
    document = fibsem_header(ION)
    del document["image"]["beam_type"]

    assert extract_metadata(document, "fibsem").beam_type == ION


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------

def test_round_trip_through_a_real_tiff(tmp_path: Path):
    """One test that the tag numbers and JSON handling are right."""
    import json

    import numpy as np
    import tifffile

    path = tmp_path / "img.tif"
    tifffile.imwrite(
        str(path), np.zeros((4, 4), dtype=np.uint8),
        description=json.dumps(fibsem_header(ION)),
    )
    metadata = read_image_metadata(path)

    assert metadata is not None
    assert metadata.source == "fibsem"
    assert metadata.beam_type == ION
    assert metadata.acceleration_voltage_v == 30000.0


def test_an_unreadable_file_is_reported_as_absent_rather_than_raising(tmp_path: Path):
    """12 of 2680 images in the reference collection fail to parse. A batch
    over thousands must not stop for one of them."""
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a tiff at all")

    assert read_image_metadata(broken) is None


def test_a_tiff_without_a_recognised_header_is_absent_not_empty(tmp_path: Path):
    import numpy as np
    import tifffile

    path = tmp_path / "plain.tif"
    tifffile.imwrite(str(path), np.zeros((4, 4), dtype=np.uint8))

    assert read_image_metadata(path) is None


def test_records_without_readable_headers_are_simply_missing(tmp_path: Path):
    import json

    import numpy as np
    import tifffile

    good = tmp_path / "good.tif"
    bad = tmp_path / "bad.tif"
    tifffile.imwrite(str(good), np.zeros((4, 4), dtype=np.uint8),
                     description=json.dumps(fibsem_header(ELECTRON)))
    bad.write_bytes(b"junk")

    result = read_records_metadata([ImageRecord(image_path=good), ImageRecord(image_path=bad)])

    assert set(result) == {good}


# ---------------------------------------------------------------------------
# Cross-checking the directory against the instrument
# ---------------------------------------------------------------------------

def test_a_header_disagreeing_with_its_directory_is_reported():
    """The directory stays authoritative; the instrument's own statement is a
    cross-check, and a mismatch usually means a file was copied to the wrong
    place. It is reported, never silently corrected."""
    record = ImageRecord(image_path=Path("stage/sem/img.tif"), modality="SEM")
    metadata = {record.image_path: extract_metadata(fibsem_header(ION), "fibsem")}

    assert modality_disagreements([record], metadata) == [(record.image_path, "SEM", "FIB")]


def test_agreement_produces_no_finding():
    record = ImageRecord(image_path=Path("stage/sem/img.tif"), modality="SEM")
    metadata = {record.image_path: extract_metadata(fibsem_header(ELECTRON), "fibsem")}

    assert modality_disagreements([record], metadata) == []


def test_a_record_with_no_metadata_is_not_a_disagreement():
    record = ImageRecord(image_path=Path("stage/sem/img.tif"), modality="SEM")

    assert modality_disagreements([record], {}) == []


def test_metadata_is_not_attached_to_records():
    """Metadata must stay out of the scan: a mapping keyed by path, so the cost
    is paid only by commands that ask for it."""
    assert not hasattr(ImageRecord(image_path=Path("x.tif")), "metadata")
    assert "metadata" not in ImageMetadata.__dataclass_fields__
