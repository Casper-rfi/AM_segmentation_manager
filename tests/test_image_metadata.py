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
    condition_disagreements,
    extract_metadata,
    modality_disagreements,
    read_condition_headers,
    read_image_metadata,
    read_records_metadata,
)
from am_segmentation_manager.io.image import has_image_stack_support
from am_segmentation_manager.models import ImageRecord


def fibsem_header(beam: str = ELECTRON) -> dict:
    """A tag-270 document, with **both** beams present as the real ones are."""
    electron_state = {
        "beam_type": "ELECTRON", "voltage": 2000.0, "beam_current": 2.5e-11,
        "dwell_time": 2e-07, "hfw": 4e-05,
        "shift": {"x": 4.5e-06, "y": -1.25e-06},
        "working_distance": 0.007, "scan_rotation": 0.0,
    }
    ion_state = {
        "beam_type": "ION", "voltage": 30000.0, "beam_current": 2e-11,
        "dwell_time": 3e-07, "hfw": 4e-05,
        "shift": {"x": -3.75e-06, "y": 6.25e-07},
        "working_distance": 0.019, "scan_rotation": 3.141592653589793,
    }
    return {
        "experiment": {
            "application": "autolamella", "application_version": "0.5.1",
            "fibsem_version": "0.5.1", "method": "null",
        },
        "image": {
            "beam_type": beam, "resolution": [3072, 2048],
            # The image block records this acquisition, so it follows the beam.
            "dwell_time": 2e-07 if beam == ELECTRON else 3e-07,
            "hfw": 4e-05, "autocontrast": True, "autogamma": False,
        },
        "pixel_size": {"x": 1.3020833333333335e-08, "y": 1.3020833333333333e-08},
        "shape": [2048, 3072],
        "microscope_state": {
            "timestamp": "04/15/2025 05:01:10",
            # Only the beam that took the image is populated. The other is null.
            "electron_beam": electron_state if beam == ELECTRON else None,
            "ion_beam": ion_state if beam == ION else None,
            "electron_detector": (
                {"type": "ETD", "mode": "SecondaryElectrons",
                 "brightness": 0.25, "contrast": 0.75}
                if beam == ELECTRON else None
            ),
            "ion_detector": (
                {"type": "ETD", "mode": "SecondaryElectrons",
                 "brightness": 0.5, "contrast": 0.875}
                if beam == ION else None
            ),
            # One stage, whichever beam fired. Metres and radians, as written.
            "stage_position": {
                "x": -2.28825e-03, "y": 4.1535e-03, "z": 3.2486e-02,
                "r": -1.3089405962989709, "t": 0.3316123777623847,
                "coordinate_system": "RAW",
            },
        },
        # system holds BOTH beams' configured defaults whichever took the image.
        "system": {
            "electron": {"voltage": 2000, "plasma": False, "plasma_gas": None},
            "ion": {"voltage": 30000, "plasma": True, "plasma_gas": "Argon", "current": 2e-11},
            "info": {
                "model": "Arctis", "serial_number": "9952468", "manufacturer": "Thermo",
                # Named the wrong way round at the source: the first is the
                # microscope server's version and the second the API client's.
                "hardware_version": "28.4.1.25560", "software_version": "4.9.0.902",
                "fibsem_version": "0.5.1",
            },
            "stage": {"shuttle_pre_tilt": 26.0, "milling_angle": 15.0, "rotation": True},
        },
    }


def fei_header(beam: str = ELECTRON) -> dict:
    """A tag-34682 mapping, with both beam columns present as the real ones are."""
    return {
        "User": {"Date": "01/28/2024", "Time": "02:53:08 PM"},
        "System": {"SystemType": "Arctis", "Dnumber": 9952468, "Type": "DualBeam",
                   "Software": "28.4.0.24398"},
        "Beam": {"HV": 2000 if beam == ELECTRON else 30000,
                 "Beam": "EBeam" if beam == ELECTRON else "IBeam"},
        "EBeam": {"HV": 2000, "BeamCurrent": 1e-10, "HFW": 4.144e-05, "Source": "FEG",
                  "BeamShiftX": -1.72956e-05, "BeamShiftY": 9.09873e-07,
                  "WD": 0.00983654, "ScanRotation": 3.14159},
        "IBeam": {"HV": 30000, "BeamCurrent": 2e-11, "HFW": 2.59e-05, "Source": "Argon",
                  "BeamShiftX": 2.5e-06, "BeamShiftY": -1.5e-06,
                  "WD": 0.0165, "ScanRotation": 0.0},
        "Scan": {"Dwelltime": 5e-08, "PixelWidth": 1.34896e-08, "Integrate": 16},
        "EScan": {"Dwell": 5e-08, "PixelWidth": 1.34896e-08, "LineIntegration": 1},
        "IScan": {"Dwell": 5e-08, "PixelWidth": 8.43099e-09, "LineIntegration": 4},
        "Image": {"ResolutionX": 3072, "ResolutionY": 2048, "Integrate": 16},
        "Detectors": {"Name": "ETD", "Mode": "SE"},
        # A cryo stage states its position under Compustage*, and leaves the
        # flat stage's keys holding uninitialised memory rather than nothing.
        "Stage": {
            "StageX": 6.95268e-310, "StageY": 0, "StageZ": "",
            "StageR": "", "StageT": "",
            "CompustageX": 4.80519e-04, "CompustageY": 2.14798e-04,
            "CompustageZ": -2.66696e-05, "CompustageRX": -0.453719,
        },
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
    assert metadata.acceleration_voltage_kv == 2.0
    assert metadata.beam_current_pa == pytest.approx(25.0)
    assert metadata.dwell_time_ns == pytest.approx(200.0)


def test_an_ion_image_reads_the_ion_beam():
    metadata = extract_metadata(fibsem_header(ION), "fibsem")

    assert metadata.beam_type == ION
    assert metadata.modality == "FIB"
    assert metadata.acceleration_voltage_kv == 30.0
    assert metadata.beam_current_pa == pytest.approx(20.0)
    assert metadata.dwell_time_ns == pytest.approx(300.0)


def test_only_an_ion_image_reports_a_plasma_gas():
    assert extract_metadata(fibsem_header(ELECTRON), "fibsem").plasma_gas is None
    assert extract_metadata(fibsem_header(ION), "fibsem").plasma_gas == "Argon"


def test_a_gas_recorded_under_the_electron_column_is_not_the_sems_gas():
    """A fibsem header carries system.electron.plasma_gas whatever beam fired,
    and an electron column has no plasma. Reading it reported Argon for SEM."""
    document = fibsem_header(ELECTRON)
    document["system"]["electron"]["plasma_gas"] = "Argon"

    assert extract_metadata(document, "fibsem").plasma_gas is None


def test_an_emitter_is_never_reported_as_a_plasma_gas():
    """Both are `Source` under their own column in the legacy dialect, so one
    field for the two reported the SEM's FEG emitter as a plasma gas."""
    electron = extract_metadata(fei_header(ELECTRON), "fei")
    ion = extract_metadata(fei_header(ION), "fei")

    assert (electron.electron_source, electron.plasma_gas) == ("FEG", None)
    assert (ion.plasma_gas, ion.electron_source) == ("Argon", None)


def test_the_fibsem_dialect_records_no_electron_emitter():
    assert extract_metadata(fibsem_header(ELECTRON), "fibsem").electron_source is None


def test_the_legacy_dialect_reads_the_matching_beam_column():
    electron = extract_metadata(fei_header(ELECTRON), "fei")
    ion = extract_metadata(fei_header(ION), "fei")

    assert (electron.acceleration_voltage_kv, ion.acceleration_voltage_kv) == (2.0, 30.0)
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

    assert extract_metadata(document, "fibsem").acceleration_voltage_kv is None


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def test_units_are_converted_from_the_si_values_headers_store():
    metadata = extract_metadata(fibsem_header(ION), "fibsem")

    assert metadata.pixel_size_nm == pytest.approx(13.02, abs=0.01)   # m  -> nm
    assert metadata.dwell_time_ns == pytest.approx(300.0)             # s  -> ns
    assert metadata.beam_current_pa == pytest.approx(20.0)            # A  -> pA
    assert metadata.hfw_um == pytest.approx(40.0)                     # m  -> um
    assert metadata.acceleration_voltage_kv == pytest.approx(30.0)    # V  -> kV


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


def test_a_converted_value_carries_no_noise_from_the_conversion():
    """The values below are transcribed from real headers. Neither is odd:
    6e-05 has no exact binary representation, so the header already holds
    5.9999999999999995e-05, and multiplying by 1e6 exposes it."""
    document = fibsem_header(ION)
    document["image"]["hfw"] = 5.9999999999999995e-05
    document["microscope_state"]["ion_beam"]["beam_current"] = 2e-09
    metadata = extract_metadata(document, "fibsem")

    assert metadata.hfw_um == 60.0
    assert metadata.beam_current_pa == 2000.0


def test_a_measured_value_is_kept_rather_than_snapped_to_its_set_point():
    """Both of these are what the instrument reported, not conversion noise: a
    30 kV column measured at 29996 V, drawing 10 pA against a 30 pA preset.
    Rounding must not turn either into the round number it is near."""
    document = fibsem_header(ION)
    document["microscope_state"]["ion_beam"]["voltage"] = 29996.41597249603
    document["microscope_state"]["ion_beam"]["beam_current"] = 1.0002222222222226e-11
    metadata = extract_metadata(document, "fibsem")

    assert metadata.acceleration_voltage_kv == 29.996416
    assert metadata.beam_current_pa == 10.002222


def test_uninitialised_memory_is_not_a_measurement():
    """A subnormal float is what a dialect leaves in a key it never filled in,
    and a NaN is not a number the instrument measured. Both read as absent."""
    document = fibsem_header(ION)
    del document["image"]["hfw"]
    document["microscope_state"]["ion_beam"]["hfw"] = 6.95268e-310
    document["microscope_state"]["ion_beam"]["voltage"] = float("nan")
    metadata = extract_metadata(document, "fibsem")

    assert metadata.hfw_um is None
    assert metadata.acceleration_voltage_kv is None


def test_a_path_stating_nothing_falls_through_to_the_next():
    """image.hfw describes this acquisition and the beam state is the fallback.
    A key present but holding nothing usable must not hide the fallback."""
    document = fibsem_header(ION)
    document["image"]["hfw"] = 6.95268e-310
    document["microscope_state"]["ion_beam"]["hfw"] = 4e-05

    assert extract_metadata(document, "fibsem").hfw_um == 40.0


# ---------------------------------------------------------------------------
# Where the image was taken
# ---------------------------------------------------------------------------

def test_the_stage_position_is_read_in_micrometres_and_degrees():
    metadata = extract_metadata(fibsem_header(ION), "fibsem")

    assert metadata.stage_x_um == pytest.approx(-2288.25)
    assert metadata.stage_y_um == pytest.approx(4153.5)
    assert metadata.stage_z_um == pytest.approx(32486.0)
    assert metadata.stage_rotation_deg == pytest.approx(-74.996772)
    assert metadata.stage_tilt_deg == pytest.approx(18.99999)


def test_the_coordinate_system_is_carried_with_the_position():
    """This collection holds both RAW and specimen positions, and a z of
    32486 um means nothing next to one of -59 um without saying which."""
    assert extract_metadata(fibsem_header(), "fibsem").stage_coordinate_system == "RAW"


def test_the_stage_is_the_same_stage_whichever_beam_fired():
    electron = extract_metadata(fibsem_header(ELECTRON), "fibsem")
    ion = extract_metadata(fibsem_header(ION), "fibsem")

    assert electron.stage_x_um == ion.stage_x_um
    assert electron.stage_tilt_deg == ion.stage_tilt_deg


def test_a_cryo_stage_is_read_from_its_own_keys_not_the_ones_left_unset():
    """Stage.StageX holds a subnormal and Stage.StageY a plausible zero, while
    the position the stage was actually at is under Compustage*."""
    metadata = extract_metadata(fei_header(), "fei")

    assert metadata.stage_x_um == pytest.approx(480.519)
    assert metadata.stage_y_um == pytest.approx(214.798)
    assert metadata.stage_z_um == pytest.approx(-26.6696)
    assert metadata.stage_tilt_deg == pytest.approx(-25.996184)


def test_a_compustage_states_no_rotation_and_no_coordinate_system():
    """It states a tilt about x and nothing about z. Absent stays absent."""
    metadata = extract_metadata(fei_header(), "fei")

    assert metadata.stage_rotation_deg is None
    assert metadata.stage_coordinate_system is None


def test_beam_shift_follows_the_beam_that_took_the_image():
    """Both dialects carry both columns' shift, and reading the wrong one
    yields a plausible number rather than an obvious mistake."""
    fibsem = [
        extract_metadata(fibsem_header(ELECTRON), "fibsem"),
        extract_metadata(fibsem_header(ION), "fibsem"),
    ]
    fei = [
        extract_metadata(fei_header(ELECTRON), "fei"),
        extract_metadata(fei_header(ION), "fei"),
    ]

    assert [(m.beam_shift_x_um, m.beam_shift_y_um) for m in fibsem] == [
        (4.5, -1.25), (-3.75, 0.625),
    ]
    assert [(m.beam_shift_x_um, m.beam_shift_y_um) for m in fei] == [
        (-17.2956, 0.909873), (2.5, -1.5),
    ]


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

    assert metadata.acceleration_voltage_kv is None
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

    assert extract_metadata(document, "fibsem").plasma_gas is None


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


def test_the_manufacturer_is_read_where_fibsem_records_it_and_never_inferred():
    """The FEI dialect names no maker; its tag is not taken to be one."""
    assert extract_metadata(fibsem_header(), "fibsem").microscope_manufacturer == "Thermo"
    assert extract_metadata(fei_header(), "fei").microscope_manufacturer is None


@pytest.mark.parametrize("loc", ["C", "de_DE.UTF-8", "fr_FR.UTF-8", "ja_JP.UTF-8"])
def test_acquisition_time_does_not_depend_on_the_locale(loc):
    """The header says "PM" whatever machine it was written on.

    strptime's %p matches whatever the *current locale* calls the halves of the
    day, so parsing with it read the time on one developer's machine and
    returned None on another's -- silently, since an unparsed time is simply
    absent. The marker is now stripped and applied by hand.
    """
    import locale

    previous = locale.setlocale(locale.LC_TIME)
    try:
        try:
            locale.setlocale(locale.LC_TIME, loc)
        except locale.Error:
            pytest.skip(f"locale {loc} not available here")
        assert extract_metadata(fei_header(), "fei").acquired_at == datetime(
            2024, 1, 28, 14, 53, 8
        )
    finally:
        locale.setlocale(locale.LC_TIME, previous)


@pytest.mark.parametrize(
    "time, expected_hour",
    [("12:30:00 AM", 0), ("01:30:00 AM", 1), ("12:30:00 PM", 12), ("11:30:00 PM", 23)],
)
def test_midnight_and_noon_are_the_awkward_cases(time: str, expected_hour: int):
    """12 AM is midnight and 12 PM is noon, so the hour wraps rather than
    simply gaining twelve."""
    document = fei_header()
    document["User"]["Time"] = time

    assert extract_metadata(document, "fei").acquired_at.hour == expected_hour


def test_a_24_hour_time_without_a_marker_still_parses():
    document = fei_header()
    document["User"]["Time"] = "14:53:08"

    assert extract_metadata(document, "fei").acquired_at == datetime(2024, 1, 28, 14, 53, 8)


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

#: Reading a header needs the image stack, so each reading test skips without it.
needs_image_stack = pytest.mark.skipif(
    not has_image_stack_support(), reason="optional image libs not installed"
)

@needs_image_stack
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
    assert metadata.acceleration_voltage_kv == 30.0


@needs_image_stack
def test_an_unreadable_file_is_reported_as_absent_rather_than_raising(tmp_path: Path):
    """12 of 2680 images in the reference collection fail to parse. A batch
    over thousands must not stop for one of them."""
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a tiff at all")

    assert read_image_metadata(broken) is None


@needs_image_stack
def test_a_tiff_without_a_recognised_header_is_absent_not_empty(tmp_path: Path):
    import numpy as np
    import tifffile

    path = tmp_path / "plain.tif"
    tifffile.imwrite(str(path), np.zeros((4, 4), dtype=np.uint8))

    assert read_image_metadata(path) is None


@needs_image_stack
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


# ---------------------------------------------------------------------------
# Imaging geometry, and what the acquisition did to its own grey levels
# ---------------------------------------------------------------------------

def test_the_imaging_geometry_is_read_in_degrees_and_millimetres():
    metadata = extract_metadata(fibsem_header(ION), "fibsem")

    assert metadata.scan_rotation_deg == pytest.approx(180.0)
    assert metadata.working_distance_mm == pytest.approx(19.0)
    assert metadata.shuttle_pretilt_deg == pytest.approx(26.0)
    assert metadata.milling_angle_deg == pytest.approx(15.0)


def test_the_working_distance_and_scan_rotation_follow_the_beam():
    """The columns are focused independently and scanned independently, so
    reading the other one's gives a number that looks perfectly ordinary."""
    electron = extract_metadata(fibsem_header(ELECTRON), "fibsem")
    ion = extract_metadata(fibsem_header(ION), "fibsem")

    assert (electron.working_distance_mm, ion.working_distance_mm) == (7.0, 19.0)
    assert (electron.scan_rotation_deg, ion.scan_rotation_deg) == (0.0, 180.0)
    assert extract_metadata(fei_header(ELECTRON), "fei").working_distance_mm == pytest.approx(
        9.83654
    )


def test_the_stage_geometry_is_the_same_for_both_beams():
    """Unlike everything else read out of system.*, the stage has one geometry
    whichever column fired, so there is no other beam's value to confuse."""
    electron = extract_metadata(fibsem_header(ELECTRON), "fibsem")
    ion = extract_metadata(fibsem_header(ION), "fibsem")

    assert electron.shuttle_pretilt_deg == ion.shuttle_pretilt_deg == 26.0
    assert electron.milling_angle_deg == ion.milling_angle_deg == 15.0


def test_an_acquisition_says_whether_it_processed_its_own_grey_levels():
    metadata = extract_metadata(fibsem_header(), "fibsem")

    assert metadata.autocontrast is True
    assert metadata.autogamma is False


def test_a_flag_that_is_off_is_not_the_same_as_one_that_is_absent():
    """Autocontrast off means the grey levels are the detector's own, which is
    a fact about the image; a header not saying is not."""
    document = fibsem_header()
    document["image"]["autocontrast"] = False
    del document["image"]["autogamma"]
    metadata = extract_metadata(document, "fibsem")

    assert metadata.autocontrast is False
    assert metadata.autogamma is None


def test_detector_settings_are_read_for_the_detector_that_was_used():
    electron = extract_metadata(fibsem_header(ELECTRON), "fibsem")
    ion = extract_metadata(fibsem_header(ION), "fibsem")

    assert (electron.detector_brightness, electron.detector_contrast) == (0.25, 0.75)
    assert (ion.detector_brightness, ion.detector_contrast) == (0.5, 0.875)


def test_the_fei_dialect_states_no_comparable_detector_setting():
    """It writes them under a section named after the detector, on a 0-100
    scale rather than as a fraction. One column would hold two quantities."""
    metadata = extract_metadata(fei_header(), "fei")

    assert metadata.detector_brightness is None
    assert metadata.detector_contrast is None
    assert metadata.detector_type == "ETD", "which detector is still read"


# ---------------------------------------------------------------------------
# Three programs took part, and each states its own version
# ---------------------------------------------------------------------------

def test_the_three_software_versions_are_kept_apart():
    """The microscope's own software, the scripting interface driving it, and
    the application that asked. They are on separate release series -- xT at
    28.x, AutoScript at 4.x -- so one column for "the software version" would
    answer a different question for each image."""
    metadata = extract_metadata(fibsem_header(), "fibsem")

    assert metadata.microscope_software_version == "28.4.1.25560"
    assert metadata.microscope_api_version == "4.9.0.902"
    assert metadata.acquisition_software == "autolamella"
    assert metadata.acquisition_software_version == "0.5.1"


def test_the_microscope_software_version_is_read_from_both_dialects():
    """The one of the three the FEI headers state, which is what makes it the
    only version column covering the pre-Autolamella datasets."""
    metadata = extract_metadata(fei_header(), "fei")

    assert metadata.microscope_software_version == "28.4.0.24398"
    assert metadata.microscope_api_version is None
    assert metadata.acquisition_software is None


def test_an_epoch_timestamp_is_read_as_utc_and_says_so():
    """fibsemOS writes a stitched overview's time as seconds since the epoch,
    which names no zone; read as UTC it means the same on every machine."""
    from am_segmentation_manager.image_metadata import extract_metadata

    metadata = extract_metadata(
        {"microscope_state": {"timestamp": 1775639052.187201}}, "fibsem"
    )

    assert metadata.acquired_at == datetime(2026, 4, 8, 9, 4, 12)
    assert metadata.acquired_at_timezone == "UTC"


def test_a_clock_time_is_left_as_written_and_names_no_zone():
    from am_segmentation_manager.image_metadata import extract_metadata

    metadata = extract_metadata(
        {"microscope_state": {"timestamp": "04/08/2026 10:40:01"}}, "fibsem"
    )

    assert metadata.acquired_at == datetime(2026, 4, 8, 10, 40, 1)
    assert metadata.acquired_at_timezone is None



# ---------------------------------------------------------------------------
# Conditions of one family
# ---------------------------------------------------------------------------

def _header(width: int = 3072, pixel: float = 13.49, kv: float | None = 2.0) -> ImageMetadata:
    return ImageMetadata(
        image_width=width, image_height=2048, pixel_size_nm=pixel, acceleration_voltage_kv=kv
    )


def test_conditions_of_one_acquisition_agree():
    assert condition_disagreements({"sem": _header(), "sem_fi8": _header()}) == []


def test_each_quantity_a_family_disagrees_about_is_named():
    found = condition_disagreements(
        {"sem": _header(), "sem_mag2": _header(width=1536, pixel=26.98, kv=5.0)}
    )

    assert [(item.quantity, item.blocks_sharing) for item in found] == [
        ("image size", True), ("pixel size", False), ("accelerating voltage", False),
    ]
    assert found[1].describe() == "pixel size: sem 13.49 nm, sem_mag2 26.98 nm"


def test_a_header_that_does_not_state_a_quantity_takes_no_part_in_it():
    assert condition_disagreements({"sem": _header(kv=None), "sem_fi8": _header(kv=5.0)}) == []


def test_values_that_print_alike_agree():
    """Rounding noise in a header is not a different acquisition."""
    assert condition_disagreements({"sem": _header(), "sem_fi8": _header(pixel=13.49 + 1e-9)}) == []


def test_one_acquisition_is_read_in_every_condition(tmp_path: Path):
    read: list[Path] = []
    images = {
        "sem": {"a.tif": tmp_path / "sem/a.tif", "b.tif": tmp_path / "sem/b.tif"},
        "sem_fi8": {"b.tif": tmp_path / "sem_fi8/b.tif", "c.tif": tmp_path / "sem_fi8/c.tif"},
    }

    [(filename, headers)] = read_condition_headers(
        images, lambda path: read.append(path) or _header()
    )

    assert filename == "b.tif"
    assert sorted(read) == [tmp_path / "sem/b.tif", tmp_path / "sem_fi8/b.tif"]
    assert set(headers) == {"sem", "sem_fi8"}


def test_conditions_sharing_no_acquisition_are_not_read(tmp_path: Path):
    images = {"sem": {"a.tif": tmp_path / "a.tif"}, "sem_fi8": {"b.tif": tmp_path / "b.tif"}}

    assert read_condition_headers(images, lambda path: _header()) == []


def test_conditions_lacking_a_common_acquisition_are_still_compared(tmp_path: Path):
    """No filename is held by all three, yet each pair sharing one shares a label."""
    images = {
        "sem": {name: tmp_path / "sem" / name for name in ("a.tif", "b.tif")},
        "sem_fi1": {"c.tif": tmp_path / "sem_fi1/c.tif"},
        "sem2kVb": {name: tmp_path / "sem2kVb" / name for name in ("a.tif", "c.tif")},
    }

    readings = read_condition_headers(images, lambda path: _header())

    assert [(name, sorted(headers)) for name, headers in readings] == [
        ("a.tif", ["sem", "sem2kVb"]),
        ("c.tif", ["sem2kVb", "sem_fi1"]),
    ]


def test_one_filename_held_by_every_condition_is_read_once(tmp_path: Path):
    images = {
        condition: {name: tmp_path / condition / name for name in ("a.tif", "b.tif")}
        for condition in ("sem", "sem_fi1", "sem_fi4")
    }

    readings = read_condition_headers(images, lambda path: _header())

    assert [name for name, _ in readings] == ["a.tif"]
