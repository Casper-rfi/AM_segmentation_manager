"""Tests for ``scripts/refine_overview_positions.py``, the parts with no window.

The answers it records are a person's work, so what matters most is that
recording one never loses another; and the first guess has to say honestly
where it came from.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("tifffile")

from refine_overview_positions import (  # noqa: E402
    answered,
    first_guess,
    load_document,
    record,
    save,
    turned_about,
)

GEOMETRY = {"width": 600, "height": 600}


def entry(name: str, x: float) -> dict:
    return {"name": name, "answer": "placed", "x": x, "y": 0.0}


def test_an_answer_replaces_only_that_lamella_s(tmp_path):
    document = load_document(tmp_path / "truth.json")
    record(document, "a/overview.tif", GEOMETRY, entry("01-ruling-ibex", 1.0))
    record(document, "a/overview.tif", GEOMETRY, entry("02-ruling-oryx", 2.0))

    record(document, "a/overview.tif", GEOMETRY, entry("02-ruling-oryx", 3.0))

    assert answered(document, "a/overview.tif", "01-ruling-ibex")["x"] == 1.0
    assert answered(document, "a/overview.tif", "02-ruling-oryx")["x"] == 3.0
    assert len(document["cases"][0]["positions"]) == 2


def test_another_overview_is_recorded_beside_the_first(tmp_path):
    document = load_document(tmp_path / "truth.json")
    record(document, "a/overview.tif", GEOMETRY, entry("01-ruling-ibex", 1.0))
    record(document, "b/overview.tif", {"width": 9}, entry("01-ruling-ibex", 5.0))

    assert [case["overview"] for case in document["cases"]] == [
        "a/overview.tif", "b/overview.tif"
    ]
    assert document["cases"][1]["width"] == 9


def test_what_was_saved_is_resumed(tmp_path):
    path = tmp_path / "truth.json"
    document = load_document(path)
    record(document, "a/overview.tif", GEOMETRY, entry("01-ruling-ibex", 1.0))
    save(document, path)

    resumed = load_document(path)

    assert answered(resumed, "a/overview.tif", "01-ruling-ibex")["x"] == 1.0


def test_a_clear_match_is_the_first_guess():
    """Found near the projection, and said to be the match's."""
    rng = np.random.default_rng(3)
    picture = rng.normal(size=(600, 600)).astype(np.float32)
    template = picture[250:290, 300:360].copy()

    point, source = first_guess(picture, template, projected=(310.0, 250.0))

    assert source == "matched"
    assert point == pytest.approx((330.0, 270.0))


def test_an_unclear_match_falls_back_to_the_projection_and_says_so():
    """A template found nowhere must not be offered as if it had been."""
    rng = np.random.default_rng(4)
    picture = rng.normal(size=(600, 600)).astype(np.float32)
    template = rng.normal(size=(40, 60)).astype(np.float32)

    point, source = first_guess(picture, template, projected=(310.0, 250.0))

    assert source == "projected"
    assert point == (310.0, 250.0)


@pytest.mark.parametrize(
    "reference, overview, expected",
    [(180.0, 180.0, False), (0.0, 180.0, True), (179.9998, 0.0, True),
     (90.0, 0.0, False), (None, None, False)],
)
def test_a_reference_at_the_other_scan_rotation_is_turned_about(reference, overview, expected):
    """Laid over the overview the wrong way round, it could never be lined up."""
    assert turned_about(reference, overview) is expected

