# Third-party notices

Code in this package derived from someone else's work, and the licence it came
under. Listed whether or not that licence requires it, so that "what is not
ours" has one answer.

---

## fibsemOS (`fibsem`)

**Used by:** `src/am_segmentation_manager/overview.py`, whose `reproject` is a
reimplementation of `calculate_reprojected_stage_position2` in
`fibsem/imaging/tiling/reprojection.py`, and the `inverse_view_corrected_dy` in
`fibsem/transformations.py` it reaches, as fibsemOS's FIB-766 left them (first
released in fibsem 0.5.2rc1). The projection places a lamella's
recorded stage coordinates on a grid overview; fibsemOS's acquisition software
wrote both, so its arithmetic is the definition of where they fall.
`tests/test_overview.py` holds the reimplementation to fibsemOS's own answers on
recorded headers.

**Source:** https://github.com/fibsem-os/fibsem-os

**Licence:** MIT

```
MIT License

Copyright (c) 2025 fibsemOS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## AM_training

**Used by:** `src/am_segmentation_manager/ml/preprocessing.py`, a copy of that
project's module with its training-only transform removed and nothing else
changed, so that a `diff` against that source shows only what this package did
to it. It is excluded from `ruff` for the same reason.

**Licence:** not declared upstream. The copy is internal to the group that
wrote it; settle this before publishing, along with this package's own licence
(`docs/00_roadmap.md`, "Before publishing").
