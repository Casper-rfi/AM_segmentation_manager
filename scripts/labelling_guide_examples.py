#!/usr/bin/env python3
"""Make the labelling guide's before/after examples from a label's saved versions.

**A standalone script, not an ``am-seg`` command.** Proofread one label a step
at a time in a copy of a project, pressing *Save only* after each step, having
saved a first, near-untouched version as the baseline: every save keeps the
version it replaces under ``AM_archive/``, so version 0 is the baseline and
version N is the label after step N. Then::

    uv run python scripts/labelling_guide_examples.py <label.tif>
    uv run python scripts/labelling_guide_examples.py <label.tif> --version 1 --name false_positives_outside
    uv run python scripts/labelling_guide_examples.py <label.tif> --version 1 --name ... --crop 2154,456,384,256

The first lists the versions and how much each changed. The second crops the
change from version N-1 to N, round where most of it is unless ``--crop x,y,w,h``
says otherwise -- a crop narrower than the help's example is shown magnified,
so a thin edge wants a small one -- and writes it into ``src/am_segmentation_manager/data/labelling_guide/``
under the name a step's ``example`` gives, and a preview of both into the
temporary folder, to look at. Nothing in the project is written.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
EXAMPLES = REPOSITORY / "src" / "am_segmentation_manager" / "data" / "labelling_guide"

#: The smallest crop, in pixels: big enough to see what changed beside what did not.
MIN_CROP = (384, 256)
#: Room left round the change, in pixels.
MARGIN = 48


def versions(label: Path) -> list[tuple[str, Path]]:
    """Return the label's versions, oldest first, as ``(when saved, file)``.

    Each save names the version it replaced; the last is the file itself.
    """
    from am_segmentation_manager.history import containing_project, events_for, read_log

    project = containing_project(label)
    if project is None:
        raise SystemExit(f"{label} is in no project that keeps a history")
    target = label.resolve().relative_to(project.resolve()).as_posix()
    saved = [event for event in events_for(read_log(project), target)
             if event.kind == "label-saved"]
    # A version kept by one save was written by the save before it.
    found = [
        (saved[index - 1].at if index else "before the first save",
         project / event.data["archived_as"])
        for index, event in enumerate(saved) if event.data.get("archived_as")
    ]
    found.append((saved[-1].at if saved else "on disk", label))
    return found


def image_of(label: Path) -> Path:
    """Return the image *label* describes: the same name, in the folder it is named for."""
    folder = label.parent.name
    if not folder.endswith("_labels"):
        raise SystemExit(f"cannot tell which image {label} labels: pass --image")
    return label.parent.parent / folder.removesuffix("_labels") / label.name


def propose_crop(changed, shape) -> tuple[int, int, int, int]:
    """Return ``(x, y, w, h)`` round the largest change, at least :data:`MIN_CROP`."""
    import cv2
    import numpy as np

    count, _parts, stats, _ = cv2.connectedComponentsWithStats(
        changed.astype(np.uint8), connectivity=8
    )
    if count < 2:
        raise SystemExit("the two versions are the same")
    largest = max(range(1, count), key=lambda part: stats[part, cv2.CC_STAT_AREA])
    x, y, w, h = (int(value) for value in stats[largest, :4])
    width = max(MIN_CROP[0], w + 2 * MARGIN)
    height = max(MIN_CROP[1], h + 2 * MARGIN)
    left = min(max(0, x + w // 2 - width // 2), shape[1] - width)
    top = min(max(0, y + h // 2 - height // 2), shape[0] - height)
    return left, top, width, height


def main(argv: list[str] | None = None) -> int:
    """List a label's versions, or write one step's example from them."""
    arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    arguments.add_argument("label", type=Path, help="The label proofread step by step")
    arguments.add_argument("--version", type=int, help="Write the change into this version")
    arguments.add_argument("--name", help="The example's name, as a step's example gives it")
    arguments.add_argument("--crop", help="x,y,w,h of the crop, in pixels of the frame")
    arguments.add_argument("--image", type=Path, help="The image, where it cannot be told")
    options = arguments.parse_args(argv)

    import cv2
    import numpy as np

    from am_segmentation_manager.config import default_config_path, load_segmentation_config
    from am_segmentation_manager.display import compose_overlay, prepare_display_image
    from am_segmentation_manager.gui.widgets.labelling_guide_panel import magnification
    from am_segmentation_manager.io.image import read_array

    found = versions(options.label)
    labels = [read_array(path) for _when, path in found]
    if options.version is None:
        for number, ((when, path), label) in enumerate(zip(found, labels)):
            changed = "" if number == 0 else (
                f"  {int((labels[number - 1] != label).sum())} pixel(s) changed"
            )
            print(f"{number}  {when}  {path.name}{changed}")
        return 0
    if not options.name or not 0 < options.version < len(found):
        arguments.error(f"--version 1 to {len(found) - 1} and --name are needed to write one")

    before, after = labels[options.version - 1], labels[options.version]
    image = prepare_display_image(
        options.image or image_of(options.label), None, max_edge=0
    ).image
    changed = before != after
    x, y, w, h = (
        tuple(int(value) for value in options.crop.split(","))
        if options.crop else propose_crop(changed, image.shape)
    )
    crop = (slice(y, y + h), slice(x, x + w))
    folder = EXAMPLES / options.name
    folder.mkdir(parents=True, exist_ok=True)
    for name, array in (("image", image), ("before", before), ("after", after)):
        if not cv2.imwrite(str(folder / f"{name}.png"), np.ascontiguousarray(array[crop])):
            raise OSError(f"could not write {folder / name}.png")

    config = load_segmentation_config(default_config_path())
    tiles = [compose_overlay(image[crop], label[crop], config) for label in (before, after)]
    # Magnified as the help magnifies it, so the preview is what will be seen.
    factor = magnification(w)
    tiles = [cv2.resize(tile, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)
             for tile in tiles]
    gap = np.full((h * factor, 12, 3), 255, np.uint8)
    side = np.concatenate([tiles[0], gap, tiles[1]], axis=1)
    preview = Path(tempfile.gettempdir()) / f"{options.name}_preview.png"
    cv2.imwrite(str(preview), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    print(f"Wrote {folder.relative_to(REPOSITORY)} from crop {x},{y},{w},{h}; "
          f"{int(changed[crop].sum())} of {int(changed.sum())} changed pixel(s) are in it")
    print(f"Preview: {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
