#!/usr/bin/env python3
"""Build ``data/AM_icon.icns`` by setting the mark in the macOS icon container.

**This is a standalone script, not an ``am-seg`` command.** It is run when the
mark itself changes, which has happened once; the icon it writes is committed,
so nobody needs this to install or use the package.

macOS draws no container for an application icon: every icon in the Dock
supplies its own, and they all supply the same one, which is why an icon that
does not have it reads as foreign there. The container is a superellipse
--- ``|u|**5 + |v|**5 <= 1`` --- occupying 824/1024 of the canvas, centred,
with the remaining margin left clear for the shadow the system casts. Those
numbers were measured off the system's own icons rather than guessed, and
`docs/display.md` says why the container is there at all.

The artwork is therefore supplied *without* a container: the mark alone on a
transparent ground, trimmed to its own bounds here and then given a set margin,
so that how much whitespace an export carries cannot change the icon's size. An
SVG is preferred and is rasterised through Qt, so every size is drawn at its own
resolution rather than resampled from one bitmap.

Nothing is deleted and nothing existing is overwritten: a run refuses an output
that is already there unless ``--overwrite`` says otherwise.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

#: Where the mark's own files live: the artwork this builds from, and the icon
#: it writes. The SVG is committed beside the icon so a rebuild needs nothing
#: from outside the repository.
DATA = Path(__file__).resolve().parent.parent / "src/am_segmentation_manager/data"

#: The container's side, as a fraction of the canvas. The rest is the margin
#: macOS reserves for the shadow it draws behind every Dock icon.
CONTAINER_FRACTION = 824 / 1024

#: Exponent of the superellipse the container is. Fitted against the masks of
#: Notes, Reminders and Maps, which agree to within a pixel on a 206px tile.
EXPONENT = 5.0

#: Linear supersampling factor for the container mask. The mask is the one
#: edge in the icon with no artwork behind it, so it is drawn well above the
#: output size and resampled down rather than antialiased approximately.
SUPERSAMPLE = 8

#: The ground the mark sits on. The mark is drawn for a white tile; a
#: transparent container would show the Dock through it.
GROUND = (255, 255, 255)

#: The mark's longer side, as a fraction of the container. Set here rather than
#: taken from the artwork's own whitespace, which differs between exports; the
#: value is the composition of the drawing that does carry a container, 688 of
#: its 824.65, so the mark is the size it was drawn to be.
MARK_FRACTION = 688 / 824.65

#: How far above the output size the mark is drawn before being trimmed and
#: fitted, so neither step costs sharpness.
OVERSAMPLE = 4

#: What ``iconutil`` expects to find in an ``.iconset`` folder. Each entry is
#: a point size and a scale, and the pixel size is their product.
REPRESENTATIONS = [
    (16, 1), (16, 2), (32, 1), (32, 2), (128, 1),
    (128, 2), (256, 1), (256, 2), (512, 1), (512, 2),
]


#: The Qt application the SVG is rendered through. Held for the process's
#: lifetime: a ``QApplication`` that goes out of scope takes the segfault with
#: it rather than raising, so it cannot be a local.
_APPLICATION = None


def _qt_application():
    """Return the process's Qt application, creating it once if need be."""
    global _APPLICATION
    from qtpy.QtWidgets import QApplication

    _APPLICATION = QApplication.instance() or _APPLICATION or QApplication([])
    return _APPLICATION


def container_mask(size: int) -> Image.Image:
    """Return the container's alpha mask at *size* pixels square.

    Drawn at :data:`SUPERSAMPLE` times *size* and resampled down, so the curve
    is antialiased by area rather than by an edge approximation.
    """
    fine = size * SUPERSAMPLE
    axis = (np.arange(fine) + 0.5) / fine * 2 - 1
    inside = np.abs(axis)[None, :] ** EXPONENT + np.abs(axis)[:, None] ** EXPONENT <= 1.0
    mask = Image.fromarray((inside * 255).astype(np.uint8), "L")
    return mask.resize((size, size), Image.LANCZOS)


def _load_raster(source: Path, size: int) -> Image.Image:
    """Return *source* as an RGBA image *size* pixels square.

    An SVG is rendered at *size* through Qt; anything else is read by Pillow
    and resampled, which is why the module docstring prefers the SVG.
    """
    if source.suffix.lower() != ".svg":
        return Image.open(source).convert("RGBA").resize((size, size), Image.LANCZOS)

    # Imported here, and not at module scope, so that the PNG path runs in an
    # environment without Qt.
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QImage, QPainter
    from qtpy.QtSvg import QSvgRenderer

    _qt_application()
    image = QImage(size, size, QImage.Format_RGBA8888)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        QSvgRenderer(str(source)).render(painter)
    finally:
        # Always: a QPainter outliving its paint device crashes the process
        # rather than reporting an error.
        painter.end()

    with tempfile.TemporaryDirectory() as work:
        rendered = Path(work) / "mark.png"
        image.save(str(rendered), "PNG")
        return Image.open(rendered).convert("RGBA")


def render(source: Path, canvas: int) -> Image.Image:
    """Return the icon at *canvas* pixels square: the mark, set in the container.

    The mark is trimmed to its own bounds and then given the margin below,
    rather than keeping whatever whitespace the export happened to carry --- two
    exports of one drawing differed by twice the margin, which would have shown
    as the icon changing size.

    Raises:
        ValueError: if *source* is blank, or fills its canvas. Artwork that
            carries a container but insets it is not caught, so the default
            source is the artwork known to be bare.
    """
    inner = round(canvas * CONTAINER_FRACTION)

    # Rendered well above the size it is drawn at, so trimming and fitting cost
    # no sharpness; an SVG is redrawn at this resolution rather than resampled.
    mark = _load_raster(source, inner * OVERSAMPLE)
    bounds = mark.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError(f"{source} is blank")
    if bounds == (0, 0, mark.width, mark.height):
        # Artwork that reaches every edge is a tile, not a mark: the drawing of
        # this logo exists in both forms, and setting the tile in the container
        # would shrink the mark inside a ground nobody can see against white.
        raise ValueError(
            f"{source} fills its canvas, so it carries a ground of its own; "
            "supply the mark alone, on a transparent ground"
        )
    mark = mark.crop(bounds)

    target = round(inner * MARK_FRACTION)
    scale = target / max(mark.size)
    mark = mark.resize(
        (max(1, round(mark.width * scale)), max(1, round(mark.height * scale))),
        Image.LANCZOS,
    )

    art = Image.new("RGBA", (inner, inner), (*GROUND, 255))
    art.alpha_composite(mark, ((inner - mark.width) // 2, (inner - mark.height) // 2))
    art.putalpha(container_mask(inner))

    icon = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    icon.alpha_composite(art, ((canvas - inner) // 2,) * 2)
    return icon


def build(source: Path, destination: Path, overwrite: bool = False) -> Path:
    """Write the ``.icns`` for *source* to *destination* and return its path.

    Raises:
        FileExistsError: if *destination* is there and *overwrite* is false.
        RuntimeError: if ``iconutil`` is unavailable or refuses the iconset.
    """
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass --overwrite to replace it")

    with tempfile.TemporaryDirectory() as work:
        iconset = Path(work) / "AM_icon.iconset"
        iconset.mkdir()
        for point, scale in REPRESENTATIONS:
            suffix = "" if scale == 1 else f"@{scale}x"
            name = f"icon_{point}x{point}{suffix}.png"
            render(source, point * scale).save(iconset / name)

        try:
            subprocess.run(
                ["iconutil", "--convert", "icns", str(iconset), "--output", str(destination)],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as error:  # pragma: no cover - not on macOS
            raise RuntimeError("iconutil is a macOS tool; run this on a Mac") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"iconutil refused the iconset: {error.stderr.decode()}") from error

    return destination


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and build the icon."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Supply the mark on a transparent ground, without a container.",
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=DATA / "AM_icon.svg",
        help="the mark, as SVG (preferred) or PNG (default: the committed artwork)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA / "AM_icon.icns",
        help="where to write the .icns (default: the packaged icon)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing .icns"
    )
    arguments = parser.parse_args(argv)

    if not arguments.source.exists():
        parser.error(f"no such file: {arguments.source}")

    try:
        written = build(arguments.source, arguments.output, arguments.overwrite)
    except (FileExistsError, RuntimeError, ValueError) as error:
        # Refusing to overwrite, and artwork that is the wrong shape, are both
        # things the caller is meant to act on rather than read a traceback of.
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
