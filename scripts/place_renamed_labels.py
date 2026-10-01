#!/usr/bin/env python3
"""Put labels made from renamed images back beside the images they belong to.

A subset of one project's images was once copied out, renamed and segmented.
This pairs each renamed image with the original by what its header says the
acquisition is -- the instrument's GUID, or where the script-based stack
recorded writing it -- and copies the label into the label directory the
original's image directory resolves to, named after the original.

**This is a standalone script, not an ``am-seg`` command.** A handful of
datasets were segmented this way and they all belong to one person, which makes
a general command bloat rather than a feature. Adapt it if a layout does not fit.

Previews by default; pass ``--apply`` to copy. The whole batch is validated
before anything is written: a single unmatched image, altered original, empty
label or occupied destination stops the run with nothing copied. Only paths
under ``--project`` are ever written to, and an existing label is never
replaced.

    python scripts/place_renamed_labels.py \\
        --images  <source images> --labels <labels made from them> \\
        --project <project root holding the originals> \\
        [--compress] [--mark-corrected] [--apply]

Where the source is itself laid out as a project, ``--source-project <root>``
replaces ``--images``/``--labels`` and pairs up every image directory in it.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from am_segmentation_manager.families import families_for
from am_segmentation_manager.image_metadata import FIBSEM_TAG
from am_segmentation_manager.io.atomic import write_atomically
from am_segmentation_manager.io.image import read_array, require_image_stack, to_2d
from am_segmentation_manager.io.path import make_path_relative
from am_segmentation_manager.layout import (
    LEGACY_LABEL_DIRECTORIES,
    classify_image_directory,
    classify_loose_image,
    conflicting_label_directories,
    image_files,
    is_ignored_image_name,
    resolve_labels_directory,
    subdirectories_by_name,
)
from am_segmentation_manager.models import ImageRecord
from am_segmentation_manager.operations.label_io import save_label_image
from am_segmentation_manager.operations.label_groups import record_from_image_path
from am_segmentation_manager.operations.selection import mark_corrected_records

logger = logging.getLogger(__name__)

#: TIFF tag holding the FEI XML document, whose GUID identifies an acquisition
#: rather than describing it. Read here rather than through
#: :mod:`~am_segmentation_manager.image_metadata`, which reads the tags the
#: package needs for acquisition settings and has no use for an identity. The
#: fibsem tag it does read is the fallback -- see :func:`written_identity`.
GUID_TAG = 34683

_GUID = re.compile(r"<Guid>\s*\{?([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\}?\s*</Guid>")


class PlacementConflict(RuntimeError):
    """Every reason a placement cannot go ahead, collected rather than raised
    at the first one, so a run reports all of them at once.

    Attributes:
        conflicts: One human-readable line per problem.
    """

    def __init__(self, conflicts: Iterable[str]) -> None:
        self.conflicts = list(conflicts)
        super().__init__(f"{len(self.conflicts)} problem(s) stop this placement")


@dataclass(frozen=True)
class LabelPlacement:
    """One label, and where the original image's label belongs.

    Attributes:
        label: The label file to copy, named after the renamed image.
        renamed_image: The renamed copy the label was drawn on.
        original_image: The image in the project the renamed copy was made from.
        destination: Where the label is to be written, named after
            *original_image* so the two pair by filename.
        identity: What both images say the acquisition is.
    """

    label: Path
    renamed_image: Path
    original_image: Path
    destination: Path
    identity: str

    def describe(self, root: Path | None = None) -> str:
        """Return a one-line description for a preview or a log."""
        destination = (
            make_path_relative(self.destination, root) if root else str(self.destination)
        )
        return f"{self.label.name} -> {destination}"


# ---------------------------------------------------------------------------
# Reading headers and pixels
# ---------------------------------------------------------------------------

def acquisition_guid(path: Path) -> str | None:
    """Return the GUID the instrument wrote into *path*, or None if it has none.

    An unreadable file is a missing GUID rather than an error, so one bad file
    in a project is reported alongside everything else instead of stopping the
    scan.
    """
    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as handle:
            tag = handle.pages[0].tags.get(GUID_TAG)
            # Read inside the context: a tag's value is loaded lazily, and a
            # large one is still on the closed file by the time it is asked for.
            document = None if tag is None else str(tag.value)
    except Exception as exc:  # noqa: BLE001 - an unreadable file must not raise here
        logger.warning("Could not read the header of %s: %s", path, exc)
        return None
    if document is None:
        return None
    match = _GUID.search(document)
    return match.group(1).lower() if match else None


def written_identity(path: Path) -> str | None:
    """Return where the acquisition software recorded writing *path*.

    The fibsem header names the directory an image was written to, the filename
    it was given and when it was taken. A copy carries all three unchanged, so
    together they identify an acquisition for the images carrying no GUID.
    """
    import json

    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as handle:
            tag = handle.pages[0].tags.get(FIBSEM_TAG)
            document = None if tag is None else json.loads(str(tag.value))
    except Exception as exc:  # noqa: BLE001 - an unreadable file must not raise here
        logger.warning("Could not read the fibsem header of %s: %s", path, exc)
        return None
    if not isinstance(document, dict):
        return None

    image = document.get("image") or {}
    state = document.get("microscope_state") or {}
    written, name, taken = (
        image.get("path"), image.get("filename"), state.get("timestamp")
    )
    if not (written and name and taken):
        return None
    return f"{written}|{name}|{taken}"


def acquisition_identity(path: Path) -> str | None:
    """Return what identifies the acquisition in *path*, or None if nothing does.

    The instrument's GUID where the header carries one. The script-based stack
    writes no GUID at all, so its images fall back to what it did record: where
    it wrote them. Both come from the acquisition software, which is what keeps
    this evidence rather than a guess -- a filename alone would be the latter.
    """
    guid = acquisition_guid(path)
    if guid is not None:
        return f"guid:{guid}"
    written = written_identity(path)
    return None if written is None else f"written:{written}"


def pixel_digest(path: Path) -> tuple[tuple[int, ...], str] | None:
    """Return *path*'s shape and a digest of its pixels, or None if unreadable.

    The digest covers the dtype as well as the values, so two files agree only
    if they would decode identically.
    """
    try:
        array = to_2d(read_array(path))
    except Exception as exc:  # noqa: BLE001 - an unreadable file must not raise here
        logger.warning("Could not read %s: %s", path, exc)
        return None
    digest = hashlib.sha256(
        f"{array.shape}{array.dtype}".encode() + array.tobytes()
    ).hexdigest()
    return array.shape, digest


def is_empty(path: Path) -> bool | None:
    """Return whether the label at *path* holds nothing but background.

    Returns None when the file could not be read, which is deliberately
    distinct from True: an unreadable label must never pass as an empty one.
    """
    from am_segmentation_manager.display import is_empty_label

    try:
        return is_empty_label(to_2d(read_array(path)))
    except Exception as exc:  # noqa: BLE001 - an unreadable file must not raise here
        logger.warning("Could not read label %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def project_images(project_root: Path) -> list[Path]:
    """Return every image in *project_root* that could carry a label.

    Two shapes, because Adaptive Polishing images come in two. Usually they
    fill a ``<modality>`` directory, classified through
    :func:`~am_segmentation_manager.layout.classify_image_directory`, which
    skips label, prediction, plot and state directories. The script-based
    prototype instead wrote them loose in the lamella directory, where
    :func:`~am_segmentation_manager.layout.classify_loose_image` names them.

    Only the *adaptive* series is taken from a loose directory. It is the
    series labels are expected for, and leaving the rest out keeps the
    references Autolamella copies between lamellae from reaching the index:
    a copy carries the original's recorded identity, so eleven of them would
    read as one acquisition duplicated eleven times.
    """
    found: list[Path] = []
    for directory in sorted(path for path in project_root.rglob("*") if path.is_dir()):
        if classify_image_directory(directory.name) is not None:
            found.extend(
                path for path in image_files(directory)
                if not is_ignored_image_name(path.name)
            )
            continue
        found.extend(
            path for path in image_files(directory)
            if (classified := classify_loose_image(path.name)) is not None
            and classified.series == "adaptive"
        )
    return found


def index_by_identity(paths: Iterable[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Return images keyed by acquisition identity, and the identities shared.

    A duplicated identity is returned rather than resolved: which of two copies
    an original is cannot be decided here, and only a duplicate a label actually
    matches is a problem worth stopping for.
    """
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in paths:
        identity = acquisition_identity(path)
        if identity is None:
            logger.warning("Nothing identifies the acquisition in %s; it cannot be matched", path)
            continue
        if identity in index:
            duplicates.setdefault(identity, [index[identity]]).append(path)
            continue
        index[identity] = path
    return index, duplicates


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def source_pairs(source_root: Path) -> list[tuple[Path, Path]]:
    """Return each image directory under *source_root* and where its labels are.

    Resolution goes through
    :func:`~am_segmentation_manager.layout.resolve_labels_directory`, so a
    source laid out like a project pairs ``sem*`` with its labels and ``fib*``
    with theirs, under the current naming or the pre-convention ``labels`` and
    ``labels_FIB``. An image directory whose labels do not exist is skipped:
    a source holds only what was segmented.
    """
    pairs: list[tuple[Path, Path]] = []
    for directory in sorted(path for path in source_root.rglob("*") if path.is_dir()):
        classification = classify_image_directory(directory.name)
        if classification is None or not image_files(directory):
            continue
        siblings = subdirectories_by_name(directory.parent)
        labels, exists, _ = resolve_labels_directory(
            directory, classification.modality, siblings, corrections=families_for(directory)
        )
        if exists:
            pairs.append((directory, labels))
    return pairs


def plan_placement(
    pairs: Iterable[tuple[Path, Path]],
    project_root: Path,
    *,
    skip_empty: bool = False,
) -> list[LabelPlacement]:
    """Return the copies to make, having validated every one of them.

    Each source image must have a label beside it, must match exactly one image
    in the project by GUID, must be pixel-identical to it, and must carry a
    non-empty label of the same dimensions. The destination must be free, and
    no two labels may claim it.

    Args:
        pairs: Image directory and label directory, one pair per source image
            set. Validated together, so one bad pair stops the whole batch.
        project_root: The project holding the original images.
        skip_empty: Pass over a label holding no annotation instead of
            refusing the batch for it. Each one is named as it is skipped: a
            blank label is a picture nobody segmented, and which ones those
            are is worth seeing rather than counting.

    Raises:
        PlacementConflict: If anything at all is wrong, listing all of it.
        RuntimeError: If the optional image stack is not installed.
    """
    require_image_stack("Pairing labels with their original images")
    conflicts: list[str] = []

    sources = [
        (path, labels_directory)
        for images_directory, labels_directory in pairs
        for path in image_files(images_directory)
        if not is_ignored_image_name(path.name)
    ]
    if not sources:
        raise PlacementConflict(["No images found to place labels for"])

    originals, duplicates = index_by_identity(project_images(project_root))
    if not originals:
        raise PlacementConflict(
            [f"No image in {project_root} carries anything identifying its acquisition"]
        )

    placements: list[LabelPlacement] = []
    claimed: dict[Path, Path] = {}

    for image, labels_directory in sources:
        label = labels_directory / image.name
        if not label.is_file():
            conflicts.append(f"{image.name}: no label beside it in {labels_directory}")
            continue

        identity = acquisition_identity(image)
        if identity is None:
            conflicts.append(f"{image.name}: nothing in its header identifies the acquisition")
            continue
        if identity in duplicates:
            listed = ", ".join(str(path) for path in duplicates[identity])
            conflicts.append(f"{image.name}: {identity} matches several images: {listed}")
            continue
        original = originals.get(identity)
        if original is None:
            conflicts.append(f"{image.name}: {identity} matches no image in {project_root}")
            continue

        if skip_empty and is_empty(label) is True:
            # The label's own path, not the image's name: the filename repeats
            # across lamellae, so it does not say which picture was skipped.
            logger.info("Skipping %s: it holds no annotation", label)
            continue

        conflicts.extend(_pixel_conflicts(image, original, label))

        destination = _destination_for(original, conflicts)
        if destination is None:
            continue
        if destination.exists():
            conflicts.append(f"{image.name}: {destination} already holds a label")
            continue
        already = claimed.get(destination)
        if already is not None:
            # The same label reaching one destination twice is the shared case,
            # not a collision: several imaging conditions of one acquisition
            # share a label on both sides, and it is placed once. Two *different*
            # labels landing on one path is the error this guards against.
            if already != label:
                conflicts.append(
                    f"{image.name}: {destination} is already claimed by {already}"
                )
            continue
        claimed[destination] = label

        placements.append(
            LabelPlacement(
                label=label,
                renamed_image=image,
                original_image=original,
                destination=destination,
                identity=identity,
            )
        )

    if conflicts:
        raise PlacementConflict(conflicts)
    return sorted(placements, key=lambda placement: placement.destination)


def _pixel_conflicts(image: Path, original: Path, label: Path) -> list[str]:
    """Return what disagrees between a renamed image, its original and its label.

    The GUID says the two images are the same acquisition; this says the file
    is still the same file, which is what makes a label drawn on one valid on
    the other.
    """
    conflicts: list[str] = []
    renamed_pixels = pixel_digest(image)
    original_pixels = pixel_digest(original)
    label_pixels = pixel_digest(label)

    if renamed_pixels is None:
        conflicts.append(f"{image.name}: could not be read")
    if original_pixels is None:
        conflicts.append(f"{image.name}: its original {original} could not be read")
    if label_pixels is None:
        conflicts.append(f"{image.name}: its label {label} could not be read")
    if None in (renamed_pixels, original_pixels, label_pixels):
        return conflicts

    if renamed_pixels[1] != original_pixels[1]:
        conflicts.append(
            f"{image.name}: pixels differ from its original {original}, so the "
            "label was not drawn on that image"
        )
    if label_pixels[0] != renamed_pixels[0]:
        conflicts.append(
            f"{image.name}: label is {label_pixels[0]} but the image is "
            f"{renamed_pixels[0]}"
        )
    if is_empty(label) is not False:
        conflicts.append(f"{image.name}: its label holds no annotation")
    return conflicts


def _destination_for(original: Path, conflicts: list[str]) -> Path | None:
    """Return where *original*'s label belongs, or None having said why not.

    Resolution goes through
    :func:`~am_segmentation_manager.operations.label_groups.record_from_image_path`,
    so the label lands where the package will look for it, whether the image
    fills an image directory -- where its family's conditions share one -- or
    sits loose in its stage. A pre-convention label directory is refused rather
    than added to: renaming it afterwards would be a second migration over
    freshly placed data. So is a family with several label directories, which
    placing into one of them would only deepen.
    """
    record = record_from_image_path(original)
    if record.labels_directory is None or record.modality is None:
        conflicts.append(f"{original} is in no recognised image directory")
        return None
    if record.labels_directory.name.lower() in LEGACY_LABEL_DIRECTORIES:
        conflicts.append(
            f"{original}: labels resolve to the pre-convention "
            f"{record.labels_directory}; run scripts/rename_legacy_label_dirs.py first"
        )
        return None

    if classify_image_directory(original.parent.name) is not None:
        competing = conflicting_label_directories(
            original.parent,
            record.modality,
            subdirectories_by_name(original.parent.parent),
            corrections=families_for(original.parent),
        )
        if competing:
            names = ", ".join(directory.name for directory in competing)
            conflicts.append(
                f"{original}: its family has several label directories ({names}); "
                f"merge them first"
            )
            return None
    return record.labels_directory / original.name


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def apply_placement(placements: Iterable[LabelPlacement], *, compress: bool) -> int:
    """Copy each planned label, verifying the result, and return how many landed.

    Each write is atomic and each result is read back and compared with the
    source, since a copy that silently truncated would look like a placed
    label. Compressing re-encodes rather than copies bytes, so the comparison
    is of pixels either way.

    Raises:
        PlacementConflict: If a written label does not read back as its source.
            Labels already written stay; the run stops where it failed.
    """
    written = 0
    for placement in placements:
        source = pixel_digest(placement.label)
        if source is None:
            raise PlacementConflict([f"{placement.label} could not be read"])

        if compress:
            array = to_2d(read_array(placement.label))
            write_atomically(
                placement.destination,
                lambda temporary, array=array: save_label_image(
                    array, temporary, array.dtype
                ),
            )
        else:
            write_atomically(
                placement.destination,
                lambda temporary: shutil.copyfile(placement.label, temporary),
            )

        if pixel_digest(placement.destination) != source:
            raise PlacementConflict(
                [f"{placement.destination} does not read back as {placement.label}"]
            )
        written += 1
        logger.debug("Placed %s", placement.describe())
    return written


def records_for(placements: Iterable[LabelPlacement]) -> list[ImageRecord]:
    """Return a record per placed label, for declaring its workflow state."""
    return [record_from_image_path(placement.original_image) for placement in placements]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Pair labels made from renamed images with the originals they were "
            "made from, by the GUID in the TIFF header, and copy each label to "
            "where the original's label belongs."
        ),
        epilog=(
            "Previews by default. Nothing outside --project is ever written to, "
            "and an existing label is never replaced."
        ),
    )
    parser.add_argument(
        "--images", type=Path,
        help="Directory holding the source images the labels were drawn on",
    )
    parser.add_argument(
        "--labels", type=Path,
        help="Directory holding the labels, named after the source images",
    )
    parser.add_argument(
        "--source-project", type=Path,
        help=(
            "Source laid out as a project, instead of --images/--labels: every "
            "image directory in it is paired with its own label directory, and "
            "all of them are validated as one batch"
        ),
    )
    parser.add_argument(
        "--project", required=True, type=Path,
        help="Project root holding the original images; the only tree written to",
    )
    parser.add_argument(
        "--skip-empty", action="store_true",
        help=(
            "Pass over a label holding no annotation instead of refusing the "
            "batch for it; each one skipped is named"
        ),
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Re-encode each label with LZW while placing it, instead of copying bytes",
    )
    parser.add_argument(
        "--mark-corrected", action="store_true",
        help="Declare each placed label's group corrected once every copy has landed",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Perform the copies; without it the run only reports what it would do",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the placement, returning a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.source_project is not None:
        if args.images is not None or args.labels is not None:
            parser.error("--source-project replaces --images/--labels; pass one or the other")
        pairs = source_pairs(args.source_project)
        if not pairs:
            logger.error("No image directory with labels found in %s", args.source_project)
            return 2
    elif args.images is not None and args.labels is not None:
        pairs = [(args.images, args.labels)]
    else:
        parser.error("pass either --source-project or both --images and --labels")

    try:
        placements = plan_placement(
            pairs, args.project, skip_empty=args.skip_empty
        )
    except PlacementConflict as exc:
        for conflict in exc.conflicts:
            logger.error("%s", conflict)
        logger.error("Nothing was copied.")
        return 2
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    for placement in placements:
        logger.info(
            "%s %s", "Placing" if args.apply else "Would place",
            placement.describe(args.project),
        )
    if not args.apply:
        logger.info(
            "%d label(s) to place. Re-run with --apply to copy them.", len(placements)
        )
        return 0

    try:
        written = apply_placement(placements, compress=args.compress)
    except PlacementConflict as exc:
        for conflict in exc.conflicts:
            logger.error("%s", conflict)
        return 2
    logger.info("Placed %d label(s)", written)

    if args.mark_corrected:
        changed = mark_corrected_records(records_for(placements))
        logger.info("Declared %d image(s) corrected", changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
