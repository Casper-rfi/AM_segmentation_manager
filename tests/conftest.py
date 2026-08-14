"""Shared pytest configuration and dataset builders.

Import path setup used to live here and in a copy at the top of most test
files. It is now declared once as ``pythonpath = ["src"]`` in
``pyproject.toml``, so the tests run against the source tree whether or not the
package is installed.

What lives here instead is :func:`build_dataset`, which constructs the layouts
of ``docs/reference_layouts.md`` on disk. The layout variety is the risk this
package exists to manage, so a new layout era found in the wild should be a new
branch *here* rather than another hand-built tree in another test file.

**These builders spell out real-world paths literally, and must keep doing so.**
It would be tidier to ask :mod:`am_segmentation_manager.layout` where a label
directory goes, and it would be wrong: the tests would then agree with
``layout.py`` by construction, and a bug in the conventions would be invisible
to exactly the tests written to catch it. The paths below are transcribed from
real datasets and are the independent statement of what the conventions must
produce.

Migration is deliberately opportunistic. Roughly thirty test files still build
their own trees; they are correct and passing, and rewriting them all at once
would be a large diff with no behaviour change in the code that would tell us
if something broke. Use these builders for new tests, and convert an old file
when you are editing it anyway. See ``docs/01_refactor_plan.md`` section 5.1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

#: The layout eras seen in the real collection, newest first. Each is a branch
#: in :func:`build_dataset` and a row in ``docs/reference_layouts.md``.
ERAS: tuple[str, ...] = (
    "v2026_nested",
    "v2025_timestamped",
    "v2024_untimed",
    "legacy_variants",
    "legacy_collapsed",
)

#: Label directory naming schemes, oldest first. ``convention`` is what
#: ``migrate-layout`` produces; the other two are read indefinitely.
LABEL_NAMING: dict[str, tuple[str, str]] = {
    "legacy_upper": ("labels", "labels_FIB"),
    "legacy": ("labels", "labels_fib"),
    "convention": ("sem_labels", "fib_labels"),
}


@dataclass(frozen=True)
class Dataset:
    """Where everything landed, so tests do not re-derive paths.

    Attributes:
        root: The scan root, which is what you hand to ``Scanner``.
        project: The project directory, holding the marker file.
        lamella: The lamella directory. Equal to *project* where the era has no
            lamella level.
        stage: The stage directory. Equal to *lamella* where the era has no
            stage level.
        era: Which layout was built.
        image_dirs: Image directory path per ``(modality, variant)``.
        label_dirs: Label directory path per ``(modality, variant)``.
        filenames: The image filenames written per modality, in order. Kept
            per modality because the modern eras name a FIB image
            ``..._FIB.tif`` while the legacy eras use one name for both.
    """

    root: Path
    project: Path
    lamella: Path
    stage: Path
    era: str
    image_dirs: dict[tuple[str, str | None], Path] = field(default_factory=dict)
    label_dirs: dict[tuple[str, str | None], Path] = field(default_factory=dict)
    filenames: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def image_dir(self, modality: str = "SEM", variant: str | None = None) -> Path:
        """Return the image directory for a modality and imaging condition."""
        return self.image_dirs[(modality.upper(), variant)]

    def label_dir(self, modality: str = "SEM", variant: str | None = None) -> Path:
        """Return the label directory a modality's images resolve to."""
        return self.label_dirs[(modality.upper(), variant)]

    def image(self, modality: str = "SEM", index: int = 0, variant: str | None = None) -> Path:
        """Return one image file."""
        return self.image_dir(modality, variant) / self.filenames[modality.upper()][index]

    def label(self, modality: str = "SEM", index: int = 0, variant: str | None = None) -> Path:
        """Return the label file matching :meth:`image`."""
        return self.label_dir(modality, variant) / self.filenames[modality.upper()][index]


def write_marker(project: Path, kind: str = "experiment", *, name: str = "project",
                 method: str | None = "AutoLamella-OnGrid") -> None:
    """Write the file that makes *project* a project.

    ``experiment`` is Autolamella's; ``am_metadata`` is this package's, which is
    what makes a pre-Autolamella dataset indexable at all.
    """
    project.mkdir(parents=True, exist_ok=True)
    if kind == "experiment":
        body = f"created_at: 1753107713.652424\nname: {name}\n"
        if method is not None:
            body += f"method: {method}\n"
        # A nested 'method' that must not be mistaken for the project's own.
        body += "positions:\n  - name: nested\n    method: should-not-be-read\n"
        (project / "experiment.yaml").write_text(body, encoding="utf-8")
    elif kind == "am_metadata":
        (project / "AM_metadata.yaml").write_text(
            "metadata:\n  sample: HeLa\n", encoding="utf-8"
        )
    else:  # pragma: no cover - guards a typo in a caller
        raise ValueError(f"Unknown marker kind: {kind}")


def write_label(path: Path, content: str) -> Path:
    """Write a label file.

    ``present`` writes a placeholder byte file, which is all a test of paths or
    counts needs and costs no optional dependencies. ``annotated`` and ``empty``
    write real TIFFs, because telling those apart means reading pixels -- the
    distinction that matters, since the segmentation software writes an empty
    label for every image it is pointed at.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if content == "present":
        path.write_bytes(b"")
        return path

    tifffile = pytest.importorskip("tifffile")
    np = pytest.importorskip("numpy")
    array = np.zeros((16, 16), dtype=np.uint8)
    if content == "annotated":
        array[4:12, 4:12] = 2
    elif content != "empty":  # pragma: no cover - guards a typo in a caller
        raise ValueError(f"Unknown label content: {content}")
    tifffile.imwrite(str(path), array)
    return path


#: Per-era lamella name, stage path under the lamella, and Autolamella
#: ``method``. All transcribed from real datasets rather than invented: the
#: names appear in assertions and in ``docs/reference_layouts.md``, and a
#: fixture that looks like the data is easier to reason about than one that
#: does not. ``None`` for a stage means the era has no stage directory.
_ERA_SHAPE: dict[str, tuple[str, tuple[str, ...] | None, str | None]] = {
    "v2026_nested": (
        "01-neat-ape",
        ("Milling", "Polishing", "adaptive_polish_2026-07-09-11-23-46AM"),
        "AutoLamella-OnGrid",
    ),
    "v2025_timestamped": (
        "01-full-gannet", ("adaptive_polish_2025-04-15-02-48-15AM",), "AutoLamella-OnGrid",
    ),
    # The 2024-08 grid projects really were lift-out.
    "v2024_untimed": ("07-known-mutt", ("adaptive_polish",), "AutoLiftout"),
    "legacy_variants": ("lam_010", None, None),
    "legacy_collapsed": ("", None, None),
}


def _stage_for(project: Path, era: str, number: int) -> tuple[Path, Path]:
    """Return the ``(lamella, stage)`` directories for an era.

    Where an era has no lamella or stage level, the parent stands in -- which is
    what the scanner then has to synthesise in memory. Additional lamellae are
    suffixed, so the first keeps the real name its assertions use.
    """
    base_name, stage_parts, _ = _ERA_SHAPE[era]
    if not base_name:
        return project, project  # neither a lamella nor a stage directory
    name = base_name if number == 1 else f"{base_name}-{number:02d}"
    lamella = project / name
    if stage_parts is None:
        return lamella, lamella  # a lamella, but no stage directory
    return lamella, lamella.joinpath(*stage_parts)


#: Image directory names per era, keyed by modality. The legacy eras use
#: uppercase names and no unsuffixed SEM directory at all, which is why "the
#: base directory" is not a usable way to find the primary condition.
_IMAGE_DIRS: dict[str, dict[str, str]] = {
    "v2026_nested": {"SEM": "sem", "FIB": "fib"},
    "v2025_timestamped": {"SEM": "sem", "FIB": "fib"},
    "v2024_untimed": {"SEM": "sem", "FIB": "fib"},
    "legacy_variants": {"SEM": "SEM_8", "FIB": "FIB"},
    "legacy_collapsed": {"SEM": "SEM", "FIB": "FIB"},
}


def build_dataset(
    root: Path,
    *,
    era: str = "v2025_timestamped",
    name: str = "project",
    lamellae: int = 1,
    acquisitions: int = 2,
    modalities: tuple[str, ...] = ("SEM", "FIB"),
    sem_variants: tuple[str, ...] = (),
    labels: str = "present",
    label_naming: str | None = None,
    marker: str | None = None,
    plots: bool = False,
    predictions: bool = False,
    out_of_scope: bool = False,
) -> Dataset:
    """Build one project of the given layout era under *root*.

    Args:
        root: Scan root to build under.
        era: One of :data:`ERAS`.
        name: Project directory name.
        lamellae: How many lamellae to build. The returned :class:`Dataset`
            points at the first; the others exist to be scanned.
        acquisitions: Images per modality per lamella.
        modalities: Which modalities to write. Passing only ``("SEM",)``
            reproduces the pre-Autolamella datasets, which have no FIB at all.
        sem_variants: Extra SEM imaging conditions, e.g. ``("SEM_8_fi_1",)``.
            These share the modality's label directory, which is what makes
            them one scene.
        labels: ``"present"``, ``"annotated"``, ``"empty"`` or ``"none"``.
        label_naming: A key of :data:`LABEL_NAMING`. Defaults to the naming the
            era actually used.
        marker: ``"experiment"`` or ``"am_metadata"``. Defaults to what the era
            actually had -- the legacy eras predate ``experiment.yaml``.
        plots: Write a summary plot per acquisition.
        predictions: Write ``predictions/raw`` and ``predictions/clean``.
        out_of_scope: Write the alignment and rough-milling images that real
            Autolamella projects carry alongside the adaptive-polish stages.
            They are context only and must never be indexed as segmentation
            targets, so a test that they are ignored needs them present.

    Returns:
        Where everything landed.
    """
    if era not in ERAS:
        raise ValueError(f"Unknown layout era: {era}")

    legacy = era.startswith("legacy")
    marker = marker or ("am_metadata" if legacy else "experiment")
    label_naming = label_naming or ("legacy_upper" if legacy else "legacy")
    sem_label_name, fib_label_name = LABEL_NAMING[label_naming]

    project = root / name
    write_marker(project, marker, name=name, method=_ERA_SHAPE[era][2])

    # Legacy datasets name an acquisition once and share it across modalities;
    # the modern ones carry the modality in the filename.
    filenames = {
        modality: tuple(
            f"img_{index:03d}.tif" if legacy else f"img_{index:03d}_{modality}.tif"
            for index in range(acquisitions)
        )
        for modality in ("SEM", "FIB")
    }

    image_dirs: dict[tuple[str, str | None], Path] = {}
    label_dirs: dict[tuple[str, str | None], Path] = {}
    first: tuple[Path, Path] | None = None

    for number in range(1, lamellae + 1):
        lamella, stage = _stage_for(project, era, number)
        if first is None:
            first = (lamella, stage)

        for modality in modalities:
            base_name = _IMAGE_DIRS[era][modality]
            label_name = sem_label_name if modality == "SEM" else fib_label_name
            variants: tuple[str | None, ...] = (None,)
            if modality == "SEM" and sem_variants:
                variants = (None, *sem_variants)

            for variant in variants:
                directory = stage / (base_name if variant is None else variant)
                directory.mkdir(parents=True, exist_ok=True)
                for filename in filenames[modality]:
                    (directory / filename).write_bytes(b"")
                if first == (lamella, stage):
                    image_dirs[(modality, variant)] = directory
                    label_dirs[(modality, variant)] = stage / label_name

            if labels != "none":
                for filename in filenames[modality]:
                    write_label(stage / label_name / filename, labels)

        if out_of_scope:
            # Real 2026 projects carry these beside the milling stages.
            for path in (
                lamella / "Alignment" / "Fine 1-11-01-38" / "ref.tif",
                lamella / "Milling" / "Rough-Milling" / "ignored.tif",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")

        if plots:
            for index, _ in enumerate(filenames["SEM"]):
                path = stage / "plots" / f"img_{index:03d}_plot.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
        if predictions:
            for kind in ("raw", "clean"):
                for filename in filenames["SEM"]:
                    path = stage / "predictions" / kind / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"")

    assert first is not None, "lamellae must be at least 1"
    return Dataset(
        root=root,
        project=project,
        lamella=first[0],
        stage=first[1],
        era=era,
        image_dirs=image_dirs,
        label_dirs=label_dirs,
        filenames=filenames,
    )


@pytest.fixture(params=ERAS)
def era(request) -> str:
    """Run a test against every known layout era."""
    return request.param


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """A plain two-acquisition project in the most common layout."""
    return build_dataset(tmp_path)
