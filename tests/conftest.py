"""Shared pytest configuration, dataset builders and window helpers.

Import path setup used to live here and in a copy at the top of most test
files. It is now declared once as ``pythonpath = ["src"]`` in
``pyproject.toml``, so the tests run against the source tree whether or not the
package is installed.

What lives here instead is :func:`build_dataset`, which constructs on disk the
real layouts this collection has been seen in. That variety is the risk this
package exists to manage, so a new layout found in the wild should be a new
branch *here* rather than another hand-built tree in another test file.

**These builders spell out real-world paths literally, and must keep doing so.**
It would be tidier to ask :mod:`am_segmentation_manager.layout` where a label
directory goes, and it would be wrong: the tests would then agree with
``layout.py`` by construction, and a bug in the conventions would be invisible
to exactly the tests written to catch it. The paths below are transcribed from
real datasets and are the independent statement of what the conventions must
produce.

:func:`settle` and :func:`destroy` are here for the same reason: every window
test needs both, and a per-file copy of either is how the suite grew a wait
that slept through its whole timeout and a window that was never freed.

Migration is deliberately opportunistic. Roughly thirty test files still build
their own trees; they are correct and passing, and rewriting them all at once
would be a large diff with no behaviour change in the code that would tell us
if something broke. Use these builders for new tests, and convert an old file
when you are editing it anyway. See ``docs/00_roadmap.md``.
"""
from __future__ import annotations

import importlib.util
import locale
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

#: Set this to a locale name to run the whole suite under it, e.g.
#: ``AM_SEG_TEST_LOCALE=de_DE.UTF-8 pytest -q``.
#:
#: Setting ``LC_ALL`` alone does **not** work, which is worth knowing before
#: trusting such a run: CPython starts in the "C" locale and only adopts the
#: environment when something calls ``setlocale``. Nothing in a plain test run
#: does -- so ``LC_ALL=de_DE.UTF-8 pytest`` passes while proving nothing.
#:
#: Constructing a ``QApplication`` *does* call ``setlocale(LC_ALL, "")``. That
#: is how a locale-dependent bug reached a user: the GUI tests run before
#: ``test_image_metadata`` alphabetically, so by the time it ran, Qt had
#: switched the process to the system locale and ``strptime('%p')`` no longer
#: matched "PM". Tests that care now set the locale explicitly themselves; this
#: hook is for sweeping the rest of the suite.
LOCALE_ENV_VAR = "AM_SEG_TEST_LOCALE"

#: The binding the window is written for; `docs/display.md` § The binding says
#: PyQt5 is no longer one. The suite states it here rather than leaving it to be
#: found out, because under the wrong binding it does not fail in a way that
#: names the binding: the widgets that compare pixels fail on a `sip.voidptr`,
#: about a dozen at once and spread over seven files, which reads as a set of
#: unrelated GUI regressions rather than as one wrong interpreter.
REQUIRED_QT_BINDING = "PySide6"

#: The environments this suite is meant to run in, the lighter one first.
#: `CLAUDE.md` names them too; they are here so the refusal below can turn them
#: into a command, which a sentence in a document cannot do.
SUPPORTED_ENVIRONMENTS = ("am_segmentation_manager", "am_segmentation_manager_ml")


def supported_interpreters() -> list[Path]:
    """Return the interpreters of the supported environments found beside this one.

    Located from ``sys.prefix`` rather than by asking conda: the refusal below
    is most often read in a shell where ``conda activate`` is not a function,
    which is the one shell where advice to activate something cannot be
    followed. Returns nothing on a machine laid out some other way, and the
    caller says what it can without them.
    """
    prefix = Path(sys.prefix)
    environments = prefix.parent if prefix.parent.name == "envs" else prefix / "envs"
    found = []
    for name in SUPPORTED_ENVIRONMENTS:
        for interpreter in (
            environments / name / "bin" / "python",
            environments / name / "python.exe",
        ):
            if interpreter.exists():
                found.append(interpreter)
                break
    return found


def installed_qt_binding() -> str | None:
    """Return the binding qtpy would use here, or None when qtpy is absent.

    Asked of qtpy rather than of the binding packages, because ``QT_API`` can
    select one other than the one that would otherwise be preferred; the answer
    has to be the binding the tests will actually run against. Importing qtpy
    imports ``QtCore``, which needs no platform plugin, but the offscreen
    platform is set first so this cannot be the one import that opens a display.
    """
    if importlib.util.find_spec("qtpy") is None:
        return None
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import qtpy

    return qtpy.API_NAME


def pytest_configure(config) -> None:
    """Refuse an unsupported Qt binding, then apply any requested locale.

    Both fail loudly rather than skipping, for the same reason: a run that
    quietly went ahead would report a result for a check that never happened.
    A missing qtpy is not one of these -- the curation environment without the
    window is a supported way to run, and the GUI tests skip themselves there.
    """
    binding = installed_qt_binding()
    if binding is not None and binding != REQUIRED_QT_BINDING:
        interpreters = supported_interpreters()
        remedy = (
            "Run instead:\n"
            + "\n".join(f"    {one} -m pytest" for one in interpreters)
            if interpreters
            else f"Run the suite in {' or '.join(SUPPORTED_ENVIRONMENTS)}."
        )
        raise pytest.UsageError(
            f"qtpy resolves to {binding} here, and the window is written for "
            f"{REQUIRED_QT_BINDING}. Running anyway fails about a dozen GUI "
            f"tests on differences between the bindings rather than on "
            f"anything wrong with the package -- conda's `base` is the usual "
            f"cause. See docs/decisions.md § Environment.\n\n{remedy}"
        )

    requested = os.environ.get(LOCALE_ENV_VAR)
    if not requested:
        return
    try:
        locale.setlocale(locale.LC_ALL, requested)
    except locale.Error as error:
        raise pytest.UsageError(
            f"{LOCALE_ENV_VAR}={requested!r} is not available on this machine "
            f"({error}). Install it, or choose one from `locale -a`."
        ) from error
    config.addinivalue_line("markers", "locale: run under a non-default locale")
    print(f"\nRunning under locale {locale.setlocale(locale.LC_TIME)}")

#: Sentinel for "this axis was not overridden", since None is a real value for
#: lamella_name and stage_parts (it means "this level is absent").
_KEEP = object()

#: Named sample layouts, drawn from real datasets.
#:
#: **These are representative examples, not a definitive taxonomy.** The real
#: variation is a set of independent axes -- whether there is a lamella level,
#: whether there is a stage level, how deeply the stage is nested, how image and
#: label directories are named, which modalities are present, whether there are
#: imaging-condition variants. The samples below are points in that space that
#: happen to have been observed; they are not its boundaries, and the next
#: dataset in the wild may well combine axes in a way none of them shows.
#:
#: So: a suite that passes across all of these does **not** demonstrate that
#: every layout is supported. It demonstrates that these samples work. When
#: testing a behaviour that depends on one axis, vary that axis directly with
#: the ``build_dataset`` arguments rather than reaching for the sample that
#: happens to exhibit it -- that way the test says which property it needs, and
#: keeps working when the samples are revised.
SAMPLE_LAYOUTS: tuple[str, ...] = (
    "nested_stage",
    "timestamped_stage",
    "untimed_stage",
    "no_stage_level",
    "no_lamella_level",
)

#: Kept so existing callers and any external references keep working.
ERAS = SAMPLE_LAYOUTS

#: Label directory naming schemes, oldest first. ``convention`` is what new
#: label directories are named; the other two are read indefinitely.
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


def settle(app, condition, *, timeout: float = 10.0, what: str = "the window") -> None:
    """Run the event loop until *condition* holds.

    Args:
        condition: What is being waited for. **Required**: a wait with nothing
            to wait for cannot tell the work finishing from the time running
            out, so it sleeps for the whole timeout and then asserts nothing.
        what: Named in the failure, so a timeout says which wait gave up.

    Raises:
        Failed: when *condition* has not come true within *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        app.processEvents()
        if condition():
            return
        if time.monotonic() >= deadline:
            pytest.fail(f"Timed out waiting for {what}")
        time.sleep(0.01)


def destroy(widget) -> None:
    """Close *widget* and delete the Qt object behind it.

    ``close()`` only hides a window: its widgets stay alive for the rest of the
    session, and ``QApplication.setStyleSheet`` and ``setPalette`` walk every
    live widget at a cost that grows faster than linearly. A window left behind
    therefore slows down the *later* files that apply the theme rather than the
    file that leaked it, which is why the cost shows up only in a full run.

    ``deleteLater()`` does not free one: the window is also held by the bound
    methods in its own key bindings, and Qt owns the other half of that cycle,
    so Python's collector cannot break it. Deleting through shiboken does.
    """
    import shiboken6

    widget.close()
    if shiboken6.isValid(widget):
        shiboken6.delete(widget)


def break_import(monkeypatch, module: str, error: Exception) -> None:
    """Make importing *module* raise, leaving every other import alone.

    How both optional stacks -- the image one and the inference one -- are
    tested for what they say when a dependency is missing or broken, without
    an environment that actually lacks it.
    """
    import builtins

    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == module:
            raise error
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)


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


def start_history(project: Path, *, kind: str = "experiment") -> Path:
    """Opt *project* into recording, making it a project first if it is not one.

    A log only counts where the directory is a project, so a test that touches
    ``AM_history.jsonl`` into a bare directory is testing a shape the package
    refuses. One helper, so that stays true everywhere at once.

    Returns:
        The log file.
    """
    from am_segmentation_manager.discovery import project_markers
    from am_segmentation_manager.history import history_file_path

    project.mkdir(parents=True, exist_ok=True)
    if not project_markers(project):
        write_marker(project, kind)
    path = history_file_path(project)
    path.touch()
    return path


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


@dataclass(frozen=True)
class LayoutAxes:
    """The independent properties that make one dataset layout differ.

    A sample layout is just a preset filling these in. Any of them can be
    overridden on a call to :func:`build_dataset`, so a test can build a
    combination no observed dataset happens to show -- which is the point, since
    the observed datasets are examples rather than an exhaustive classification.

    Attributes:
        lamella_name: Directory name for a lamella, or None where the layout has
            no lamella level and the project directory stands in for one.
        stage_parts: Path parts from the lamella down to the stage, or None
            where there is no stage level. A tuple longer than one models a
            stage nested under intermediate directories.
        image_dirs: Image directory name per modality.
        label_naming: A key of :data:`LABEL_NAMING`.
        marker: Which file marks the project -- ``experiment`` or
            ``am_metadata``.
        method: Autolamella's ``method`` value, or None where absent.
    """

    lamella_name: str | None
    stage_parts: tuple[str, ...] | None
    image_dirs: dict[str, str]
    label_naming: str
    marker: str
    method: str | None


#: The samples, as axis values. Names, stage timestamps and methods are
#: transcribed from real datasets so a fixture resembles the data -- but see
#: :data:`SAMPLE_LAYOUTS`: these are points observed, not a classification.
_SAMPLES: dict[str, LayoutAxes] = {
    "nested_stage": LayoutAxes(
        lamella_name="01-neat-ape",
        stage_parts=("Milling", "Polishing", "adaptive_polish_2026-07-09-11-23-46AM"),
        image_dirs={"SEM": "sem", "FIB": "fib"},
        label_naming="legacy", marker="experiment", method="AutoLamella-OnGrid",
    ),
    "timestamped_stage": LayoutAxes(
        lamella_name="01-full-gannet",
        stage_parts=("adaptive_polish_2025-04-15-02-48-15AM",),
        image_dirs={"SEM": "sem", "FIB": "fib"},
        label_naming="legacy", marker="experiment", method="AutoLamella-OnGrid",
    ),
    "untimed_stage": LayoutAxes(
        lamella_name="07-known-mutt",
        stage_parts=("adaptive_polish",),
        image_dirs={"SEM": "sem", "FIB": "fib"},
        # The 2024-08 grid projects really were lift-out.
        label_naming="legacy", marker="experiment", method="AutoLiftout",
    ),
    "no_stage_level": LayoutAxes(
        lamella_name="lam_010",
        stage_parts=None,
        image_dirs={"SEM": "SEM_8", "FIB": "FIB"},
        label_naming="legacy_upper", marker="am_metadata", method=None,
    ),
    "no_lamella_level": LayoutAxes(
        lamella_name=None,
        stage_parts=None,
        image_dirs={"SEM": "SEM", "FIB": "FIB"},
        label_naming="legacy_upper", marker="am_metadata", method=None,
    ),
}

#: Older names for the samples, so existing tests keep reading.
_ALIASES = {
    "v2026_nested": "nested_stage",
    "v2025_timestamped": "timestamped_stage",
    "v2024_untimed": "untimed_stage",
    "legacy_variants": "no_stage_level",
    "legacy_collapsed": "no_lamella_level",
}


def layout_axes(layout: str) -> LayoutAxes:
    """Return the axis values for a named sample layout."""
    resolved = _ALIASES.get(layout, layout)
    if resolved not in _SAMPLES:
        raise ValueError(f"Unknown layout era: {layout}")
    return _SAMPLES[resolved]


def _directories_for(project: Path, axes: LayoutAxes, number: int) -> tuple[Path, Path]:
    """Return the ``(lamella, stage)`` directories for a layout.

    Where a layout has no lamella or stage level, the parent stands in -- which
    is what the scanner then has to synthesise in memory. Additional lamellae
    are suffixed, so the first keeps the name its assertions use.
    """
    if axes.lamella_name is None:
        return project, project
    name = axes.lamella_name if number == 1 else f"{axes.lamella_name}-{number:02d}"
    lamella = project / name
    if axes.stage_parts is None:
        return lamella, lamella
    return lamella, lamella.joinpath(*axes.stage_parts)


def build_dataset(
    root: Path,
    *,
    layout: str = "timestamped_stage",
    era: str | None = None,
    name: str = "project",
    lamellae: int = 1,
    acquisitions: int = 2,
    modalities: tuple[str, ...] = ("SEM", "FIB"),
    sem_variants: tuple[str, ...] = (),
    labels: str = "present",
    label_naming: str | None = None,
    marker: str | None = None,
    lamella_name: str | None | object = _KEEP,
    stage_parts: tuple[str, ...] | None | object = _KEEP,
    image_dirs: dict[str, str] | None = None,
    plots: bool = False,
    predictions: bool = False,
    out_of_scope: bool = False,
) -> Dataset:
    """Build one project under *root*.

    Starts from a named sample layout and lets any axis be overridden, because
    the samples are examples rather than a classification -- see
    :data:`SAMPLE_LAYOUTS`. A test that needs "a stage nested two levels down"
    should say so with *stage_parts* rather than picking the sample that
    currently happens to nest.

    Args:
        root: Scan root to build under.
        layout: A name from :data:`SAMPLE_LAYOUTS` to start from.
        era: Deprecated alias for *layout*.
        lamella_name: Override the lamella directory name; ``None`` removes the
            lamella level so the project directory stands in for one.
        stage_parts: Override the path from lamella to stage; ``None`` removes
            the stage level. A longer tuple nests the stage more deeply.
        image_dirs: Override image directory names per modality.
        name: Project directory name.
        lamellae: How many lamellae to build. The returned :class:`Dataset`
            points at the first; the others exist to be scanned.
        acquisitions: Images per modality per lamella.
        modalities: Which modalities to write. Independent of *era*: whether a
            dataset has FIB is a property of the dataset, not of its layout.
            Two documented pre-Autolamella datasets have SEM only, but every
            lamella of the oldest dataset in the current collection has FIB.
        sem_variants: Extra SEM imaging conditions, e.g. ``("SEM_8_fi_1",)``.
            These share the modality's label directory, which is what makes
            them one scene. Independent of *layout*.
        labels: ``"present"``, ``"annotated"``, ``"empty"`` or ``"none"``.
        label_naming: A key of :data:`LABEL_NAMING`. Defaults to the sample's.
        marker: ``"experiment"`` or ``"am_metadata"``. Defaults to the
            sample's -- the oldest datasets predate ``experiment.yaml``.
        plots: Write a summary plot per acquisition.
        predictions: Write ``predictions/raw`` and ``predictions/clean``.
        out_of_scope: Write the alignment and rough-milling images that real
            Autolamella projects carry alongside the adaptive-polish stages,
            under names no rule recognises. They are context only and must
            never become segmentation targets -- they are indexed as ``other``,
            which has no labels -- so a test that they stay out of the curated
            series needs them present.

    Returns:
        Where everything landed.
    """
    axes = layout_axes(era if era is not None else layout)
    if lamella_name is not _KEEP:
        axes = replace(axes, lamella_name=lamella_name)
    if stage_parts is not _KEEP:
        axes = replace(axes, stage_parts=stage_parts)
    if image_dirs is not None:
        axes = replace(axes, image_dirs=image_dirs)
    if label_naming is not None:
        axes = replace(axes, label_naming=label_naming)
    if marker is not None:
        axes = replace(axes, marker=marker)

    sem_label_name, fib_label_name = LABEL_NAMING[axes.label_naming]
    # Filenames carry the modality only where the images themselves do. The
    # oldest datasets name an acquisition once and share it across modalities.
    shared_filenames = axes.marker == "am_metadata"

    project = root / name
    write_marker(project, axes.marker, name=name, method=axes.method)

    # Legacy datasets name an acquisition once and share it across modalities;
    # the modern ones carry the modality in the filename.
    filenames = {
        modality: tuple(
            f"img_{index:03d}.tif" if shared_filenames
            else f"img_{index:03d}_{modality}.tif"
            for index in range(acquisitions)
        )
        for modality in ("SEM", "FIB")
    }

    built_image_dirs: dict[tuple[str, str | None], Path] = {}
    label_dirs: dict[tuple[str, str | None], Path] = {}
    first: tuple[Path, Path] | None = None

    for number in range(1, lamellae + 1):
        lamella, stage = _directories_for(project, axes, number)
        if first is None:
            first = (lamella, stage)

        for modality in modalities:
            base_name = axes.image_dirs[modality]
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
                    built_image_dirs[(modality, variant)] = directory
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
        era=era if era is not None else layout,
        image_dirs=built_image_dirs,
        label_dirs=label_dirs,
        filenames=filenames,
    )


@pytest.fixture(params=SAMPLE_LAYOUTS)
def sample_layout(request) -> str:
    """Run a test against every sample layout.

    Useful for behaviour that should hold regardless of layout. It is *not* a
    claim of exhaustive coverage -- see :data:`SAMPLE_LAYOUTS`. Where a
    behaviour depends on a specific property, vary that property directly.
    """
    return request.param


@pytest.fixture(params=SAMPLE_LAYOUTS)
def era(request) -> str:
    """Deprecated alias for :func:`sample_layout`."""
    return request.param


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """A plain two-acquisition project in one of the commoner layouts."""
    return build_dataset(tmp_path)
