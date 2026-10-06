"""Loading a trained checkpoint and turning one image into a prediction.

A checkpoint holds weights and nothing else. How it is run -- preprocessing and
architecture -- is fixed by its generation, which a description file beside it
names and which defaults to generation 1; the class and channel counts are read
from its own tensors and must be the generation's. ``strict=True`` on every load
is what makes the weights and the architecture agree or fail. Why that division:
``docs/decisions.md`` § Inference.

Predictions come back at the model's own resolution, never resized to the image.
"""
from __future__ import annotations

import logging
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..io.yaml import read_yaml
from .environment import require_ml_stack

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

#: Suffix of the description file, appended to the weights file's own name:
#: ``AM_SEM_All_V01.pth`` is described by ``AM_SEM_All_V01.pth.yaml``. Appended
#: rather than substituted so that two checkpoints differing only in extension
#: cannot share one description.
DESCRIPTION_SUFFIX = ".yaml"

#: Where in a checkpoint the class count and input channel count are read from.
#: Both are the head and stem of every model here; a checkpoint missing them is
#: not one this package can run.
_HEAD_KEY = "segmentation_head.0.weight"
_STEM_KEYS = ("encoder._conv_stem.weight", "encoder.conv_stem.weight")


@dataclass(frozen=True)
class Generation:
    """How every model of one generation is run: what its checkpoints cannot record.

    A generation is frozen once predictions have been made with it; a change to
    how models are run is a new generation, so that every prediction on disk
    stays reproducible by the generation it records.

    Attributes:
        name: What a description file and a prediction's provenance call it.
        image_size: Longest edge the image is resized to before the forward
            pass, and therefore the resolution the prediction is written at.
        normalise_first: Whether normalisation runs before padding, which
            only matters to a generation that pads.
        pad: Whether the image is padded to a square before resizing.
        normalise_version: ``2`` clips the normalised image to [-1, 1].
        resize_version: ``"pytorch"`` resizes with torch's bicubic interpolation.
        architecture: The segmentation-models-pytorch model name.
        encoder: The encoder name within that architecture.
        classes: Segmentation classes the head must predict.
        in_channels: Input channels the stem must expect.
    """

    name: str
    image_size: int
    normalise_first: bool
    pad: bool
    normalise_version: int
    resize_version: str
    architecture: str
    encoder: str
    classes: int
    in_channels: int


#: Adaptive Milling's generation 1 -- ``Gen1Model`` in adaptive-milling 0.4.0,
#: ``1.4fpn`` before that -- which the microscope predicts with.
GENERATION_1 = Generation(
    name="1.0",
    image_size=1536,
    normalise_first=True,
    pad=False,
    normalise_version=2,
    resize_version="pytorch",
    architecture="FPN",
    encoder="efficientnet-b6",
    classes=5,
    in_channels=3,
)

#: The preprocessing settings :mod:`.preprocessing` implements, which are
#: generation 1's. A generation asking for others brings their implementation,
#: and a fixture to hold it to.
IMPLEMENTED_PREPROCESSING: Mapping[str, object] = {
    "pad": False,
    "normalise_version": 2,
    "resize_version": "pytorch",
}

#: Every generation this package can run, by name.
GENERATIONS: Mapping[str, Generation] = {GENERATION_1.name: GENERATION_1}

#: The generation of a checkpoint whose description does not name one. Never
#: the latest: a checkpoint left undescribed must not change how it is run
#: when a generation is added.
DEFAULT_GENERATION = GENERATION_1.name


@dataclass(frozen=True)
class ModelDescription:
    """What the description file beside a checkpoint declares.

    Which modality a model was trained on is deliberately **not** here. Nothing
    in a checkpoint says, and a name like ``AM_SEM_All_V01.pth`` is whatever
    whoever trained it chose to type. Recording it would be recording a claim
    this package cannot check, so the choice of which images to predict is the
    caller's alone -- see :mod:`..operations.predict`.

    Attributes:
        generation: How the checkpoint is run.
        source: The description file this came from, or ``None`` for defaults.
    """

    generation: Generation = GENERATION_1
    source: Path | None = None


@dataclass(frozen=True)
class CheckpointFacts:
    """What a checkpoint's own tensors state about the model it holds.

    Attributes:
        run_name: The training run that produced it, read from the archive
            rather than the filename -- ``AM_SEM_All_V01.pth`` does not say
            which run it was, and the archive inside it does.
        classes: Number of segmentation classes the head predicts.
        in_channels: Input channels the stem expects; 3 means the image is
            expanded to RGB before the forward pass, 1 that it stays greyscale.
        tensors: How many entries the state dict holds, for reporting.
    """

    run_name: str | None
    classes: int
    in_channels: int
    tensors: int

    @property
    def rgb(self) -> bool:
        """Return whether the image must be expanded to three channels."""
        return self.in_channels == 3


def description_path(weights: Path) -> Path:
    """Return where *weights*' description file would be."""
    return weights.with_name(weights.name + DESCRIPTION_SUFFIX)


def read_model_description(weights: Path) -> ModelDescription:
    """Return the description beside *weights*, or generation 1 if there is none.

    Raises:
        ValueError: If the file names anything but ``generation`` -- a typo, or
            a setting a generation fixes and a file therefore cannot change --
            or a generation this package does not know.
    """
    path = description_path(weights)
    data = read_yaml(path)
    if not data:
        logger.info("No description at %s; running it as generation %s", path, DEFAULT_GENERATION)
        return ModelDescription()

    unknown = sorted(set(data) - {"generation"})
    if unknown:
        raise ValueError(
            f"{path}: unknown setting(s) {', '.join(unknown)}. A description names "
            f"only the model's generation, which fixes everything else"
        )
    # YAML reads `generation: 1.0` as a number; str() gives back what was typed.
    name = str(data.get("generation", DEFAULT_GENERATION))
    if name not in GENERATIONS:
        raise ValueError(
            f"{path}: unknown generation {name!r}. Known generations are "
            f"{', '.join(repr(known) for known in GENERATIONS)}"
        )
    return ModelDescription(generation=GENERATIONS[name], source=path)


def run_name_of(weights: Path) -> str | None:
    """Return the training run recorded inside a checkpoint, if it has one.

    ``torch.save`` writes a zip whose single top-level directory is named after
    the object that was saved, which for these checkpoints is the training run.
    Read with :mod:`zipfile` rather than torch, so provenance costs nothing and
    works without the inference stack installed.
    """
    try:
        with zipfile.ZipFile(weights) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile):
        # A legacy (non-zip) checkpoint, or an unreadable one. The load itself
        # will report the real problem; provenance is not where to raise it.
        return None
    roots = {name.split("/")[0] for name in names if "/" in name}
    return roots.pop() if len(roots) == 1 else None


def checkpoint_facts(state: Mapping[str, Any], run_name: str | None = None) -> CheckpointFacts:
    """Return what a state dict states about its model.

    Takes the mapping rather than a path so it can be checked against a handful
    of tensor shapes in a test, without a 170 MB file or torch.

    Raises:
        ValueError: If the head or the stem is missing, which means the
            checkpoint is not a segmentation model of the expected shape.
    """
    if _HEAD_KEY not in state:
        raise ValueError(
            f"Not a segmentation checkpoint: no {_HEAD_KEY!r} among its "
            f"{len(state)} entries"
        )
    stem = next((key for key in _STEM_KEYS if key in state), None)
    if stem is None:
        raise ValueError(
            f"Cannot read the input channels: none of {', '.join(_STEM_KEYS)} is present"
        )
    return CheckpointFacts(
        run_name=run_name,
        classes=int(state[_HEAD_KEY].shape[0]),
        in_channels=int(state[stem].shape[1]),
        tensors=len(state),
    )


def check_generation(facts: CheckpointFacts, generation: Generation) -> None:
    """Refuse a checkpoint whose class or channel count is not *generation*'s.

    Raises:
        RuntimeError: Naming both counts, rather than the state-dict error the
            strict load would give.
    """
    if (facts.classes, facts.in_channels) != (generation.classes, generation.in_channels):
        raise RuntimeError(
            f"Not a generation-{generation.name} checkpoint: it predicts "
            f"{facts.classes} classes from {facts.in_channels} channel(s), and "
            f"generation {generation.name} predicts {generation.classes} from "
            f"{generation.in_channels}"
        )


@dataclass(frozen=True)
class LoadedModel:
    """A model on a device, with everything needed to describe what it produced.

    Attributes:
        model: The torch module, in evaluation mode.
        description: What its description file declares, generation included.
        facts: What its weights state about it.
        weights: Path the weights were read from.
        device: The torch device it sits on.
    """

    model: Any
    description: ModelDescription
    facts: CheckpointFacts
    weights: Path
    device: str

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return the predicted class of every pixel, at the model's resolution.

        The result is not resized back to *image*: predictions are stored at
        model resolution, as the segmentation pipeline's own already are, and
        conversion to a label is what matches them to an image.

        Preprocessing runs on the CPU and the result is moved to the device, so
        a device problem surfaces at the forward pass rather than inside
        preprocessing.

        Args:
            image: A 2D greyscale image array.

        Returns:
            A 2D ``uint8`` array of class indices.
        """
        import numpy as np
        import torch

        tensor = model_input(image, self.description.generation, rgb=self.facts.rgb)
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
            return output.argmax(dim=1).squeeze(0).to("cpu").numpy().astype(np.uint8)


def model_input(image: np.ndarray, generation: Generation, rgb: bool) -> Any:
    """Return *image* as the batched tensor *generation* feeds its model, on the CPU.

    Held to Adaptive Milling's own preprocessing by
    ``test_generation_one_preprocesses_as_adaptive_milling_does``.

    Raises:
        ValueError: If *generation* asks for preprocessing other than
            :data:`IMPLEMENTED_PREPROCESSING`.
    """
    unimplemented = {
        setting: getattr(generation, setting)
        for setting, implemented in IMPLEMENTED_PREPROCESSING.items()
        if getattr(generation, setting) != implemented
    }
    if unimplemented:
        raise ValueError(
            f"Generation {generation.name!r} asks for preprocessing this package "
            f"does not implement: {unimplemented}"
        )

    # Imported at the point of inference: reading a model's description must
    # not pay for the stack that runs one.
    from .preprocessing import preprocess

    return preprocess(image, image_size=generation.image_size, rgb=rgb)


def build_architecture(generation: Generation) -> Any:
    """Return *generation*'s model, its weights not yet loaded.

    Held to the one Adaptive Milling builds by
    ``test_generation_one_builds_adaptive_milling_s_architecture``.
    """
    import segmentation_models_pytorch as smp

    # encoder_weights=None because a state dict overwrites them. Asking for
    # pretrained weights here downloads ~100 MB to discard it, and fails on a
    # machine with no route to the internet.
    return smp.create_model(
        generation.architecture,
        encoder_name=generation.encoder,
        encoder_weights=None,
        in_channels=generation.in_channels,
        classes=generation.classes,
        activation=None,
    )


def load_model(
    weights: Path,
    description: ModelDescription | None = None,
    device: str = "cpu",
) -> LoadedModel:
    """Load *weights* onto *device*, ready to predict.

    Args:
        weights: The checkpoint file.
        description: How to run it; read from beside *weights* when omitted.
        device: A torch device string.

    Returns:
        The model in evaluation mode, with the facts read from its weights.

    Raises:
        RuntimeError: If the inference stack is not installed, or if the
            weights are not of the generation the description names.
        FileNotFoundError: If *weights* does not exist.
    """
    require_ml_stack("Loading a model")
    import torch

    if not weights.exists():
        raise FileNotFoundError(f"Model weights do not exist: {weights}")
    if description is None:
        description = read_model_description(weights)

    # map_location is not optional: these checkpoints were saved from a CUDA
    # device and record it, so loading without it tries to allocate on a GPU
    # that this machine may not have.
    state = torch.load(weights, map_location="cpu")
    facts = checkpoint_facts(state, run_name=run_name_of(weights))
    generation = description.generation
    check_generation(facts, generation)

    model = build_architecture(generation)
    try:
        # strict: the description says which architecture these weights are for,
        # and a mismatch must stop here. Loading what fits and leaving the rest
        # randomly initialised produces confident, meaningless labels.
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"{weights} does not fit generation {generation.name}'s "
            f"{generation.architecture}/{generation.encoder}"
            f"{f' as described by {description.source}' if description.source else ''}: "
            f"{error}"
        ) from error

    model = model.to(device)
    model.eval()
    logger.info(
        "Loaded %s (%s, generation %s, %d classes, %d channels) onto %s",
        weights.name, facts.run_name or "run unrecorded", generation.name,
        facts.classes, facts.in_channels, device,
    )
    return LoadedModel(model, description, facts, weights, device)
