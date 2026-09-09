"""Loading a trained checkpoint and turning one image into a prediction.

A checkpoint holds weights and nothing else, so the class and channel counts are
read from its own tensors and the rest -- preprocessing and architecture -- comes
from a description file beside it. Nothing recoverable from the weights is ever
declared, and ``strict=True`` on every load is what makes the two agree or fail.
Why that division: ``docs/decisions.md`` § Inference.

Predictions come back at the model's own resolution, never resized to the image.
"""
from __future__ import annotations

import logging
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
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
class ModelDescription:
    """The settings a checkpoint does not record, and cannot be inferred.

    Defaults describe the Adaptive Milling models in use, so a checkpoint with
    no description file beside it still runs. Every field here would produce a
    visibly wrong prediction if it were wrong, which is what makes a default
    safe: a mistake shows itself at once.

    Which modality a model was trained on is deliberately **not** here. Nothing
    in a checkpoint says, and a name like ``AM_SEM_All_V01.pth`` is whatever
    whoever trained it chose to type. Recording it would be recording a claim
    this package cannot check, so the choice of which images to predict is the
    caller's alone -- see :mod:`..operations.predict`.

    Attributes:
        image_size: Longest edge the image is resized to before the forward
            pass, and therefore the resolution the prediction is written at.
        normalise_first: Whether normalisation runs before padding.
        pad: Whether the image is padded to a square before resizing.
        normalise_version: Which of the two normalisations in
            :mod:`.preprocessing` applies.
        resize_version: ``"pytorch"`` or ``"cv2"``.
        architecture: The segmentation-models-pytorch model name.
        encoder: The encoder name within that architecture.
        source: The description file this came from, or ``None`` for defaults.
    """

    image_size: int = 1536
    normalise_first: bool = True
    pad: bool = False
    normalise_version: int = 2
    resize_version: str = "pytorch"
    architecture: str = "FPN"
    encoder: str = "efficientnet-b6"
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
    """Return the description beside *weights*, or the defaults if there is none.

    Raises:
        ValueError: If the file names a field this package does not know, which
            is more likely a typo than a setting that will be ignored safely.
    """
    path = description_path(weights)
    data = read_yaml(path)
    if not data:
        logger.info("No description at %s; using the default model settings", path)
        return ModelDescription()

    known = {field.name for field in fields(ModelDescription)} - {"source"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"{path}: unknown setting(s) {', '.join(unknown)}. "
            f"Known settings are {', '.join(sorted(known))}"
        )
    return ModelDescription(**data, source=path)


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


@dataclass(frozen=True)
class LoadedModel:
    """A model on a device, with everything needed to describe what it produced.

    Attributes:
        model: The torch module, in evaluation mode.
        description: The settings it is run with.
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
        a device problem surfaces at the forward pass rather than inside the
        vendored preprocessing.

        Args:
            image: A 2D greyscale image array.

        Returns:
            A 2D ``uint8`` array of class indices.
        """
        import numpy as np
        import torch

        # Imported at the point of inference: reading a model's description
        # must not pay for the stack that runs one.
        from .preprocessing import image_preprocess

        tensor = image_preprocess(
            image.astype(np.float32),
            image_size=self.description.image_size,
            normalise_first=self.description.normalise_first,
            rgb=self.facts.rgb,
            pad=self.description.pad,
            normalise_version=self.description.normalise_version,
            resize_version=self.description.resize_version,
        )
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
            return output.argmax(dim=1).squeeze(0).to("cpu").numpy().astype(np.uint8)


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
            weights do not fit the architecture the description names.
        FileNotFoundError: If *weights* does not exist.
    """
    require_ml_stack("Loading a model")
    import segmentation_models_pytorch as smp
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

    # encoder_weights=None because the state dict overwrites them on the next
    # line. Asking for pretrained weights here downloads ~100 MB to discard it,
    # and fails on a machine with no route to the internet.
    model = smp.create_model(
        description.architecture,
        encoder_name=description.encoder,
        encoder_weights=None,
        in_channels=facts.in_channels,
        classes=facts.classes,
        activation=None,
    )
    try:
        # strict: the description says which architecture these weights are for,
        # and a mismatch must stop here. Loading what fits and leaving the rest
        # randomly initialised produces confident, meaningless labels.
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"{weights} does not fit {description.architecture}/{description.encoder}"
            f"{f' as described by {description.source}' if description.source else ''}: "
            f"{error}"
        ) from error

    model = model.to(device)
    model.eval()
    logger.info(
        "Loaded %s (%s, %d classes, %d channels) onto %s",
        weights.name, facts.run_name or "run unrecorded", facts.classes,
        facts.in_channels, device,
    )
    return LoadedModel(model, description, facts, weights, device)
