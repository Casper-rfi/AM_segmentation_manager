"""What the Train tab's Predictions view knows: the runs looked at, the images drawn, the masks made.

Holds no Qt. Which images are offered and which checkpoint a run is looked at with
are :mod:`~am_segmentation_manager.training_preview`'s to decide, and predicting is
:mod:`~am_segmentation_manager.operations.preview_process`'s; this keeps what they
gave. **Every mask is held in memory and nowhere else**, by checkpoint and image, so
a run ticked or an image drawn predicts only what is not already held, and all of it
goes with the window.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import SegmentationConfig
from ..display import (
    DEFAULT_ALPHA,
    DisplayImage,
    compose_overlay,
    prepare_display_image,
    resize_label_to,
)
from ..progress import Cancellation
from ..training_preview import (
    DEFAULT_PREVIEW_COUNT,
    MOST_PREVIEW_IMAGES,
    PREVIEW_FILTERS,
    PreviewImage,
    PreviewModel,
    offered_values,
    pick_images,
    preview_images,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from ..operations.preview_process import PreviewOutcome

logger = logging.getLogger(__name__)

#: The longest edge a picture is prepared at: the model's own resolution, so
#: zooming in shows every pixel it predicted.
PREVIEW_MAX_EDGE = 1536

#: How many masks are kept before the oldest go; each is a megabyte or two.
MOST_MASKS = 256

Predict = Callable[..., "PreviewOutcome"]


class PreviewController:
    """The Predictions view's state, and the predictions it asks for.

    Args:
        config: Names and colours the classes.
        predict: Predicts images with checkpoints; the child process unless a test says otherwise.

    Attributes:
        models: The runs looked at, a column each.
        images: The images drawn, a row each.
        chosen: What an image must be to be drawn, by column of the images file; "" is any.
        count: How many images are drawn.
        unseen_only: Whether an image one of the runs trained on is left out.
        alpha: How opaque the classes are drawn.
        label_shown: Whether the classes are drawn at all.
    """

    def __init__(self, config: SegmentationConfig, predict: Predict | None = None) -> None:
        self.config = config
        self._predict = predict
        self.models: list[PreviewModel] = []
        self.images: list[PreviewImage] = []
        self.chosen: dict[str, str] = {column: "" for _, column in PREVIEW_FILTERS}
        self.count: int = DEFAULT_PREVIEW_COUNT
        self.seed: int = 0
        self.unseen_only: bool = True
        self.alpha: float = DEFAULT_ALPHA
        self.label_shown: bool = True
        self._masks: dict[tuple[Path, Path], Any] = {}
        self._failures: dict[Path | tuple[Path, Path], str] = {}
        self._shown: dict[Path, DisplayImage | str] = {}

    # -- what is looked at ----------------------------------------------------

    def set_models(self, models: Sequence[PreviewModel]) -> bool:
        """Look at *models*, and return whether that changed what is looked at."""
        models = list(models)
        if self._keys(models) == self._keys(self.models):
            return False
        self.models = models
        return True

    @staticmethod
    def _keys(models: Sequence[PreviewModel]) -> list[tuple[str, Path | None, str]]:
        return [
            (model.name, model.checkpoint.path if model.checkpoint else None, model.which)
            for model in models
        ]

    def offered(self) -> list[PreviewImage]:
        """Return the images that can be drawn for the runs looked at."""
        return preview_images([model.run for model in self.models], self.unseen_only)

    def values(self, column: str) -> tuple[str, ...]:
        """Return the values the images offered hold in *column*."""
        return offered_values(self.offered(), column)

    def describe_offered(self) -> str:
        """Return how many images there are to draw from, or why there are none."""
        if not self.models:
            return "Tick the runs to compare."
        offered = self.offered()
        runs = f"the {len(self.models)} run(s) ticked"
        if offered:
            seen = "none of them trained on" if self.unseen_only else "some may have been trained on"
            return f"{len(offered)} validation image(s) of {runs}; {seen}."
        if not any(model.run.validation for model in self.models):
            return f"No validation list could be read for {runs}."
        return (
            f"Every validation image of {runs} was trained on by one of them. Switch "
            f"Unseen only off to draw them anyway."
        )

    def pick(self, again: bool = True) -> list[PreviewImage]:
        """Draw the images to look at, a new draw unless *again* is False, and return them."""
        if again:
            self.seed += 1
        count = max(1, min(self.count, MOST_PREVIEW_IMAGES))
        self.images = pick_images(self.offered(), self.chosen, count, self.seed)
        return self.images

    # -- predicting -----------------------------------------------------------

    def missing(self) -> tuple[list[Path], list[Path]]:
        """Return the images and the checkpoints with a prediction still to make between them."""
        weights = [model.checkpoint.path for model in self.models if model.checkpoint is not None]
        wanted = [
            (model, image.image) for model in weights for image in self.images
            if (model, image.image) not in self._masks
            and (model, image.image) not in self._failures and model not in self._failures
        ]
        return (
            [image.image for image in self.images if any(image.image == i for _, i in wanted)],
            [model for model in weights if any(model == m for m, _ in wanted)],
        )

    def predict(
        self,
        device: str = "auto",
        on_progress: Callable[[int, int], None] | None = None,
        cancellation: Cancellation | None = None,
    ) -> str:
        """Make the predictions still missing, keep them, and return what went wrong, or "".

        Blocking; nothing is written anywhere.
        """
        images, weights = self.missing()
        if not images or not weights:
            return ""
        predict = self._predict
        if predict is None:
            from ..operations.preview_process import preview_out_of_process as predict
        outcome = predict(images, weights, device, on_progress=on_progress, cancellation=cancellation)
        self._masks.update(outcome.masks)
        self._failures.update(outcome.failures)
        while len(self._masks) > MOST_MASKS:
            del self._masks[next(iter(self._masks))]
        return outcome.error

    def forget_failures(self) -> None:
        """Let what could not be predicted be tried again."""
        self._failures.clear()

    # -- drawing --------------------------------------------------------------

    def shown(self, image: PreviewImage) -> DisplayImage | str:
        """Return *image* ready to draw with its own label, or why it cannot be opened."""
        if image.image not in self._shown:
            try:
                self._shown[image.image] = prepare_display_image(
                    image.image, image.label, max_edge=PREVIEW_MAX_EDGE
                )
            except (OSError, RuntimeError) as error:
                logger.warning("Could not open %s: %s", image.image, error)
                self._shown[image.image] = f"Could not open the image: {error}"
        return self._shown[image.image]

    def note(self, model: PreviewModel, image: PreviewImage) -> str:
        """Return what there is to say of *model*'s tile for *image* beside its picture, or ""."""
        if model.checkpoint is None:
            return model.which
        key = (model.checkpoint.path, image.image)
        if key in self._masks:
            return "trained on this image" if model.run.trained_on(image) else ""
        return self._failures.get(key) or self._failures.get(key[0]) or "not predicted yet"

    def picture(
        self, model: PreviewModel, image: PreviewImage, proofread: bool = False
    ) -> np.ndarray | None:
        """Return *model*'s prediction drawn over *image*, or None where there is no picture.

        The image alone until it is predicted. The mask is resized to the
        picture by nearest neighbour, as a prediction converted to a label is.

        Args:
            proofread: Draw the image's own label in place of the prediction.
        """
        shown = self.shown(image)
        if isinstance(shown, str):
            return None
        mask = shown.label if proofread else None
        if not proofread and model.checkpoint is not None:
            made = self._masks.get((model.checkpoint.path, image.image))
            mask = resize_label_to(made, shown.image.shape) if made is not None else None
        if mask is None or not self.label_shown:
            return shown.image
        return compose_overlay(shown.image, mask, self.config, alpha=self.alpha)
