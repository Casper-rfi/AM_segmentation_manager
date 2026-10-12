"""What the Train tab's Predictions view knows: the runs looked at, the images drawn, the masks made.

Holds no Qt. Which images are offered and which checkpoint a run is looked at with
are :mod:`~am_segmentation_manager.training_preview`'s to decide, and predicting is
:mod:`~am_segmentation_manager.operations.preview_process`'s; this keeps what they
gave. **Every mask is held in memory and nowhere else**, by checkpoint and image, so
a run ticked or an image drawn predicts only what is not already held, and all of it
goes with the window. **A picture is the size of its image**, the mask resized to it;
only the images drawn are held at that size.
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
    describe_not_found,
    offered_values,
    pick_images,
    preview_images,
    split_found,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from ..operations.preview_process import PreviewOutcome

logger = logging.getLogger(__name__)

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
        images: The images drawn, a row each: those of the last draw whose file is there.
        not_found: Those of the last draw whose file is not.
        hidden: Those of :attr:`images` left out of what is shown, to give the rest the room.
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
        self.not_found: list[PreviewImage] = []
        self.hidden: set[Path] = set()
        self.chosen: dict[str, str] = {column: "" for _, column in PREVIEW_FILTERS}
        self.count: int = DEFAULT_PREVIEW_COUNT
        self.seed: int = 0
        self.unseen_only: bool = True
        self.alpha: float = DEFAULT_ALPHA
        self.label_shown: bool = True
        self._masks: dict[tuple[Path, Path], Any] = {}
        self._failures: dict[Path | tuple[Path, Path], str] = {}
        self._shown: dict[Path, DisplayImage | str] = {}
        #: The classes drawn over an image at :attr:`_drawn_alpha`, by the weights that
        #: predicted them and the image; by None and the image for its own label.
        self._drawn: dict[tuple[Path | None, Path], Any] = {}
        self._drawn_alpha: float = self.alpha

    # -- what is looked at ----------------------------------------------------

    def set_models(self, models: Sequence[PreviewModel]) -> bool:
        """Look at *models*, and return whether that changed what is looked at."""
        models = list(models)
        if self._keys(models) == self._keys(self.models):
            return False
        self.models = models
        self.keep_offered()
        return True

    def keep_offered(self) -> None:
        """Keep, of the images drawn, those still offered; what was not found is forgotten."""
        offered = {image.image for image in self.offered()}
        self.images = [image for image in self.images if image.image in offered]
        self.not_found = []
        self._forget_undrawn()

    def _forget_undrawn(self) -> None:
        """Let go of every picture of an image no longer drawn; its masks are kept."""
        drawn = {image.image for image in self.images}
        self._shown = {path: shown for path, shown in self._shown.items() if path in drawn}
        self._drawn = {key: picture for key, picture in self._drawn.items() if key[1] in drawn}
        self.hidden &= drawn

    def set_hidden(self, image: PreviewImage, hidden: bool) -> None:
        """Leave *image* out of what is shown, or show it again; it stays drawn and predicted."""
        (self.hidden.add if hidden else self.hidden.discard)(image.image)

    @staticmethod
    def _keys(models: Sequence[PreviewModel]) -> list[tuple[str, Path | None, str, Path | None]]:
        return [
            (
                model.name, model.checkpoint.path if model.checkpoint else None, model.which,
                model.run.looked_in,
            )
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
        """Draw the images to look at, a new draw unless *again* is False, and return those found.

        An image whose file is not on this machine is kept apart, in
        :attr:`not_found`, and is neither drawn nor predicted.
        """
        if again:
            self.seed += 1
        count = max(1, min(self.count, MOST_PREVIEW_IMAGES))
        drawn = pick_images(self.offered(), self.chosen, count, self.seed)
        self.images, self.not_found = split_found(drawn)
        self.hidden = set()
        self._forget_undrawn()
        return self.images

    def describe_not_found(self) -> str:
        """Return what to say of the images drawn that are not on this machine, or ""."""
        return describe_not_found(
            self.not_found, len(self.images) + len(self.not_found),
            [model.run for model in self.models],
            "Scan the collection these runs were trained from.",
        )

    # -- predicting -----------------------------------------------------------

    def missing(self) -> tuple[list[Path], list[Path]]:
        """Return the images and the checkpoints with a prediction still to make between them.

        An image that could not be opened to draw is not one of them.
        """
        weights = [model.checkpoint.path for model in self.models if model.checkpoint is not None]
        wanted = [
            (model, image.image) for model in weights for image in self.images
            if not isinstance(self._shown.get(image.image), str)
            and (model, image.image) not in self._masks
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
            from ..operations.preview_process import preview_out_of_process

            predict = preview_out_of_process
        outcome = predict(images, weights, device, on_progress=on_progress, cancellation=cancellation)
        self._masks.update(outcome.masks)
        self._failures.update(outcome.failures)
        while len(self._masks) > MOST_MASKS:
            del self._masks[next(iter(self._masks))]
        return outcome.error

    def forget_failures(self) -> None:
        """Let what could not be predicted, or opened, be tried again."""
        self._failures.clear()
        self._shown = {path: shown for path, shown in self._shown.items() if not isinstance(shown, str)}

    # -- drawing --------------------------------------------------------------

    def shown(self, image: PreviewImage) -> DisplayImage | str:
        """Return *image* ready to draw with its own label, or why it cannot be opened."""
        if image.image not in self._shown:
            try:
                self._shown[image.image] = prepare_display_image(
                    image.image, image.label, max_edge=0
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

        The image alone until it is predicted, and at its own size. The mask
        is resized to it by nearest neighbour, as a prediction converted to a
        label is. Not to be written to: the same array is returned when asked again.

        Args:
            proofread: Draw the image's own label in place of the prediction.
        """
        shown = self.shown(image)
        if isinstance(shown, str):
            return None
        weights = None if proofread or model.checkpoint is None else model.checkpoint.path
        mask = shown.label if proofread else self._masks.get((weights, image.image))
        if mask is None or not self.label_shown:
            return shown.image
        if self._drawn_alpha != self.alpha:
            self._drawn, self._drawn_alpha = {}, self.alpha
        key = (weights, image.image)
        if key not in self._drawn:
            self._drawn[key] = compose_overlay(
                shown.image, resize_label_to(mask, shown.image.shape), self.config, alpha=self.alpha
            )
        return self._drawn[key]
