"""Optional inference support: turning images into predictions with a model.

Everything under here needs torch, which the rest of the package does not, so
this subpackage is the boundary: no module outside it may import torch, and
nothing here is imported unless a prediction is actually being made.
:mod:`.environment` reports what is missing instead of failing on import, the
way :mod:`..io.image` does for the image stack.

``preprocessing.py`` is a verbatim copy of the AM_training module of the same
name. Keeping it byte-identical is the point -- it can be diffed against
upstream -- so nothing here edits it, and code that needs different behaviour
wraps it rather than changing it.
"""
from __future__ import annotations

import os

# albumentations makes a network request at import time to check for a newer
# release. Inference runs on microscope-room machines that may not reach the
# internet, and a dependency that pauses on import is indistinguishable from one
# that has hung. Set before any import of preprocessing.py, which is why it is
# here and not in that file: importing a submodule runs this first.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
