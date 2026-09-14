"""Optional inference support: turning images into predictions with a model.

Everything under here needs torch, which nothing outside it may import, and
nothing here is imported unless a prediction is being made; :mod:`.environment`
reports what is missing rather than failing on import. ``preprocessing.py`` is a
verbatim copy of AM_training's module, kept byte-identical so it can be diffed
against upstream: code needing different behaviour wraps it.
"""
from __future__ import annotations

import os

# albumentations makes a network request at import time to check for a newer
# release. Inference runs on microscope-room machines that may not reach the
# internet, and a dependency that pauses on import is indistinguishable from one
# that has hung. Set before any import of preprocessing.py, which is why it is
# here and not in that file: importing a submodule runs this first.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
