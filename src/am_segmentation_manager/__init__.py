"""am_segmentation_manager

Package entrypoint and version.
"""

from importlib.metadata import PackageNotFoundError, version

from .models import AdaptiveMillingStage, ImageRecord, Lamella, Project
from .scanner import ProjectIndex, Scanner

try:
    #: Read from the installed distribution metadata, so the version is stated
    #: once, in pyproject.toml. Hardcoding it here as well guarantees the two
    #: drift apart the first time one is bumped without the other.
    __version__ = version("am_segmentation_manager")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    # Running straight from a checkout that was never pip-installed. The code
    # works; only the version is unknowable, and saying so beats reporting a
    # number that may be wrong.
    __version__ = "0.0.0+unknown"

__all__ = [
    "Project",
    "Lamella",
    "AdaptiveMillingStage",
    "ImageRecord",
    "Scanner",
    "ProjectIndex",
]
