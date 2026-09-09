"""The command-line surface, one module per step of the workflow.

`main.py` builds the parser and dispatches; each of the five command modules
owns both its subparsers and its handlers, so a command's arguments and the
code reading them cannot drift apart. `scope.py` holds the vocabulary they
share and `logs.py` the log file every command may be asked for.

Re-exported here so ``am_segmentation_manager.cli:main`` stays the entry point
it has always been.
"""
from __future__ import annotations

from .logs import open_log_file
from .main import COMMANDS, build_parser, main

__all__ = ["COMMANDS", "build_parser", "main", "open_log_file"]
