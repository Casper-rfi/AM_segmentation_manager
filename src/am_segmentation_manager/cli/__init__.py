"""The command-line surface, one module per step of the workflow.

`main.py` builds the parser and dispatches; each command module owns its
heading, its subparsers and its handlers together, so a command's arguments,
the code reading them and the step it is filed under cannot drift apart.
`scope.py` holds the vocabulary they share and `logs.py` the log file every
command may be asked for.

Re-exported here so ``am_segmentation_manager.cli:main`` stays the entry point
it has always been.
"""
from __future__ import annotations

from .logs import open_log_file
from .main import COMMANDS, build_parser, main

__all__ = ["COMMANDS", "build_parser", "main", "open_log_file"]
