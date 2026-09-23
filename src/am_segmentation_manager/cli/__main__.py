"""``python -m am_segmentation_manager.cli``: the same as the ``am-seg`` command.

What a launcher runs, needing only the environment's interpreter. **The guard
below is load-bearing**: a worker process started by *spawn* imports this module
again as ``__mp_main__``, and without it every worker would run the command.
"""
from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
