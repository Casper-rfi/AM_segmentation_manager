"""Training runs: preparing, starting, reading, validating and comparing them.

Beside the spine rather than in it: a run lives outside the collection, so these modules
read and write run directories, not the ProjectIndex. They build on ``operations/``
(a run is prepared from a training export) and nothing in ``operations/`` imports them.
Each module is imported by its own name; this package re-exports nothing.
"""
