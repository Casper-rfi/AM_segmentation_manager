"""CSV helpers for exporting training and metadata files.

This module will eventually contain functions to create CSV files used for
training and validation. For now it exposes a small helper to turn a list of
ImageRecord objects into a pandas DataFrame. pandas is optional at runtime so
we only import it inside the function.
"""
from __future__ import annotations

from typing import Iterable
import logging
from pathlib import Path

from ..models import ImageRecord

logger = logging.getLogger(__name__)


def records_to_dataframe(records: Iterable[ImageRecord]):
    """Convert ImageRecord objects into a pandas.DataFrame.

    Raises RuntimeError if pandas is not available.
    """
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("pandas is required to create DataFrames: install pandas") from exc

    rows = []
    for r in records:
        rows.append({
            "image_path": str(r.image_path),
            "project": r.project,
            "lamella": r.lamella,
            "stage": r.stage,
            "modality": r.modality,
            "selected": bool(r.selected),
        })

    df = pd.DataFrame(rows)
    logger.debug("Created DataFrame with %d rows", len(df))
    return df
