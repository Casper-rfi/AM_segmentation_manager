"""Filesystem scanner and project index.

The scanner is responsible for discovering Autolamella projects by locating
experiment.yaml files and building an in-memory ProjectIndex. The remainder of
the package should operate on the ProjectIndex rather than re-traversing the
filesystem.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Iterator
import logging
import re
from datetime import datetime

from .models import Project, Lamella, AdaptiveMillingStage, ImageRecord
from .io.yaml import read_yaml
from .io.path import make_path_relative
from .metadata import merge_metadata

logger = logging.getLogger(__name__)


class ProjectIndex:
    """Container for discovered projects.

    Provides lookup utilities that other components can depend on.
    """

    def __init__(self, start_path: Path | None = None) -> None:
        self.start_path = start_path
        self.projects: Dict[str, Project] = {}

    def add(self, project: Project) -> None:
        self.projects[project.name] = project

    def all_projects(self) -> List[Project]:
        return list(self.projects.values())

    def find_by_name(self, name: str) -> Project:
        return self.projects[name]

    def relative_path(self, path: Path) -> str:
        """Return a string path relative to the scan start_path."""
        if self.start_path is None:
            return str(path)
        return make_path_relative(path, self.start_path)


class Scanner:
    """Scan a starting directory for Autolamella projects.

    The scanner looks for experiment.yaml files. When found, it loads
    experiment metadata (via read_yaml) and optional AM_metadata.yaml that can
    override experiment values. It then constructs a lightweight Project model
    and adds it to a ProjectIndex.
    """

    EXPERIMENT_FILENAME = "experiment.yaml"
    AM_METADATA_FILENAME = "AM_metadata.yaml"

    def __init__(self, start_path: Path) -> None:
        self.start_path = start_path

    def find_experiment_files(self) -> Iterator[Path]:
        """Yield paths to experiment.yaml files under start_path.

        Uses rglob so nested projects are supported.
        """
        logger.info("Scanning for %s under %s", self.EXPERIMENT_FILENAME, self.start_path)
        for p in self.start_path.rglob(self.EXPERIMENT_FILENAME):
            logger.debug("Found experiment file: %s", p)
            yield p

    def build_index(self) -> ProjectIndex:
        """Scan and build a ProjectIndex for discovered projects.

        The implementation focuses on a robust, minimal index that can be
        extended later. This version additionally discovers Adaptive Milling
        stages and image records inside each lamella so downstream operations
        can operate on in-memory objects instead of re-traversing the FS.

        Behaviour:
        - discovers projects by experiment.yaml
        - reads experiment.yaml and AM_metadata.yaml (if present)
        - collects lamella folders directly under the project folder
        - for each lamella, finds stage folders that contain a 'sem' or 'fib' subfolder
        - for each stage, indexes images in sem/ and fib/ (including deselected subfolders)
        - matches predictions and labels by filename when possible
        """
        index = ProjectIndex(start_path=self.start_path)

        for project_index, experiment_path in enumerate(sorted(self.find_experiment_files()), start=1):
            project_root = experiment_path.parent
            project_folder_name = project_root.name

            experiment_meta = read_yaml(experiment_path)
            am_meta_path = project_root / self.AM_METADATA_FILENAME
            am_meta = read_yaml(am_meta_path) if am_meta_path.exists() else {}
            merged_meta = merge_metadata(experiment_meta, am_meta)

            metadata_name = merged_meta.get("name")
            project_name = metadata_name if isinstance(metadata_name, str) and metadata_name else project_folder_name
            project = Project(name=project_name, path=project_root, metadata=merged_meta)
            # Project numbers are scan-order identifiers, independent of the
            # project folder name, so users can select projects as 001, 002…
            project.number = f"{project_index:03d}"

            # Discover lamella folders under the project root.
            # A lamella is any top-level directory that contains one or more
            # Adaptive Milling stages. This avoids hardcoding numeric prefixes.
            for child in sorted(project_root.iterdir()):
                if not child.is_dir():
                    continue

                stages = self._discover_stages(child, project_name=project_name, lamella_name=child.name)
                if not stages:
                    continue

                lam = Lamella(name=child.name, path=child)
                lam.number = self._parse_numeric_prefix(child.name)
                lam.stages = stages
                project.lamellae.append(lam)

            index.add(project)
            logger.info("Indexed project %s with %d lamellae", project_name, len(project.lamellae))

        return index

    def _discover_stages(self, lamella_path: Path, project_name: str, lamella_name: str) -> List[AdaptiveMillingStage]:
        """Discover Adaptive Milling stages inside a lamella folder.
        
        Heuristic: any directory under lamella_path that contains a 'sem' or 'fib'
        subdirectory is considered a stage folder. This covers Adaptive_polish_*
        and other similarly structured stage directories.
        """
        stages: List[AdaptiveMillingStage] = []

        candidate_paths = []
        for p in lamella_path.rglob("*"):
            if not p.is_dir():
                continue
            sem_dir = p / "sem"
            fib_dir = p / "fib"
            if sem_dir.exists() or fib_dir.exists():
                candidate_paths.append(p)

        for index, p in enumerate(sorted(candidate_paths, key=self._stage_sort_key), start=1):
            stage = AdaptiveMillingStage(name=p.name, path=p)
            stage.number = f"{index:02d}"
            stage.images = self._discover_images_in_stage(p, project_name=project_name, lamella_name=lamella_name)
            stages.append(stage)
            logger.debug("Discovered stage %s with %d images", p, len(stage.images))

        return stages

    @classmethod
    def _stage_sort_key(cls, stage_path: Path) -> tuple[bool, datetime, str]:
        """Order stages by the timestamp embedded in their folder name.

        Undated folders are placed after dated stages, then ordered by path for
        deterministic output.
        """
        timestamp = cls._timestamp_from_filename(stage_path.name)
        return (timestamp is None, timestamp or datetime.max, str(stage_path))

    @staticmethod
    def _timestamp_from_filename(filename: str) -> datetime | None:
        """Extract a common Autolamella timestamp from a folder name."""
        match = re.search(
            r"(\d{4})[-_]?([01]\d)[-_]?([0-3]\d)[T _-]?([0-2]\d)[-_:]?([0-5]\d)[-_:]?([0-5]\d)",
            filename,
        )
        if match is None:
            return None
        try:
            return datetime(*(int(value) for value in match.groups()))
        except ValueError:
            return None

    @staticmethod
    def _parse_numeric_prefix(value: str) -> str | None:
        """Return a zero-padded numeric prefix from a folder name when present."""
        match = re.match(r"^0*(\d+)", value)
        if not match:
            return None
        return f"{int(match.group(1)):02d}"

    def _discover_images_in_stage(self, stage_path: Path, project_name: str, lamella_name: str) -> List[ImageRecord]:
        """Discover image files, predictions and labels inside a stage folder.

        The function searches for sem/ and fib/ image files (including any
        'deselected' subfolders). Predictions and labels are matched by
        filename when possible from the standard locations:
        - predictions/ (with optional subfolders raw/ and clean/)
        - predictions_fib/
        - labels/
        - labels_fib/

        Only .tif and .tiff file extensions are considered.
        """
        image_records: List[ImageRecord] = []
        image_exts = {".tif", ".tiff"}

        def collect_files(folder: Path) -> List[Path]:
            if not folder.exists():
                return []
            return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in image_exts]

        # build maps of predictions and labels by filename for quick lookup
        def collect_by_name(base: Path) -> dict:
            files = {}
            if not base.exists():
                return files
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in image_exts:
                    files.setdefault(p.name, []).append(p)
            return files

        predictions_map = collect_by_name(stage_path / "predictions")
        predictions_map.update(collect_by_name(stage_path / "predictions" / "raw"))
        predictions_map.update(collect_by_name(stage_path / "predictions" / "clean"))
        predictions_fib_map = collect_by_name(stage_path / "predictions_fib")
        predictions_fib_map.update(collect_by_name(stage_path / "predictions_fib" / "raw"))
        predictions_fib_map.update(collect_by_name(stage_path / "predictions_fib" / "clean"))

        labels_map = collect_by_name(stage_path / "labels")
        labels_map.update(collect_by_name(stage_path / "labels" / "deselected"))
        labels_map.update(collect_by_name(stage_path / "labels" / "corrected"))
        labels_fib_map = collect_by_name(stage_path / "labels_fib")
        labels_fib_map.update(collect_by_name(stage_path / "labels_fib" / "deselected"))
        labels_fib_map.update(collect_by_name(stage_path / "labels_fib" / "corrected"))

        # discover SEM images (include corrected and deselected folders)
        for sem_folder in [stage_path / "sem", stage_path / "sem" / "deselected", stage_path / "sem" / "corrected"]:
            for img in collect_files(sem_folder):
                rec = ImageRecord(
                    image_path=img,
                    project=project_name,
                    lamella=lamella_name,
                    stage=stage_path.name,
                    modality="SEM",
                )
                # attach matching predictions and labels
                rec.predictions = predictions_map.get(img.name, [])
                rec.labels = labels_map.get(img.name, [])
                rec.corrected = ("corrected" in str(img.parent)) or any(("corrected" in str(p.parent)) for p in rec.labels)
                # States are mutually exclusive: corrected records are not
                # also reported as selected.
                rec.selected = not rec.corrected and ("deselected" not in str(img.parent))
                image_records.append(rec)
  
        # discover FIB images (include corrected and deselected folders)
        for fib_folder in [stage_path / "fib", stage_path / "fib" / "deselected", stage_path / "fib" / "corrected"]:
            for img in collect_files(fib_folder):
                rec = ImageRecord(
                    image_path=img,
                    project=project_name,
                    lamella=lamella_name,
                    stage=stage_path.name,
                    modality="FIB",
                )
                rec.predictions = predictions_fib_map.get(img.name, [])
                rec.labels = labels_fib_map.get(img.name, [])
                rec.corrected = ("corrected" in str(img.parent)) or any(("corrected" in str(p.parent)) for p in rec.labels)
                rec.selected = not rec.corrected and ("deselected" not in str(img.parent))
                image_records.append(rec)

        return image_records
