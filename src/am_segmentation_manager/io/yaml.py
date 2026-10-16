"""Reading and writing the YAML files the package keeps.

A missing file reads as an empty mapping, most of these files being optional;
one there that will not read as a mapping raises :class:`UnreadableYAMLError`.
**A write lands whole or not at all**, through :func:`.atomic.write_atomically`,
so an interrupted run leaves the previous file rather than a truncated one.
PyYAML's C loader is used where it was built against libyaml: Autolamella's
``experiment.yaml`` reaches a megabyte and a scan reads one per project.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .atomic import Fingerprint, fingerprint_of, unrefused, write_atomically

logger = logging.getLogger(__name__)


def _loader():
    """Return the fastest available safe YAML loader.

    Raises:
        RuntimeError: If PyYAML is not installed.
    """
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("PyYAML is required to read YAML files: install pyyaml") from exc

    try:
        return yaml.CSafeLoader
    except AttributeError:  # pragma: no cover - depends on the local PyYAML build
        logger.debug("libyaml not available; using the pure Python YAML loader")
        return yaml.SafeLoader


class UnreadableYAMLError(RuntimeError):
    """A YAML file is there but does not parse, or does not hold a mapping."""


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary.

    A missing file reads as an empty mapping, since most of the files this
    package reads are optional.

    Raises:
        UnreadableYAMLError: If the file does not parse or its top level is not
            a mapping.
        RuntimeError: If PyYAML is not installed.
    """
    import yaml

    loader = _loader()
    if not path.exists():
        logger.debug("YAML file does not exist: %s", path)
        return {}

    try:
        with unrefused(lambda: path.open("r", encoding="utf-8")) as fh:
            data = yaml.load(fh, Loader=loader) or {}  # noqa: S506 - CSafeLoader or SafeLoader
    except Exception as exc:
        raise UnreadableYAMLError(f"Failed to parse YAML file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise UnreadableYAMLError(
            f"Expected a mapping at the top level of {path}, found {type(data).__name__}"
        )
    logger.debug("Read YAML from %s: %s keys", path, len(data))
    return data


class DuplicateKeyError(UnreadableYAMLError):
    """A YAML mapping names one key twice."""


_unique_loader: Any = None


def load_unique_keys(stream: Any) -> Any:
    """Parse YAML from *stream*, refusing a repeated key rather than keeping one.

    ``SafeLoader`` keeps the last of a repeated key silently, which in a
    hand-editable file is an edit read as its opposite.

    Raises:
        DuplicateKeyError: If a mapping repeats a key.
    """
    global _unique_loader
    import yaml

    if _unique_loader is None:

        class _NoDuplicateKeys(yaml.SafeLoader):
            """A loader for which a repeated key is an error."""

            def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
                seen: set[Any] = set()
                for key_node, _ in node.value:
                    key = self.construct_object(key_node, deep=deep)
                    if key in seen:
                        raise DuplicateKeyError(f"Duplicate entry for {key!r}")
                    seen.add(key)
                return super().construct_mapping(node, deep)

        _unique_loader = _NoDuplicateKeys

    return yaml.load(stream, Loader=_unique_loader)  # noqa: S506 - a SafeLoader subclass


def read_yaml_or_problem(path: Path) -> tuple[dict[str, Any], str | None]:
    """Read a YAML file as :func:`read_yaml` does, returning what is wrong rather than raising.

    Returns:
        The mapping, empty where the file will not read, and the problem or None.
    """
    try:
        return read_yaml(path), None
    except UnreadableYAMLError as error:
        return {}, str(error)


def write_yaml(
    path: Path,
    body: Mapping[str, Any],
    *,
    header: str = "",
    sort_keys: bool = False,
    commit_if: Callable[[Path], bool] | None = None,
) -> bool:
    """Write *body* to *path* as YAML beneath *header*, replacing the file whole.

    Args:
        header: Comment lines to open the file with, each ending in a newline.
        sort_keys: Sort every mapping's keys; otherwise they keep their order.
        commit_if: As :func:`.atomic.write_atomically`: asked just before the
            rename whether to go ahead.

    Returns:
        Whether *path* was replaced.
    """
    import yaml

    text = header + yaml.safe_dump(
        dict(body), sort_keys=sort_keys, allow_unicode=True, default_flow_style=False
    )
    return write_atomically(
        path, lambda temporary: temporary.write_text(text, encoding="utf-8"), commit_if=commit_if
    )


def write_yaml_unchanged(
    path: Path, body: Mapping[str, Any], *, header: str, expected: Fingerprint | None
) -> bool:
    """Write *body* as :func:`write_yaml` does, only while *path* is as *expected*.

    Asked before anything is encoded and again just before the rename, which is
    as late as it can be asked.

    Args:
        expected: The file's fingerprint when it was read, None if it did not exist.

    Returns:
        Whether *path* was replaced; False where it changed since it was read.
    """
    if fingerprint_of(path) != expected:
        return False
    return write_yaml(
        path, body, header=header,
        commit_if=lambda _temporary: fingerprint_of(path) == expected,
    )


def version_problem(
    version: Any, path: Path, latest: int, *, required: bool = False
) -> str | None:
    """Return why a file's ``version`` cannot be read by this AM-seg, or None if it can.

    A version above *latest* was written by a later AM-seg, which is a refusal
    rather than a misreading. Each reader raises its own error with the message.

    Args:
        required: Refuse a file that states no version, rather than reading it
            as the first.
    """
    if version is None and not required:
        return None
    if not isinstance(version, int) or isinstance(version, bool):
        return f"{path}: version must be a whole number"
    if version > latest:
        return (
            f"{path} was written by a later version of AM-seg "
            f"(file version {version}, this reads up to {latest})"
        )
    return None
