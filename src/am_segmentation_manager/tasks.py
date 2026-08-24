"""The vocabulary of milling task names one lamella can use.

Autolamella's task names come from the protocol somebody wrote, so they are
arbitrary strings rather than a closed set this package could enumerate. Each
lamella's entry in ``experiment.yaml`` lists its own, and matching a directory
or filename against *that* list is what makes an arbitrary name safe to read
and what splits a composite like ``Rough-Milling-Rough-2``.

**Tasks only.** A milling stage rolls up into the task that ran it, so
``Fine 1`` and ``Rough 2`` are aliases of their task rather than a second axis.
Names are compared through :func:`~.layout.normalise_task_name`, which is what
makes ``mill_polishing``, ``MillPolishing`` and ``Mill Polishing`` one task.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .layout import normalise_task_name, task_name_runs, task_name_words

logger = logging.getLogger(__name__)

#: Where a lamella's tasks are named, newest source last. A later source
#: overrides an earlier one's *spelling* of the same task, which is how
#: ``MillPolishing`` wins over the ``mill_polishing`` of the protocol it is
#: keyed by: the state is the spelling the filenames of that era use.
#:
#: ``task_history`` and ``task_config`` are the 2026-07 era; ``protocol`` and
#: ``states`` are what came before it, and a 2026-04 experiment carries both.
_TASK_SOURCES: tuple[str, ...] = ("protocol", "states", "task_config", "task_history")


@dataclass(frozen=True)
class TaskVocabulary:
    """The milling task names one lamella used, and what else names each.

    Attributes:
        aliases: Normalised name to the task it belongs to, spelt as the
            experiment spells it. A task's own name is an alias of itself, and
            so is each of its milling stages, so one lookup answers both.
        max_words: The longest alias in words, which bounds the runs
            :func:`~.layout.task_name_runs` needs to generate.
    """

    aliases: Mapping[str, str] = field(default_factory=dict)
    max_words: int = 0

    def resolve(self, name: str) -> str | None:
        """Return the task *name* belongs to, or None if it names none.

        *name* is a directory or file name; the longest run of its words that
        this vocabulary knows decides, so ``Rough-Milling-Rough-2_finished_``
        resolves through ``Rough Milling`` rather than through ``Rough 2``.
        Both name the same task, which is the point: a name with no task in it
        at all -- beam-shift alignment, an alignment run's own images -- gets
        None rather than a guess.
        """
        if not self.aliases:
            return None
        for run in task_name_runs(name, self.max_words):
            task = self.aliases.get(run)
            if task is not None:
                return task
        return None


#: A vocabulary that resolves nothing, for a lamella whose project records no
#: tasks -- every dataset predating ``experiment.yaml``, and any lamella absent
#: from its ``positions``.
NO_TASKS = TaskVocabulary()


def _text(value: Any) -> str | None:
    """Return *value* as a non-empty string, or None."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _stage_names(entry: Any) -> list[str]:
    """Return the milling stage names in one protocol or task-config entry.

    A protocol entry is a list of stages, each with its own ``name``; a
    task-config entry nests those under ``milling.<key>.stages`` and adds the
    group's own ``name``. Both are read for the same purpose -- everything that
    names this task and nothing else.
    """
    names: list[str] = []
    if isinstance(entry, list):
        for stage in entry:
            if isinstance(stage, Mapping) and (name := _text(stage.get("name"))):
                names.append(name)
        return names
    if not isinstance(entry, Mapping):
        return names
    milling = entry.get("milling")
    if isinstance(milling, Mapping):
        for group in milling.values():
            if not isinstance(group, Mapping):
                continue
            if name := _text(group.get("name")):
                names.append(name)
            names.extend(_stage_names(group.get("stages")))
    return names


def _collect(
    position: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, set[str]], int]:
    """Return one lamella's ``(tasks, aliases, max_words)``.

    *tasks* maps a task to the spelling to report it under and *aliases* maps
    every name that could appear in a directory or filename to the tasks
    claiming it, both keyed by normalised name; more than one claim happens
    only where the experiment itself is ambiguous. *max_words* is counted from
    the names as written, before normalising took their separators out.
    """
    tasks: dict[str, str] = {}
    aliases: dict[str, set[str]] = {}
    max_words = 0

    def add(task: str, *names: str) -> None:
        nonlocal max_words
        key = normalise_task_name(task)
        if not key:
            return
        tasks[key] = task
        for name in (task, *names):
            alias = normalise_task_name(name)
            if alias:
                aliases.setdefault(alias, set()).add(key)
                max_words = max(max_words, len(task_name_words(name)))

    for source in _TASK_SOURCES:
        value = position.get(source)
        if source == "states" and isinstance(value, Mapping):
            for name in value:
                if task := _text(name):
                    add(task)
        elif source == "task_history" and isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping) and (task := _text(item.get("name"))):
                    add(task)
        elif isinstance(value, Mapping):
            for name, entry in value.items():
                if task := _text(name):
                    add(task, *_stage_names(entry))
    return tasks, aliases, max_words


def build_vocabulary(position: Mapping[str, Any]) -> TaskVocabulary:
    """Return the task vocabulary one ``positions`` entry describes.

    An alias two tasks both claim is dropped rather than assigned to one of
    them: a name that could mean either says nothing, and reporting a task an
    image may not belong to is worse than reporting none.
    """
    tasks, aliases, max_words = _collect(position)
    resolved: dict[str, str] = {}
    for alias, claiming in aliases.items():
        if len(claiming) > 1:
            logger.debug(
                "Ignoring %r: it names %d tasks of this lamella",
                alias, len(claiming),
            )
            continue
        resolved[alias] = tasks[next(iter(claiming))]
    return TaskVocabulary(aliases=resolved, max_words=max_words)


def read_vocabularies(experiment: Mapping[str, Any] | None) -> dict[str, TaskVocabulary]:
    """Return a task vocabulary per lamella of one ``experiment.yaml``.

    Keyed by the lamella's ``petname``, which is the directory Autolamella
    writes its images into; an entry with none is keyed by the last part of its
    recorded ``path`` instead, since that is the same directory. An experiment
    with no ``positions`` -- every dataset predating ``experiment.yaml`` --
    yields nothing, and those lamellae then get :data:`NO_TASKS`.
    """
    positions = (experiment or {}).get("positions")
    if not isinstance(positions, list):
        return {}

    vocabularies: dict[str, TaskVocabulary] = {}
    for position in positions:
        if not isinstance(position, Mapping):
            continue
        name = _text(position.get("petname"))
        if name is None and (path := _text(position.get("path"))):
            name = path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if name is None:
            continue
        vocabulary = build_vocabulary(position)
        if vocabulary.aliases:
            vocabularies[name] = vocabulary
    return vocabularies
