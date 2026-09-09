"""Running a prediction in a process of its own.

Inference inside the window's own process crashed it: repeatedly, in a
different place each time, always in an allocator or destructor on the
interface thread shortly after a run finished, and never once in a hundred-odd
command-line runs doing the same work on the same machine. What the two differ
in is the thread inference runs on and the Qt event loop around it, and neither
is something this package can change from inside.

So it does not try. A child process holds torch, loads the model and writes the
predictions; the window holds none of it and cannot be corrupted by it. Nothing
is lost when that child dies, however it dies -- each prediction is written by
rename and its provenance flushed before the next begins, so what finished is
on disk and the next run resumes from it. That property already existed for the
Stop button, and is what makes this cheap.

The child is started with the *spawn* method: a fresh interpreter that inherits
no imports, so nothing Qt has touched is carried into it.
"""
from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import os
import signal
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..models import ImageRecord
from ..progress import Cancellation, OperationCancelled
from .predict import PredictionSource, plan_predictions, run_predictions

logger = logging.getLogger(__name__)

#: How long to wait for a stopped child to finish the image it is on, before
#: taking it down harder. A prediction is seconds; a minute is generous enough
#: that expiring it means the child is wedged rather than busy.
STOP_GRACE_SECONDS = 60.0

#: How long to wait for a device probe. It only imports torch and asks what is
#: available, but that import is seconds on a cold filesystem.
PROBE_SECONDS = 60.0


@dataclass(frozen=True)
class ProcessOutcome:
    """What a child process did, brought back to the window.

    Everything here is plain data: it crosses a process boundary, so it holds
    no record objects and nothing that would need torch to unpickle.

    Attributes:
        written: Predictions written to disk.
        total: Predictions the plan held.
        modality: The one modality the run covered, or "" if it was refused.
        modalities: What the selection held, for the refusal message.
        destinations: Where each prediction went, relative paths as given.
        up_to_date: Images this model had already predicted.
        represented: Images covered by another image's prediction.
        stopped: Whether the run was asked to stop before it finished.
        error: Why nothing ran, or "" when it did.
    """

    written: int = 0
    total: int = 0
    modality: str = ""
    modalities: tuple[str, ...] = ()
    destinations: tuple[str, ...] = ()
    up_to_date: int = 0
    represented: int = 0
    stopped: bool = False
    error: str = ""
    devices: tuple[str, ...] = field(default=())

    @property
    def refused(self) -> bool:
        """Return whether the selection named more than one modality."""
        return len(self.modalities) > 1


class _QueueLog(logging.handlers.QueueHandler):
    """Sends the child's log records to the parent, tagged for its pump.

    Without this the child logs into a process nobody is reading: no handler is
    configured there, so which device the model loaded onto -- the line that
    placed the last crash -- would be lost exactly when it is wanted. Inherited
    from ``QueueHandler`` for its ``prepare``, which renders the message and
    drops what cannot be pickled.
    """

    def enqueue(self, record) -> None:
        self.queue.put(("log", record))


def _forward_logs(queue) -> None:
    """Send everything this child logs to the parent. Runs in the child."""
    root = logging.getLogger()
    root.handlers = [_QueueLog(queue)]
    root.setLevel(logging.INFO)


def _child(records: Sequence[ImageRecord], weights: str, device: str, queue) -> None:
    """Plan and run the predictions, reporting progress down *queue*.

    Runs in the child. Imports torch and nothing here does anywhere else, which
    is the entire point of the module. Every outcome leaves by the queue,
    including failure: an exception here reaches no handler that could show it.
    """
    _forward_logs(queue)
    # Inside the child process, deliberately: torch is never imported into
    # the parent, which is the whole point of running inference here and is
    # what CLAUDE.md's rule about ml/ asks for.
    from ..ml.environment import available_devices, resolve_device
    from ..ml.model import load_model, read_model_description, run_name_of

    path = Path(weights)
    try:
        description = read_model_description(path)
        source = PredictionSource(name=path.name, run_name=run_name_of(path))
        plan = plan_predictions(records, source, overwrite=False)
        destinations = tuple(str(item.destination) for item in plan.predictions)
        common = {
            "total": len(plan.predictions),
            "modality": plan.modality or "",
            "modalities": plan.modalities,
            "destinations": destinations,
            "up_to_date": len(plan.up_to_date),
            "represented": plan.represented,
            "devices": tuple(available_devices()),
        }
        if plan.is_refused or not plan.predictions:
            queue.put(("done", ProcessOutcome(**common)))
            return

        resolved = resolve_device(device)
        logger.info("Loading %s onto %s", path.name, resolved)
        loaded = load_model(path, description, device=resolved)

        def report(index: int, _planned) -> None:
            queue.put(("progress", index, common["total"]))

        try:
            written = run_predictions(plan, loaded.predict, source, on_progress=report)
            stopped = False
        except KeyboardInterrupt:
            # The parent asking to stop. Everything written is kept and
            # recorded, so counting the files is the honest answer.
            written = sum(1 for item in plan.predictions if item.destination.exists())
            stopped = True
        queue.put(("done", ProcessOutcome(written=written, stopped=stopped, **common)))
    except BaseException as error:  # noqa: BLE001 - the only way back is the queue
        queue.put(("done", ProcessOutcome(error=f"{type(error).__name__}: {error}")))


def predict_out_of_process(
    records: Iterable[ImageRecord],
    weights: Path,
    device: str = "auto",
    *,
    on_progress: Callable[[int, int], None] | None = None,
    cancellation: Cancellation | None = None,
    context: str | None = None,
) -> ProcessOutcome:
    """Run the predictions in a child process and report what it did.

    Blocking: it is already called on a worker thread, and a second layer of
    asynchrony would buy nothing.

    Args:
        records: The images to predict.
        weights: The checkpoint to predict with.
        device: A torch device, or "auto" for the most capable one the *child*
            can see -- which is the only process that can answer.
        on_progress: Called as ``on_progress(done, total)`` as the child works.
        cancellation: Polled while waiting. Stopping interrupts the child, which
            finishes the image it is on and keeps it.
        context: Multiprocessing start method. Defaults to "spawn", which is
            what makes the child inherit none of this process's imports.

    Returns:
        What the child did, including why it could not start.
    """
    records = list(records)
    method = context or "spawn"
    ctx = multiprocessing.get_context(method)
    queue = ctx.Queue()
    process = ctx.Process(
        target=_child, args=(records, str(weights), device, queue), daemon=True
    )
    logger.info(
        "Predicting %d image(s) in a child process (%s)", len(records), method
    )
    process.start()
    try:
        return _await(process, queue, on_progress, cancellation)
    finally:
        _reap(process)


def _await(process, queue, on_progress, cancellation) -> ProcessOutcome:
    """Pump the child's messages until it reports an outcome or dies."""
    import queue as queue_module

    asked_to_stop = False
    while True:
        # Checked every time round rather than only when the queue runs dry: a
        # child reporting steadily would otherwise never leave that branch, and
        # Stop would do nothing for exactly the runs long enough to want it.
        if not asked_to_stop and cancellation is not None and cancellation.requested:
            # SIGINT rather than terminate: the child catches it between whole
            # images, where a signal it cannot handle would leave the temporary
            # of a half-written prediction behind.
            asked_to_stop = True
            logger.info("Asking the prediction process to stop")
            _interrupt(process)

        try:
            message = queue.get(timeout=0.2)
        except queue_module.Empty:
            if not process.is_alive():
                # No outcome and no process: it was killed from outside, or
                # died the way this module exists to survive.
                return ProcessOutcome(
                    error=(
                        f"The prediction process stopped without reporting "
                        f"anything (exit code {process.exitcode})."
                    )
                )
            continue

        if message[0] == "log":
            # Re-emitted here so one log file holds the whole run, the child's
            # part of it included -- which is what makes a crash placeable.
            logging.getLogger(message[1].name).handle(message[1])
        elif message[0] == "progress":
            if on_progress is not None:
                on_progress(message[1], message[2])
        elif message[0] == "done":
            return message[1]


def _interrupt(process) -> None:
    """Ask the child to stop at its next image."""
    try:
        os.kill(process.pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError, AttributeError) as error:
        logger.warning("Could not interrupt the prediction process: %s", error)
        process.terminate()


def _reap(process) -> None:
    """Make sure the child is gone, whatever happened to it."""
    process.join(STOP_GRACE_SECONDS)
    if process.is_alive():
        logger.warning("The prediction process did not stop; terminating it")
        process.terminate()
        process.join(5.0)


def _probe(queue) -> None:
    """Report the devices torch can see. Runs in the child."""
    try:
        # In the child, as above: the parent never imports torch.
        from ..ml.environment import available_devices

        queue.put(tuple(available_devices()))
    except BaseException:  # noqa: BLE001 - a machine that cannot answer says nothing
        queue.put(())


def probe_devices(context: str | None = None) -> tuple[str, ...]:
    """Return the torch devices available, asked of a separate process.

    The window no longer imports torch at all -- that is what this module is
    for -- so it cannot answer this itself, and the answer differs per machine:
    several CUDA devices on a workstation, an MPS one on a Mac. Costs one
    interpreter start and one torch import, so callers ask once and keep it.

    Returns:
        The devices, most capable first, or ``()`` if the child could not say.
    """
    ctx = multiprocessing.get_context(context or "spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_probe, args=(queue,), daemon=True)
    process.start()
    try:
        import queue as queue_module

        try:
            return queue.get(timeout=PROBE_SECONDS)
        except queue_module.Empty:
            logger.warning("Timed out asking which devices are available")
            return ()
    finally:
        _reap(process)


def raise_if_cancelled(cancellation: Cancellation | None) -> None:
    """Raise if stopping was asked for. Kept here so callers need no import."""
    if cancellation is not None:
        cancellation.check()


__all__ = [
    "OperationCancelled",
    "ProcessOutcome",
    "predict_out_of_process",
    "probe_devices",
    "raise_if_cancelled",
]
