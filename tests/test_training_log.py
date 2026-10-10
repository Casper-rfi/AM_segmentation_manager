"""The end of a run's log as a person reads it, and what a failure in it means."""
from __future__ import annotations

from pathlib import Path

from am_segmentation_manager.operations.training_process import LOG_FILENAME
from am_segmentation_manager.training_log import failure_advice, fold_progress, read_log_tail

UP = "\x1b[A"

#: The start of a run as an A100 wrote it on 2026-10-10: a bar of epochs, then a
#: bar of steps rewritten twice a second, each update a line of its own.
STARTING = (
    "2026-10-10T10:29:12Z INFO Freezing encoder for 25 epochs\n"
    "\rTraining progress:   1%|          | 1/90 [00:00<?, ?epoch/s]\n"
    f"\rEpoch 1 training:   0%|          | 0/532 [00:00<?, ?step/s]{UP}\n"
    f"\rEpoch 1 training:   0%|          | 1/532 [00:29<4:19:01, 29.27s/step]{UP}\n"
    f"\rEpoch 1 training:   3%|▎         | 16/532 [00:36<03:58,  2.16step/s]{UP}\n"
)

ENDING = (
    "2026-10-10T10:33:45Z INFO Current epoch: 1, stage: train\n"
    "loss_mean: 0.6952\n"
    "confusion_matrix_mean: 200760360.9600\n"
    "mean_of_key_metrics_mean: 0.4401\n"
    "Best train mean_of_key_metrics: 0.4401 at epoch 1\n"
    "\rTraining progress:  90%|????????? | 81/90 [7:35:06<49:37, 330.79s/epoch]"
    "2026-10-10T18:04:19Z INFO EarlyStopper: stopping training\n"
    "\r                                                                        \r"
    "Stopped early after epoch 81 due to val stopper\n"
)


def test_a_bar_rewritten_many_times_is_one_line_as_it_last_stood():
    assert fold_progress(STARTING) == [
        "2026-10-10T10:29:12Z INFO Freezing encoder for 25 epochs",
        "Training progress:   1%|          | 1/90 [00:00<?, ?epoch/s]",
        "Epoch 1 training:   3%|▎         | 16/532 [00:36<03:58,  2.16step/s]",
    ]


def test_what_is_written_on_the_end_of_a_bar_has_a_line_of_its_own():
    folded = fold_progress(ENDING)

    assert folded[-3:] == [
        "Training progress:  90%|????????? | 81/90 [7:35:06<49:37, 330.79s/epoch]",
        "2026-10-10T18:04:19Z INFO EarlyStopper: stopping training",
        "Stopped early after epoch 81 due to val stopper",
    ]


def test_an_epoch_s_table_of_metrics_is_left_out_and_what_frames_it_kept():
    folded = fold_progress(ENDING)

    assert folded[:2] == [
        "2026-10-10T10:33:45Z INFO Current epoch: 1, stage: train",
        "Best train mean_of_key_metrics: 0.4401 at epoch 1",
    ]


def test_the_same_bar_met_again_after_something_else_is_shown_again():
    bar = "Training progress:  90%|#| 81/90 [7:35:06<49:37, 330.79s/epoch]"

    assert fold_progress(f"{bar}\nModel saved\n{bar}\n") == [bar, "Model saved", bar]


def test_a_traceback_is_kept_line_for_line():
    traceback = (
        "Traceback (most recent call last):\n"
        '  File "train.py", line 3, in run\n'
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.\n"
    )

    assert fold_progress(traceback) == [line.strip() for line in traceback.splitlines()]


def test_windows_line_ends_leave_no_blank_lines():
    assert fold_progress("one\r\ntwo\r\n") == ["one", "two"]


def test_the_end_of_a_log_is_read_from_its_directory(tmp_path: Path):
    (tmp_path / LOG_FILENAME).write_text(STARTING, encoding="utf-8")

    assert read_log_tail(tmp_path)[-1].startswith("Epoch 1 training:   3%")


def test_a_directory_with_no_log_reads_as_nothing(tmp_path: Path):
    assert read_log_tail(tmp_path) == []


def test_only_the_end_of_a_long_log_is_read_and_never_half_a_line(tmp_path: Path):
    (tmp_path / LOG_FILENAME).write_text(
        "".join(f"line {number:05}\n" for number in range(2000)), encoding="utf-8"
    )

    tail = read_log_tail(tmp_path, lines=5, most_bytes=100)

    assert tail == [f"line {number:05}" for number in range(1995, 2000)]
    assert read_log_tail(tmp_path, lines=50, most_bytes=100)[0].startswith("line 0")


def test_bytes_that_are_not_text_do_not_stop_the_log_being_read(tmp_path: Path):
    (tmp_path / LOG_FILENAME).write_bytes(b"before\n\xff\xfe broken\nafter\n")

    assert read_log_tail(tmp_path)[0] == "before" and read_log_tail(tmp_path)[-1] == "after"


def test_running_out_of_gpu_memory_is_said_in_words_with_the_batch():
    error = "OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has...')"

    advice = failure_advice(error, 6)

    assert "ran out of memory with batches of 6" in advice
    assert "smaller Training batch" in advice


def test_a_batch_of_one_that_does_not_fit_is_not_told_to_go_lower():
    advice = failure_advice("RuntimeError('CUDA error: out of memory')", 1)

    assert "smaller" not in advice and "more memory" in advice


def test_a_failure_not_recognised_is_given_no_advice():
    assert failure_advice("FileNotFoundError('weights.pth')", 6) == ""
