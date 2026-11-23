# Copyright (c) 2026
"""Exception hierarchy for the multi-metric checkpoint callback."""


class MultiMetricCheckpointError(RuntimeError):
    """Base class for every error raised by the multi-metric callback."""


class MultiMetricStateMismatchError(MultiMetricCheckpointError):
    """Raised when metadata restored from a checkpoint does not match it.

    The message always names the validation route and the missing/invalid
    field, so a corrupted or hand-edited checkpoint can never be silently
    restarted from zero.
    """


class MultiMetricWriteError(MultiMetricCheckpointError):
    """Raised when an atomic checkpoint write fails.

    The partially written temporary file is removed before this propagates,
    and the callback bookkeeping is rolled back, so later epochs can recognise
    the situation instead of believing the checkpoint exists.
    """
