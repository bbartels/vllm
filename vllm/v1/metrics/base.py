# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.metrics.stats import IterationStats, SchedulerStats


class StatLoggerBase(ABC):
    """Interface for logging metrics.

    API users may define custom loggers that implement this interface.
    However, note that the `SchedulerStats` and `IterationStats` classes
    are not considered stable interfaces and may change in future versions.
    """

    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        ...

    @abstractmethod
    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0):
        ...

    @abstractmethod
    def log_engine_initialized(self):
        ...

    def log(self):  # noqa: D401
        """Optional periodic log hook for human-readable output."""
        pass

