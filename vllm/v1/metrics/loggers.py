# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import time
from typing import Callable, Optional

from vllm.config import SupportsMetricsInfo, VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import PrefixCachingMetrics
from vllm.v1.metrics.base import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from vllm.v1.metrics.agnostic_logger import (
    OpenTelemetryAgnosticMetricsLogger,
    PrometheusAgnosticMetricsLogger,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.spec_decode.metrics import SpecDecodingLogging

logger = init_logger(__name__)

StatLoggerFactory = Callable[[VllmConfig, int], "StatLoggerBase"]


## StatLoggerBase now lives in vllm.v1.metrics.base


class LoggingStatLogger(StatLoggerBase):

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.engine_index = engine_index
        self.vllm_config = vllm_config
        self._reset(time.monotonic())
        self.last_scheduler_stats = SchedulerStats()
        # Prefix cache metrics. This cannot be reset.
        # TODO: Make the interval configurable.
        self.prefix_caching_metrics = PrefixCachingMetrics()
        self.spec_decoding_logging = SpecDecodingLogging()
        self.last_prompt_throughput: float = 0.0
        self.last_generation_throughput: float = 0.0

    def _reset(self, now):
        self.last_log_time = now

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: int = 0
        self.num_generation_tokens: int = 0

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        # Save tracked stats for token counters.
        self.num_prompt_tokens += iteration_stats.num_prompt_tokens
        self.num_generation_tokens += iteration_stats.num_generation_tokens

    def _get_throughput(self, tracked_stats: int, now: float) -> float:
        # Compute summary metrics for tracked stats
        delta_time = now - self.last_log_time
        if delta_time <= 0.0:
            return 0.0
        return float(tracked_stats / delta_time)

    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0):
        """Log Stats to standard output."""

        if iteration_stats:
            self._track_iteration_stats(iteration_stats)

        if scheduler_stats is not None:
            self.prefix_caching_metrics.observe(
                scheduler_stats.prefix_cache_stats)

            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_logging.observe(
                    scheduler_stats.spec_decoding_stats)

            self.last_scheduler_stats = scheduler_stats

    def log(self):
        now = time.monotonic()
        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(
            self.num_generation_tokens, now)

        self._reset(now)

        scheduler_stats = self.last_scheduler_stats

        log_fn = logger.info
        if not any(
            (prompt_throughput, generation_throughput,
             self.last_prompt_throughput, self.last_generation_throughput)):
            # Avoid log noise on an idle production system
            log_fn = logger.debug
        self.last_generation_throughput = generation_throughput
        self.last_prompt_throughput = prompt_throughput

        # Format and print output.
        log_fn(
            "Engine %03d: "
            "Avg prompt throughput: %.1f tokens/s, "
            "Avg generation throughput: %.1f tokens/s, "
            "Running: %d reqs, Waiting: %d reqs, "
            "GPU KV cache usage: %.1f%%, "
            "Prefix cache hit rate: %.1f%%",
            self.engine_index,
            prompt_throughput,
            generation_throughput,
            scheduler_stats.num_running_reqs,
            scheduler_stats.num_waiting_reqs,
            scheduler_stats.kv_cache_usage * 100,
            self.prefix_caching_metrics.hit_rate * 100,
        )
        self.spec_decoding_logging.log(log_fn=log_fn)

    def log_engine_initialized(self):
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.info(
                "Engine %03d: vllm cache_config_info with initialization "
                "after num_gpu_blocks is: %d", self.engine_index,
                self.vllm_config.cache_config.num_gpu_blocks)


class PrometheusStatLogger(PrometheusAgnosticMetricsLogger):
    """Prometheus-backed provider-agnostic logger (keeps class name)."""
    pass


def _is_otel_metrics_enabled(vllm_config: VllmConfig) -> bool:
    try:
        endpoint = vllm_config.observability_config.otlp_metrics_endpoint
        return endpoint is not None and str(endpoint) != ""
    except Exception:
        return False


class StatLoggerManager:
    """
    StatLoggerManager:
        Logging happens at the level of the EngineCore (per scheduler).
         * DP: >1 EngineCore per AsyncLLM - loggers for each EngineCore.
         * With Local Logger, just make N copies for N EngineCores.
         * With Prometheus, we need a single logger with N "labels"

        This class abstracts away this implementation detail from
        the AsyncLLM, allowing the AsyncLLM to just call .record()
        and .log() to a simple interface.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_idxs: Optional[list[int]] = None,
        custom_stat_loggers: Optional[list[StatLoggerFactory]] = None,
        enable_default_loggers: bool = True,
    ):
        self.engine_idxs = engine_idxs if engine_idxs else [0]

        factories: list[StatLoggerFactory] = []
        if custom_stat_loggers is not None:
            factories.extend(custom_stat_loggers)

        if enable_default_loggers and logger.isEnabledFor(logging.INFO):
            factories.append(LoggingStatLogger)

        # engine_idx: StatLogger
        self.per_engine_logger_dict: dict[int, list[StatLoggerBase]] = {}
        prometheus_factory = PrometheusStatLogger
        for engine_idx in self.engine_idxs:
            loggers: list[StatLoggerBase] = []
            for logger_factory in factories:
                # If we get a custom prometheus logger, use that
                # instead. This is typically used for the ray case.
                if (isinstance(logger_factory, type)
                        and issubclass(logger_factory, PrometheusStatLogger)):
                    prometheus_factory = logger_factory
                    continue
                loggers.append(logger_factory(vllm_config,
                                              engine_idx))  # type: ignore
            self.per_engine_logger_dict[engine_idx] = loggers

        # Always configure Prometheus as before
        unregister_vllm_metrics()
        self.prometheus_logger = prometheus_factory(vllm_config, engine_idxs)

        # Optionally enable OTel metrics in addition to Prometheus
        self.otel_logger: Optional[StatLoggerBase] = None
        if _is_otel_metrics_enabled(vllm_config):
            self.otel_logger = OpenTelemetryAgnosticMetricsLogger(
                vllm_config, engine_idxs)

    def record(
        self,
        scheduler_stats: Optional[SchedulerStats],
        iteration_stats: Optional[IterationStats],
        engine_idx: Optional[int] = None,
    ):
        if engine_idx is None:
            engine_idx = 0

        per_engine_loggers = self.per_engine_logger_dict[engine_idx]
        for logger in per_engine_loggers:
            logger.record(scheduler_stats, iteration_stats, engine_idx)

        self.prometheus_logger.record(scheduler_stats, iteration_stats,
                                      engine_idx)
        if self.otel_logger is not None:
            self.otel_logger.record(scheduler_stats, iteration_stats,
                                    engine_idx)

    def log(self):
        for per_engine_loggers in self.per_engine_logger_dict.values():
            for logger in per_engine_loggers:
                logger.log()

    def log_engine_initialized(self):
        self.prometheus_logger.log_engine_initialized()
        if self.otel_logger is not None:
            self.otel_logger.log_engine_initialized()

        for per_engine_loggers in self.per_engine_logger_dict.values():
            for logger in per_engine_loggers:
                logger.log_engine_initialized()
