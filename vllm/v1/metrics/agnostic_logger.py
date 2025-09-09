# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.metrics.stats import IterationStats, SchedulerStats
from vllm.v1.metrics.base import StatLoggerBase

logger = init_logger(__name__)


class _CounterHandle(ABC):
    @abstractmethod
    def inc(self, value: float = 1.0) -> None:  # prometheus parity
        ...


class _GaugeHandle(ABC):
    @abstractmethod
    def set(self, value: float) -> None:
        ...


class _HistogramHandle(ABC):
    @abstractmethod
    def observe(self, value: float) -> None:
        ...


class AgnosticMetricsLogger(StatLoggerBase, ABC):
    """Provider-agnostic metrics recorder.

    This class defines metric names and recording logic without depending on a
    specific backend (Prometheus, OpenTelemetry, etc.). Concrete subclasses
    implement instrument creation/binding for their provider.
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 engine_indexes: Optional[list[int]] = None):
        if engine_indexes is None or len(engine_indexes) == 0:
            engine_indexes = [0]
        self.engine_indexes = engine_indexes
        self.vllm_config = vllm_config

        # Common labels for all metrics produced here
        self._labelnames = ("model_name", "engine")
        self._model_name = vllm_config.model_config.served_model_name

        # Define a minimal, useful set of metrics in an agnostic way.
        self._init_metrics()

    # Instrument creation/binding to be implemented by subclasses
    @abstractmethod
    def _create_counter(self, name: str, description: str,
                        labelnames: Iterable[str]) -> Any: ...

    @abstractmethod
    def _create_gauge(self, name: str, description: str,
                      labelnames: Iterable[str]) -> Any: ...

    @abstractmethod
    def _create_histogram(self, name: str, description: str,
                          labelnames: Iterable[str],
                          buckets: Optional[Iterable[float]] = None) -> Any: ...

    @abstractmethod
    def _bind_counter(self, metric: Any,
                      labels: Dict[str, str]) -> _CounterHandle: ...

    @abstractmethod
    def _bind_gauge(self, metric: Any,
                    labels: Dict[str, str]) -> _GaugeHandle: ...

    @abstractmethod
    def _bind_histogram(self, metric: Any,
                        labels: Dict[str, str]) -> _HistogramHandle: ...

    def _bind_per_engine_counter(self, metric: Any
                                 ) -> dict[int, _CounterHandle]:
        handles: dict[int, _CounterHandle] = {}
        for idx in self.engine_indexes:
            handles[idx] = self._bind_counter(metric, {
                "model_name": self._model_name,
                "engine": str(idx),
            })
        return handles

    def _bind_per_engine_gauge(self, metric: Any) -> dict[int, _GaugeHandle]:
        handles: dict[int, _GaugeHandle] = {}
        for idx in self.engine_indexes:
            handles[idx] = self._bind_gauge(metric, {
                "model_name": self._model_name,
                "engine": str(idx),
            })
        return handles

    def _bind_per_engine_hist(self, metric: Any
                               ) -> dict[int, _HistogramHandle]:
        handles: dict[int, _HistogramHandle] = {}
        for idx in self.engine_indexes:
            handles[idx] = self._bind_histogram(metric, {
                "model_name": self._model_name,
                "engine": str(idx),
            })
        return handles

    def _init_metrics(self) -> None:
        # Gauges
        g_running = self._create_gauge(
            name="vllm:num_requests_running",
            description="Number of requests in execution batches.",
            labelnames=self._labelnames,
        )
        self.gauge_running = self._bind_per_engine_gauge(g_running)

        g_waiting = self._create_gauge(
            name="vllm:num_requests_waiting",
            description="Number of requests waiting to be processed.",
            labelnames=self._labelnames,
        )
        self.gauge_waiting = self._bind_per_engine_gauge(g_waiting)

        g_kv = self._create_gauge(
            name="vllm:kv_cache_usage_perc",
            description="KV-cache usage. 1 means 100 percent usage.",
            labelnames=self._labelnames,
        )
        self.gauge_kv_cache = self._bind_per_engine_gauge(g_kv)

        # Counters
        c_preempt = self._create_counter(
            name="vllm:num_preemptions",
            description="Cumulative number of preemptions from the engine.",
            labelnames=self._labelnames,
        )
        self.counter_preempt = self._bind_per_engine_counter(c_preempt)

        c_prompt = self._create_counter(
            name="vllm:prompt_tokens",
            description="Number of prefill tokens processed.",
            labelnames=self._labelnames,
        )
        self.counter_prompt = self._bind_per_engine_counter(c_prompt)

        c_gen = self._create_counter(
            name="vllm:generation_tokens",
            description="Number of generation tokens processed.",
            labelnames=self._labelnames,
        )
        self.counter_gen = self._bind_per_engine_counter(c_gen)

        # Histograms (no custom buckets for simplicity)
        h_ttft = self._create_histogram(
            name="vllm:time_to_first_token_seconds",
            description="Time to first token per request (seconds).",
            labelnames=self._labelnames,
        )
        self.hist_ttft = self._bind_per_engine_hist(h_ttft)

        h_itl = self._create_histogram(
            name="vllm:inter_token_latency_seconds",
            description="Inter-token latency during decode (seconds).",
            labelnames=self._labelnames,
        )
        self.hist_itl = self._bind_per_engine_hist(h_itl)

        h_e2e = self._create_histogram(
            name="vllm:request_e2e_time_seconds",
            description="End-to-end latency per finished request (seconds).",
            labelnames=self._labelnames,
        )
        self.hist_e2e = self._bind_per_engine_hist(h_e2e)

    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0) -> None:
        if scheduler_stats is not None:
            self.gauge_running[engine_idx].set(scheduler_stats.num_running_reqs)
            self.gauge_waiting[engine_idx].set(scheduler_stats.num_waiting_reqs)
            self.gauge_kv_cache[engine_idx].set(scheduler_stats.kv_cache_usage)

        if iteration_stats is None:
            return

        # Counters
        if iteration_stats.num_preempted_reqs:
            self.counter_preempt[engine_idx].inc(
                iteration_stats.num_preempted_reqs)
        if iteration_stats.num_prompt_tokens:
            self.counter_prompt[engine_idx].inc(
                iteration_stats.num_prompt_tokens)
        if iteration_stats.num_generation_tokens:
            self.counter_gen[engine_idx].inc(
                iteration_stats.num_generation_tokens)

        # Histograms from iteration-level stats
        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.hist_ttft[engine_idx].observe(ttft)
        for itl in iteration_stats.inter_token_latencies_iter:
            self.hist_itl[engine_idx].observe(itl)

        # Histograms from finished requests
        for finished in iteration_stats.finished_requests:
            self.hist_e2e[engine_idx].observe(finished.e2e_latency)

    def log_engine_initialized(self) -> None:
        # Agnostic base has no-op initialization hook. Providers may override
        # separately by exposing their own APIs if needed.
        pass


# --- Prometheus implementation ---


class _PromCounterHandle(_CounterHandle):
    def __init__(self, child: Any):
        self._child = child

    def inc(self, value: float = 1.0) -> None:
        self._child.inc(value)


class _PromGaugeHandle(_GaugeHandle):
    def __init__(self, child: Any):
        self._child = child

    def set(self, value: float) -> None:
        self._child.set(value)


class _PromHistogramHandle(_HistogramHandle):
    def __init__(self, child: Any):
        self._child = child

    def observe(self, value: float) -> None:
        self._child.observe(value)


class PrometheusAgnosticMetricsLogger(AgnosticMetricsLogger):
    """Agnostic metrics logger backed by Prometheus instruments."""

    def _create_counter(self, name: str, description: str,
                        labelnames: Iterable[str]) -> Any:
        import prometheus_client  # type: ignore
        return prometheus_client.Counter(name=name,
                                         documentation=description,
                                         labelnames=list(labelnames))

    def _create_gauge(self, name: str, description: str,
                      labelnames: Iterable[str]) -> Any:
        import prometheus_client  # type: ignore
        return prometheus_client.Gauge(name=name,
                                       documentation=description,
                                       multiprocess_mode="mostrecent",
                                       labelnames=list(labelnames))

    def _create_histogram(self, name: str, description: str,
                          labelnames: Iterable[str],
                          buckets: Optional[Iterable[float]] = None) -> Any:
        import prometheus_client  # type: ignore
        kwargs: Dict[str, Any] = {
            "name": name,
            "documentation": description,
            "labelnames": list(labelnames),
        }
        if buckets is not None:
            kwargs["buckets"] = list(buckets)
        return prometheus_client.Histogram(**kwargs)

    def _bind_counter(self, metric: Any,
                      labels: Dict[str, str]) -> _CounterHandle:
        return _PromCounterHandle(metric.labels(**labels))

    def _bind_gauge(self, metric: Any, labels: Dict[str, str]) -> _GaugeHandle:
        return _PromGaugeHandle(metric.labels(**labels))

    def _bind_histogram(self, metric: Any,
                        labels: Dict[str, str]) -> _HistogramHandle:
        return _PromHistogramHandle(metric.labels(**labels))


# --- OpenTelemetry implementation ---


class _OtelCounterHandle(_CounterHandle):
    def __init__(self, counter: Any, attributes: Dict[str, str]):
        self._counter = counter
        self._attributes = attributes

    def inc(self, value: float = 1.0) -> None:
        # OTel counters are monotonic; use add()
        self._counter.add(float(value), attributes=self._attributes)


class _OtelUpDownGaugeHandle(_GaugeHandle):
    def __init__(self, updown: Any, attributes: Dict[str, str]):
        self._updown = updown
        self._attributes = attributes
        self._last_value = 0.0

    def set(self, value: float) -> None:
        # Emulate a gauge using an UpDownCounter by applying a delta
        delta = float(value) - self._last_value
        if delta != 0.0:
            self._updown.add(delta, attributes=self._attributes)
            self._last_value = float(value)


class _OtelHistogramHandle(_HistogramHandle):
    def __init__(self, histogram: Any, attributes: Dict[str, str]):
        self._histogram = histogram
        self._attributes = attributes

    def observe(self, value: float) -> None:
        self._histogram.record(float(value), attributes=self._attributes)


class OpenTelemetryAgnosticMetricsLogger(AgnosticMetricsLogger):
    """Agnostic metrics logger backed by OpenTelemetry instruments."""

    def __init__(self, vllm_config: VllmConfig,
                 engine_indexes: Optional[list[int]] = None):
        # Initialize OTel meter (if available)
        try:
            from opentelemetry import metrics  # type: ignore
            self._meter = metrics.get_meter("vllm")
            self._otel_available = True
        except Exception as e:  # pragma: no cover - import guard
            logger.warning("OpenTelemetry not available; metrics no-ops: %s",
                           e)
            self._meter = None
            self._otel_available = False
        super().__init__(vllm_config, engine_indexes)

    def _create_counter(self, name: str, description: str,
                        labelnames: Iterable[str]) -> Any:
        if not self._otel_available:
            return (None, "counter")
        return self._meter.create_counter(name=name, description=description)

    def _create_gauge(self, name: str, description: str,
                      labelnames: Iterable[str]) -> Any:
        if not self._otel_available:
            return (None, "gauge")
        # Emulate a gauge using UpDownCounter and deltas in set()
        return self._meter.create_up_down_counter(name=name,
                                                  description=description)

    def _create_histogram(self, name: str, description: str,
                          labelnames: Iterable[str],
                          buckets: Optional[Iterable[float]] = None) -> Any:
        if not self._otel_available:
            return (None, "histogram")
        return self._meter.create_histogram(name=name,
                                            description=description)

    def _bind_counter(self, metric: Any,
                      labels: Dict[str, str]) -> _CounterHandle:
        if isinstance(metric, tuple):
            return _OtelCounterHandle(_NoopCounter(), labels)
        return _OtelCounterHandle(metric, labels)

    def _bind_gauge(self, metric: Any, labels: Dict[str, str]) -> _GaugeHandle:
        if isinstance(metric, tuple):
            return _NoopGaugeHandle()
        return _OtelUpDownGaugeHandle(metric, labels)

    def _bind_histogram(self, metric: Any,
                        labels: Dict[str, str]) -> _HistogramHandle:
        if isinstance(metric, tuple):
            return _NoopHistogramHandle()
        return _OtelHistogramHandle(metric, labels)


# --- No-op handles (used if OTel is unavailable) ---


class _NoopCounter:
    def add(self, value: float, attributes: Dict[str, str]):
        pass


class _NoopGaugeHandle(_GaugeHandle):
    def set(self, value: float) -> None:
        pass


class _NoopHistogramHandle(_HistogramHandle):
    def observe(self, value: float) -> None:
        pass
