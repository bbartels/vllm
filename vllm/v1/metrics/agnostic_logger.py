# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

from vllm.config import SupportsMetricsInfo, VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine import FinishReason
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

        # Define the full v1 metrics in an agnostic way.
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
        labelnames = list(self._labelnames)
        model_name = self._model_name
        vcfg = self.vllm_config
        max_model_len = vcfg.model_config.max_model_len

        # Gauges: scheduler state
        g_running = self._create_gauge(
            name="vllm:num_requests_running",
            description="Number of requests in model execution batches.",
            labelnames=labelnames,
        )
        self.gauge_scheduler_running = self._bind_per_engine_gauge(g_running)

        g_waiting = self._create_gauge(
            name="vllm:num_requests_waiting",
            description="Number of requests waiting to be processed.",
            labelnames=labelnames,
        )
        self.gauge_scheduler_waiting = self._bind_per_engine_gauge(g_waiting)

        # GPU cache (deprecated and current name)
        g_gpu_cache_usage = self._create_gauge(
            name="vllm:gpu_cache_usage_perc",
            description=(
                "GPU KV-cache usage. 1 means 100 percent usage. "
                "DEPRECATED: Use vllm:kv_cache_usage_perc instead."),
            labelnames=labelnames,
        )
        self.gauge_gpu_cache_usage = self._bind_per_engine_gauge(
            g_gpu_cache_usage)

        g_kv = self._create_gauge(
            name="vllm:kv_cache_usage_perc",
            description="KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames,
        )
        self.gauge_kv_cache_usage = self._bind_per_engine_gauge(g_kv)

        # Prefix cache counters (deprecated and current names)
        c_gpu_prefix_queries = self._create_counter(
            name="vllm:gpu_prefix_cache_queries",
            description=(
                "GPU prefix cache queries, in terms of number of queried "
                "tokens. DEPRECATED: Use vllm:prefix_cache_queries instead."),
            labelnames=labelnames,
        )
        self.counter_gpu_prefix_cache_queries = self._bind_per_engine_counter(
            c_gpu_prefix_queries)

        c_gpu_prefix_hits = self._create_counter(
            name="vllm:gpu_prefix_cache_hits",
            description=(
                "GPU prefix cache hits, in terms of number of cached tokens. "
                "DEPRECATED: Use vllm:prefix_cache_hits instead."),
            labelnames=labelnames,
        )
        self.counter_gpu_prefix_cache_hits = self._bind_per_engine_counter(
            c_gpu_prefix_hits)

        c_prefix_queries = self._create_counter(
            name="vllm:prefix_cache_queries",
            description="Prefix cache queries, in terms of number of queried tokens.",
            labelnames=labelnames,
        )
        self.counter_prefix_cache_queries = self._bind_per_engine_counter(
            c_prefix_queries)

        c_prefix_hits = self._create_counter(
            name="vllm:prefix_cache_hits",
            description="Prefix cache hits, in terms of number of cached tokens.",
            labelnames=labelnames,
        )
        self.counter_prefix_cache_hits = self._bind_per_engine_counter(
            c_prefix_hits)

        # Counters
        c_preempt = self._create_counter(
            name="vllm:num_preemptions",
            description="Cumulative number of preemption from the engine.",
            labelnames=labelnames,
        )
        self.counter_num_preempted_reqs = self._bind_per_engine_counter(
            c_preempt)

        c_prompt = self._create_counter(
            name="vllm:prompt_tokens",
            description="Number of prefill tokens processed.",
            labelnames=labelnames,
        )
        self.counter_prompt_tokens = self._bind_per_engine_counter(c_prompt)

        c_gen = self._create_counter(
            name="vllm:generation_tokens",
            description="Number of generation tokens processed.",
            labelnames=labelnames,
        )
        self.counter_generation_tokens = self._bind_per_engine_counter(c_gen)

        # Request success with finish reason labels
        self.counter_request_success: dict[FinishReason, dict[
            int, _CounterHandle]] = {}
        counter_request_success_base = self._create_counter(
            name="vllm:request_success",
            description="Count of successfully processed requests.",
            labelnames=labelnames + ["finished_reason"],
        )
        for reason in FinishReason:
            self.counter_request_success[reason] = {
                idx: self._bind_counter(counter_request_success_base, {
                    "model_name": model_name,
                    "engine": str(idx),
                    "finished_reason": str(reason),
                })
                for idx in self.engine_indexes
            }

        # Histograms of counts
        h_prompt_req = self._create_histogram(
            name="vllm:request_prompt_tokens",
            description="Number of prefill tokens processed.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_num_prompt_tokens_request = self._bind_per_engine_hist(
            h_prompt_req)

        h_gen_req = self._create_histogram(
            name="vllm:request_generation_tokens",
            description="Number of generation tokens processed.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_num_generation_tokens_request = \
            self._bind_per_engine_hist(h_gen_req)

        h_iter_tokens = self._create_histogram(
            name="vllm:iteration_tokens_total",
            description="Histogram of number of tokens per engine_step.",
            labelnames=labelnames,
            buckets=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                     16384],
        )
        self.histogram_iteration_tokens = self._bind_per_engine_hist(
            h_iter_tokens)

        h_max_gen_req = self._create_histogram(
            name="vllm:request_max_num_generation_tokens",
            description=(
                "Histogram of maximum number of requested generation tokens."),
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_max_num_generation_tokens_request = \
            self._bind_per_engine_hist(h_max_gen_req)

        h_n_param = self._create_histogram(
            name="vllm:request_params_n",
            description="Histogram of the n request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )
        self.histogram_n_request = self._bind_per_engine_hist(h_n_param)

        h_max_tokens_req = self._create_histogram(
            name="vllm:request_params_max_tokens",
            description="Histogram of the max_tokens request parameter.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_max_tokens_request = self._bind_per_engine_hist(
            h_max_tokens_req)

        # Timing histograms
        h_ttft = self._create_histogram(
            name="vllm:time_to_first_token_seconds",
            description="Histogram of time to first token in seconds.",
            labelnames=labelnames,
            buckets=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25,
                     0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
                     160.0, 640.0, 2560.0],
        )
        self.histogram_time_to_first_token = self._bind_per_engine_hist(h_ttft)

        h_time_per_tok = self._create_histogram(
            name="vllm:time_per_output_token_seconds",
            description=(
                "Histogram of time per output token in seconds. "
                "DEPRECATED: Use vllm:inter_token_latency_seconds instead."),
            labelnames=labelnames,
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4,
                     0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0],
        )
        self.histogram_time_per_output_token = self._bind_per_engine_hist(
            h_time_per_tok)

        h_itl = self._create_histogram(
            name="vllm:inter_token_latency_seconds",
            description="Histogram of inter-token latency in seconds.",
            labelnames=labelnames,
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4,
                     0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0],
        )
        self.histogram_inter_token_latency = self._bind_per_engine_hist(h_itl)

        request_latency_buckets = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0,
                                   10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                                   120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0]
        h_e2e = self._create_histogram(
            name="vllm:e2e_request_latency_seconds",
            description="Histogram of e2e request latency in seconds.",
            labelnames=labelnames,
            buckets=request_latency_buckets,
        )
        self.histogram_e2e_time_request = self._bind_per_engine_hist(h_e2e)

        h_queue = self._create_histogram(
            name="vllm:request_queue_time_seconds",
            description=(
                "Histogram of time spent in WAITING phase for request."),
            labelnames=labelnames,
            buckets=request_latency_buckets,
        )
        self.histogram_queue_time_request = self._bind_per_engine_hist(h_queue)

        h_infer = self._create_histogram(
            name="vllm:request_inference_time_seconds",
            description=(
                "Histogram of time spent in RUNNING phase for request."),
            labelnames=labelnames,
            buckets=request_latency_buckets,
        )
        self.histogram_inference_time_request = self._bind_per_engine_hist(
            h_infer)

        h_prefill = self._create_histogram(
            name="vllm:request_prefill_time_seconds",
            description=(
                "Histogram of time spent in PREFILL phase for request."),
            labelnames=labelnames,
            buckets=request_latency_buckets,
        )
        self.histogram_prefill_time_request = self._bind_per_engine_hist(
            h_prefill)

        h_decode = self._create_histogram(
            name="vllm:request_decode_time_seconds",
            description=(
                "Histogram of time spent in DECODE phase for request."),
            labelnames=labelnames,
            buckets=request_latency_buckets,
        )
        self.histogram_decode_time_request = self._bind_per_engine_hist(
            h_decode)

    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0) -> None:
        if scheduler_stats is not None:
            self.gauge_scheduler_running[engine_idx].set(
                scheduler_stats.num_running_reqs)
            self.gauge_scheduler_waiting[engine_idx].set(
                scheduler_stats.num_waiting_reqs)

            self.gauge_gpu_cache_usage[engine_idx].set(
                scheduler_stats.kv_cache_usage)
            self.gauge_kv_cache_usage[engine_idx].set(
                scheduler_stats.kv_cache_usage)

            self.counter_gpu_prefix_cache_queries[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.queries)
            self.counter_gpu_prefix_cache_hits[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.hits)

            self.counter_prefix_cache_queries[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.queries)
            self.counter_prefix_cache_hits[engine_idx].inc(
                scheduler_stats.prefix_cache_stats.hits)

            self._observe_spec_decoding(scheduler_stats)

        if iteration_stats is None:
            return

        self.counter_num_preempted_reqs[engine_idx].inc(
            iteration_stats.num_preempted_reqs)
        self.counter_prompt_tokens[engine_idx].inc(
            iteration_stats.num_prompt_tokens)
        self.counter_generation_tokens[engine_idx].inc(
            iteration_stats.num_generation_tokens)
        self.histogram_iteration_tokens[engine_idx].observe(
            iteration_stats.num_prompt_tokens +
            iteration_stats.num_generation_tokens)

        for max_gen_tokens in iteration_stats.max_num_generation_tokens_iter:
            self.histogram_max_num_generation_tokens_request[
                engine_idx].observe(max_gen_tokens)
        for n_param in iteration_stats.n_params_iter:
            self.histogram_n_request[engine_idx].observe(n_param)
        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.histogram_time_to_first_token[engine_idx].observe(ttft)
        for itl in iteration_stats.inter_token_latencies_iter:
            self.histogram_inter_token_latency[engine_idx].observe(itl)
            self.histogram_time_per_output_token[engine_idx].observe(itl)

        for finished_request in iteration_stats.finished_requests:
            self.counter_request_success[
                finished_request.finish_reason][engine_idx].inc()
            self.histogram_e2e_time_request[engine_idx].observe(
                finished_request.e2e_latency)
            self.histogram_queue_time_request[engine_idx].observe(
                finished_request.queued_time)
            self.histogram_prefill_time_request[engine_idx].observe(
                finished_request.prefill_time)
            self.histogram_inference_time_request[engine_idx].observe(
                finished_request.inference_time)
            self.histogram_decode_time_request[engine_idx].observe(
                finished_request.decode_time)
            self.histogram_num_prompt_tokens_request[engine_idx].observe(
                finished_request.num_prompt_tokens)
            self.histogram_num_generation_tokens_request[engine_idx].observe(
                finished_request.num_generation_tokens)
            if finished_request.max_tokens_param:
                self.histogram_max_tokens_request[engine_idx].observe(
                    finished_request.max_tokens_param)

        self._record_lora_info(iteration_stats)

    def log_engine_initialized(self) -> None:
        # Hook to export static info on engine initialization
        self._log_info_cache_config(self.vllm_config.cache_config)

    # --- Provider-specific extension hooks (default no-ops) ---

    def _observe_spec_decoding(self, scheduler_stats: SchedulerStats) -> None:
        return

    def _record_lora_info(self, iteration_stats: IterationStats) -> None:
        return

    def _log_info_cache_config(self, config_obj: SupportsMetricsInfo) -> None:
        return


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

    # Spec decoding metrics (prometheus-specific)
    def __init__(self, vllm_config: VllmConfig,
                 engine_indexes: Optional[list[int]] = None):
        from vllm.v1.spec_decode.metrics import SpecDecodingProm
        super().__init__(vllm_config, engine_indexes)
        # Spec decoding metrics are only supported with a single engine when
        # speculative decoding is enabled (following previous behavior).
        self._spec_decoding = None
        if (len(self.engine_indexes) > 0 and
                vllm_config.speculative_config is not None):
            labelnames = ["model_name", "engine"]
            labelvalues = [
                vllm_config.model_config.served_model_name,
                str(self.engine_indexes[0])
            ]
            self._spec_decoding = SpecDecodingProm(
                vllm_config.speculative_config, labelnames, labelvalues)

        # LoRA info gauge (prometheus supports set_to_current_time)
        self._gauge_lora_info = None
        if vllm_config.lora_config is not None:
            if len(self.engine_indexes) > 1:
                raise NotImplementedError("LoRA in DP mode is not supported.")
            import prometheus_client  # type: ignore
            self._labelname_max_lora = "max_lora"
            self._labelname_waiting_lora_adapters = "waiting_lora_adapters"
            self._labelname_running_lora_adapters = "running_lora_adapters"
            self._max_lora = vllm_config.lora_config.max_loras
            self._gauge_lora_info = prometheus_client.Gauge(
                name="vllm:lora_requests_info",
                documentation="Running stats on lora requests.",
                multiprocess_mode="sum",
                labelnames=[
                    self._labelname_max_lora,
                    self._labelname_waiting_lora_adapters,
                    self._labelname_running_lora_adapters,
                ],
            )

    def _observe_spec_decoding(self, scheduler_stats: SchedulerStats) -> None:
        if self._spec_decoding is not None and \
                scheduler_stats.spec_decoding_stats is not None:
            self._spec_decoding.observe(scheduler_stats.spec_decoding_stats)

    def _record_lora_info(self, iteration_stats: IterationStats) -> None:
        if self._gauge_lora_info is None:
            return
        running_lora_adapters = \
            ",".join(iteration_stats.running_lora_adapters.keys())
        waiting_lora_adapters = \
            ",".join(iteration_stats.waiting_lora_adapters.keys())
        labels = {
            self._labelname_running_lora_adapters: running_lora_adapters,
            self._labelname_waiting_lora_adapters: waiting_lora_adapters,
            self._labelname_max_lora: self._max_lora,
        }
        self._gauge_lora_info.labels(**labels).set_to_current_time()

    def _log_info_cache_config(self, config_obj: SupportsMetricsInfo) -> None:
        metrics_info = config_obj.metrics_info()
        metrics_info["engine"] = ""
        import prometheus_client  # type: ignore
        info_gauge = prometheus_client.Gauge(
            name="vllm:cache_config_info",
            documentation="Information of the LLMEngine CacheConfig",
            multiprocess_mode="mostrecent",
            labelnames=list(metrics_info.keys()),
        )
        for engine_index in self.engine_indexes:
            labels = config_obj.metrics_info()
            labels["engine"] = str(engine_index)
            info_gauge.labels(**labels).set(1)


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

    def _sanitize_name(self, name: str) -> str:
        # OTel instrument names must be [a-zA-Z_][a-zA-Z0-9_]*
        # Replace ':' with '_' to preserve compatibility
        return name.replace(":", "_")

    def _create_counter(self, name: str, description: str,
                        labelnames: Iterable[str]) -> Any:
        if not self._otel_available:
            return (None, "counter")
        return self._meter.create_counter(name=self._sanitize_name(name),
                                          description=description)

    def _create_gauge(self, name: str, description: str,
                      labelnames: Iterable[str]) -> Any:
        if not self._otel_available:
            return (None, "gauge")
        # Emulate a gauge using UpDownCounter and deltas in set()
        return self._meter.create_up_down_counter(
            name=self._sanitize_name(name), description=description)

    def _create_histogram(self, name: str, description: str,
                          labelnames: Iterable[str],
                          buckets: Optional[Iterable[float]] = None) -> Any:
        if not self._otel_available:
            return (None, "histogram")
        return self._meter.create_histogram(name=self._sanitize_name(name),
                                            description=description)

    def _record_lora_info(self, iteration_stats: IterationStats) -> None:
        # Emit current LoRA info as a gauge-like value set to 1 with labels
        if not self._otel_available:
            return
        # Create metric lazily to avoid creating when not needed
        if not hasattr(self, "_lora_metric"):
            self._lora_metric = self._create_gauge(
                name="vllm:lora_requests_info",
                description="Running stats on lora requests.",
                labelnames=[
                    "max_lora", "waiting_lora_adapters",
                    "running_lora_adapters"
                ],
            )
        running = ",".join(iteration_stats.running_lora_adapters.keys())
        waiting = ",".join(iteration_stats.waiting_lora_adapters.keys())
        labels = {
            "max_lora": str(getattr(getattr(self.vllm_config, "lora_config",
                                             None), "max_loras", "")),
            "waiting_lora_adapters": waiting,
            "running_lora_adapters": running,
        }
        handle = self._bind_gauge(self._lora_metric, labels)
        handle.set(1)

    def _log_info_cache_config(self, config_obj: SupportsMetricsInfo) -> None:
        # Represent info as a gauge-like value set to 1 with attributes
        if not self._otel_available:
            return
        metrics_info = config_obj.metrics_info()
        metrics_info["engine"] = ""
        # Lazy create
        if not hasattr(self, "_info_metric"):
            self._info_metric = self._create_gauge(
                name="vllm:cache_config_info",
                description="Information of the LLMEngine CacheConfig",
                labelnames=list(metrics_info.keys()),
            )
        for engine_index in self.engine_indexes:
            labels = config_obj.metrics_info()
            labels["engine"] = str(engine_index)
            self._bind_gauge(self._info_metric, labels).set(1)


# --- Bucket helpers ---

def build_buckets(mantissa_lst: list[int], max_value: int) -> list[int]:
    exponent = 0
    buckets: list[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> list[int]:
    return build_buckets([1, 2, 5], max_value)

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
