# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Optional, Union

from typing import Dict

from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.agnostic_logger import MetricsProvider, _CounterHandle, \
    _GaugeHandle, _HistogramHandle

try:
    from ray.util import metrics as ray_metrics
    from ray.util.metrics import Metric
except ImportError:
    ray_metrics = None


class _RayCounterHandle(_CounterHandle):
    def __init__(self, metric: Metric, tags: Dict[str, str]):
        self._metric = metric
        self._tags = tags

    def inc(self, value: float = 1.0) -> None:
        if value == 0:
            return
        self._metric.set_default_tags(self._tags)
        self._metric.inc(value)


class _RayGaugeHandle(_GaugeHandle):
    def __init__(self, metric: Metric, tags: Dict[str, str]):
        self._metric = metric
        self._tags = tags

    def set(self, value: float) -> None:
        self._metric.set_default_tags(self._tags)
        self._metric.set(value)


class _RayHistogramHandle(_HistogramHandle):
    def __init__(self, metric: Metric, tags: Dict[str, str]):
        self._metric = metric
        self._tags = tags

    def observe(self, value: float) -> None:
        self._metric.set_default_tags(self._tags)
        self._metric.observe(value)


class _RayMetricsProvider(MetricsProvider):
    def __init__(self) -> None:
        if ray_metrics is None:
            raise ImportError("Ray metrics provider requires Ray installed.")

    def create_counter(self, name: str, description: str,
                       labelnames: Optional[list[str]]) -> Metric:
        tag_keys = tuple(labelnames) if labelnames else None
        return ray_metrics.Counter(name=name, description=description,
                                   tag_keys=tag_keys)

    def create_gauge(self, name: str, description: str,
                     labelnames: Optional[list[str]]) -> Metric:
        tag_keys = tuple(labelnames) if labelnames else None
        return ray_metrics.Gauge(name=name, description=description,
                                 tag_keys=tag_keys)

    def create_histogram(self, name: str, description: str,
                         labelnames: Optional[list[str]],
                         buckets: Optional[list[float]] = None) -> Metric:
        tag_keys = tuple(labelnames) if labelnames else None
        boundaries = buckets if buckets else []
        return ray_metrics.Histogram(name=name, description=description,
                                     tag_keys=tag_keys,
                                     boundaries=boundaries)

    def bind_labels_counter(self, metric: Metric,
                            labels: Dict[str, str]) -> _CounterHandle:
        return _RayCounterHandle(metric, labels)

    def bind_labels_gauge(self, metric: Metric,
                          labels: Dict[str, str]) -> _GaugeHandle:
        return _RayGaugeHandle(metric, labels)

    def bind_labels_histogram(self, metric: Metric,
                              labels: Dict[str, str]) -> _HistogramHandle:
        return _RayHistogramHandle(metric, labels)


class RayPrometheusStatLogger(PrometheusStatLogger):
    """PrometheusStatLogger backed by Ray metrics provider."""

    def _build_provider(self, vllm_config):
        return _RayMetricsProvider()
