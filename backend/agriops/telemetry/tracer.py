# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps Telemetry and OpenTelemetry Instrumentation Provider
"""

import time
import logging
import functools
from typing import Any, Callable, Dict, List

logger = logging.getLogger("AgriOps.Telemetry")

# Attempt importing standard OpenTelemetry API
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class TelemetryManager:
    def __init__(self):
        self.tracer = None
        self.traces_log: List[Dict[str, Any]] = []
        if OTEL_AVAILABLE:
            try:
                self.tracer = trace.get_tracer("agriops.tracer")
            except Exception:
                pass

    def trace_span(self, name: str):
        """
        Decorator to instrument functions with tracing span logging and telemetry instrumentation.
        """

        def decorator(func: Callable[..., Any]):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                trace_id = f"tr-{int(start_time * 1000)}"
                logger.info(f"[TRACE START] {name} | Trace ID: {trace_id}")

                # If OpenTelemetry API is configured
                if self.tracer:
                    with self.tracer.start_as_current_span(name) as span:
                        try:
                            result = await func(*args, **kwargs)
                            latency = (time.perf_counter() - start_time) * 1000
                            self._log_trace(name, latency, "success")
                            span.set_status(Status(StatusCode.OK))
                            return result
                        except Exception as e:
                            latency = (time.perf_counter() - start_time) * 1000
                            self._log_trace(name, latency, "error", details=str(e))
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                            raise e
                else:
                    try:
                        result = await func(*args, **kwargs)
                        latency = (time.perf_counter() - start_time) * 1000
                        logger.info(f"[TRACE END] {name} | Latency: {latency:.2f}ms")
                        self._log_trace(name, latency, "success")
                        return result
                    except Exception as e:
                        latency = (time.perf_counter() - start_time) * 1000
                        logger.error(
                            f"[TRACE ERROR] {name} failed: {e} | Latency: {latency:.2f}ms"
                        )
                        self._log_trace(name, latency, "error", details=str(e))
                        raise e

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                trace_id = f"tr-{int(start_time * 1000)}"
                logger.info(f"[TRACE START] {name} | Trace ID: {trace_id}")

                if self.tracer:
                    with self.tracer.start_as_current_span(name) as span:
                        try:
                            result = func(*args, **kwargs)
                            latency = (time.perf_counter() - start_time) * 1000
                            self._log_trace(name, latency, "success")
                            span.set_status(Status(StatusCode.OK))
                            return result
                        except Exception as e:
                            latency = (time.perf_counter() - start_time) * 1000
                            self._log_trace(name, latency, "error", details=str(e))
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            span.record_exception(e)
                            raise e
                else:
                    try:
                        result = func(*args, **kwargs)
                        latency = (time.perf_counter() - start_time) * 1000
                        logger.info(f"[TRACE END] {name} | Latency: {latency:.2f}ms")
                        self._log_trace(name, latency, "success")
                        return result
                    except Exception as e:
                        latency = (time.perf_counter() - start_time) * 1000
                        logger.error(
                            f"[TRACE ERROR] {name} failed: {e} | Latency: {latency:.2f}ms"
                        )
                        self._log_trace(name, latency, "error", details=str(e))
                        raise e

            import inspect

            if inspect.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    def _log_trace(self, name: str, latency_ms: float, status: str, details: str = ""):
        self.traces_log.append(
            {
                "name": name,
                "latency_ms": round(latency_ms, 2),
                "status": status,
                "details": details,
                "timestamp": time.time(),
            }
        )
        if len(self.traces_log) > 200:
            self.traces_log.pop(0)

    def get_traces(self) -> List[Dict[str, Any]]:
        return self.traces_log


telemetry = TelemetryManager()
trace_span = telemetry.trace_span
