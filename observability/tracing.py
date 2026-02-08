"""OpenTelemetry tracing configuration."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from config.settings import AppSettings


def configure_tracing(settings: AppSettings) -> None:
    """Configure OpenTelemetry tracing."""

    resource = Resource.create({"service.name": settings.app_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    if settings.otel_exporter_otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))


def instrument_fastapi(app) -> None:
    """Attach FastAPI instrumentation."""

    FastAPIInstrumentor.instrument_app(app)
