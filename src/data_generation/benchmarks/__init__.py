"""Benchmark catalog and manifest discovery."""

from data_generation.benchmarks.registry import (
    BenchmarkDefinition,
    list_benchmarks,
    resolve_benchmark,
)

__all__ = ["BenchmarkDefinition", "list_benchmarks", "resolve_benchmark"]
