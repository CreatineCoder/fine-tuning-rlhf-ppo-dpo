"""Helpers for the Phase 6 VRAM/bytes-exchanged benchmark (see PLANNING.md).
Not run for real on this CPU-only dev machine — `measure_peak_vram_mb`
returns 0.0 off-CUDA; real numbers come from the RTX 5080 run."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def measure_peak_vram_mb(device: torch.device | None = None) -> float:
    """Peak CUDA memory allocated on `device` since the last reset, in MB.
    Returns 0.0 if CUDA isn't available (e.g. this CPU-only dev machine) —
    callers must not treat that as a real measurement."""
    if not torch.cuda.is_available():
        return 0.0
    device = device or torch.cuda.current_device()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def reset_vram_stats(device: torch.device | None = None) -> None:
    if torch.cuda.is_available():
        device = device or torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device)


@dataclass
class BenchmarkResult:
    baseline_vram_mb: float
    split_vram_mb: float
    baseline_bytes: int
    split_bytes: int

    @property
    def vram_reduction_pct(self) -> float:
        if self.baseline_vram_mb == 0:
            return 0.0
        return 100.0 * (self.baseline_vram_mb - self.split_vram_mb) / self.baseline_vram_mb

    @property
    def bytes_reduction_pct(self) -> float:
        if self.baseline_bytes == 0:
            return 0.0
        return 100.0 * (self.baseline_bytes - self.split_bytes) / self.baseline_bytes

    def to_dict(self) -> dict:
        return {
            "baseline_vram_mb": self.baseline_vram_mb,
            "split_vram_mb": self.split_vram_mb,
            "vram_reduction_pct": self.vram_reduction_pct,
            "baseline_bytes": self.baseline_bytes,
            "split_bytes": self.split_bytes,
            "bytes_reduction_pct": self.bytes_reduction_pct,
        }
