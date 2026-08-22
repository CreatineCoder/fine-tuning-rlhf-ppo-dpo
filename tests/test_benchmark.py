import torch

from rlhf_scratch.distributed import BenchmarkResult, ByteCounter, measure_peak_vram_mb


def test_measure_peak_vram_mb_returns_zero_off_cuda():
    if torch.cuda.is_available():
        return  # this dev machine has no CUDA; real behavior tested on the RTX 5080
    assert measure_peak_vram_mb() == 0.0


def test_byte_counter_accumulates_across_multiple_tensors():
    counter = ByteCounter()
    counter.count(torch.zeros(10, dtype=torch.float32))  # 40 bytes
    counter.count(torch.zeros(5, dtype=torch.float64))  # 40 bytes
    assert counter.total_bytes == 80
    counter.reset()
    assert counter.total_bytes == 0


def test_benchmark_result_computes_reduction_percentages():
    result = BenchmarkResult(baseline_vram_mb=1000.0, split_vram_mb=600.0, baseline_bytes=5000, split_bytes=1000)
    assert result.vram_reduction_pct == 40.0
    assert result.bytes_reduction_pct == 80.0

    d = result.to_dict()
    assert d["vram_reduction_pct"] == 40.0
    assert d["bytes_reduction_pct"] == 80.0


def test_benchmark_result_handles_zero_baseline():
    result = BenchmarkResult(baseline_vram_mb=0.0, split_vram_mb=0.0, baseline_bytes=0, split_bytes=0)
    assert result.vram_reduction_pct == 0.0
    assert result.bytes_reduction_pct == 0.0
