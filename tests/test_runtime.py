"""CPU 路径不执行 GPU 工作，也不能误报 CUDA 通过。"""

import json
from types import SimpleNamespace

import pytest

from llm_lab.runtime import probe


def test_cpu_probe_does_not_execute_gpu_work(monkeypatch):
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: fake_torch)
    monkeypatch.setattr(probe, "_driver_info", lambda: {"available": False})
    report = probe.collect_report(device="cpu")
    assert report["cuda_available"] is False
    assert report["gpu_test"]["status"] == "skipped"
    assert "CPU" in report["gpu_test"]["reason"]


def test_required_cuda_fails_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "collect_report", lambda *args: {"gpu_test": {"status": "skipped"}})
    destination = tmp_path / "probe.json"
    assert probe.main(["--require-cuda", "--output", str(destination)]) == 1
    assert json.loads(destination.read_text(encoding="utf-8"))["gpu_test"]["status"] == "skipped"


def test_atomic_report_write(tmp_path):
    output = tmp_path / "reports" / "environment.json"
    report = {"gpu_test": {"status": "skipped"}, "备注": "CPU 仅验证控制流程"}
    assert probe.write_report(report, output) == output
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("size,iterations", [(0, 1), (1, 0), (-1, 1)])
def test_probe_rejects_invalid_benchmark_shape(size, iterations):
    with pytest.raises(ValueError):
        probe.collect_report(matmul_size=size, iterations=iterations)
