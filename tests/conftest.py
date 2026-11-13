"""不联网的可选 SDK 夹具；只有显式使用的测试才启用。"""
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.fixture
def fake_wandb(monkeypatch):
    sdk = ModuleType("wandb")
    sdk.__version__ = "0.30.0"
    sdk.calls = []
    sdk.Settings = lambda **kwargs: SimpleNamespace(**kwargs)

    def init(**kwargs):
        run = SimpleNamespace(entity="fixture-owner", url="https://wandb.ai/fixture/run", logs=[], definitions=[], finishes=[])
        run.log = lambda data, **options: run.logs.append((data, options))
        run.define_metric = lambda *args, **options: run.definitions.append((args, options))
        run.finish = lambda **options: run.finishes.append(options)
        sdk.calls.append((kwargs, run))
        return run

    sdk.init = init
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    return sdk
