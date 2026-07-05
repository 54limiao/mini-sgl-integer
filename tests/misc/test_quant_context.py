import json

import pytest

from minisgl.engine import EngineConfig
from minisgl.quant import create_quant_context
from minisgl.kernel.tilelang.profile import profile_cuda


def test_quant_artifact_requires_backend():
    cfg = EngineConfig(
        model_path="dummy",
        tp_info=None,  # type: ignore[arg-type]
        dtype=None,  # type: ignore[arg-type]
        quant_artifact="/tmp/missing",
    )
    with pytest.raises(ValueError, match="requires --quant-backend"):
        create_quant_context(cfg)


def test_int_w8a8_static_context_loads_artifact(tmp_path):
    (tmp_path / "int_w8a8_static_config.json").write_text(
        json.dumps({"format": "int-w8a8-static", "version": 1}),
        encoding="utf-8",
    )
    cfg = EngineConfig(
        model_path="dummy",
        tp_info=None,  # type: ignore[arg-type]
        dtype=None,  # type: ignore[arg-type]
        quant_backend="int-w8a8-static",
        quant_artifact=str(tmp_path),
    )

    ctx = create_quant_context(cfg)
    assert ctx.is_int_w8a8_static
    assert ctx.artifact is not None


def test_profile_cuda_is_passthrough_by_default():
    assert profile_cuda("unit", lambda x, y: x + y, 1, 2) == 3
