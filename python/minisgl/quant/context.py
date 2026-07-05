from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.engine import EngineConfig

from .artifact import IntW8A8StaticArtifact


@dataclass(frozen=True)
class QuantContext:
    backend: str = "none"
    artifact: IntW8A8StaticArtifact | None = None

    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    @property
    def is_int_w8a8_static(self) -> bool:
        return self.backend == "int-w8a8-static"


_GLOBAL_QUANT_CONTEXT = QuantContext()


def create_quant_context(config: "EngineConfig") -> QuantContext:
    if config.quant_backend == "none":
        if config.quant_artifact is not None:
            raise ValueError("--quant-artifact requires --quant-backend int-w8a8-static")
        return QuantContext()
    if config.quant_backend != "int-w8a8-static":
        raise ValueError(f"Unsupported quant backend: {config.quant_backend}")
    if config.quant_artifact is None:
        raise ValueError("--quant-backend int-w8a8-static requires --quant-artifact")
    return QuantContext(
        backend=config.quant_backend,
        artifact=IntW8A8StaticArtifact.load(config.quant_artifact),
    )


def set_quant_context(ctx: QuantContext) -> None:
    global _GLOBAL_QUANT_CONTEXT
    _GLOBAL_QUANT_CONTEXT = ctx


def reset_quant_context() -> None:
    set_quant_context(QuantContext())


def get_quant_context() -> QuantContext:
    return _GLOBAL_QUANT_CONTEXT
