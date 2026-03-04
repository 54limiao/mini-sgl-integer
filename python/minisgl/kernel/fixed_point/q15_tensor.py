from __future__ import annotations

from dataclasses import dataclass

import torch

FIXED_POINT_SCALE = 1 << 16
Q15_MIN = -(1 << 31)
Q15_MAX = (1 << 31) - 1


def assert_q15(x: torch.Tensor, name: str = "tensor") -> None:
    if x.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32 (Q15.16), got {x.dtype}")


def to_fixed(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        torch.round(x.to(torch.float32) * FIXED_POINT_SCALE),
        Q15_MIN,
        Q15_MAX,
    ).to(torch.int32)


def from_fixed(x: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    assert_q15(x, "x")
    return (x.to(torch.float32) / FIXED_POINT_SCALE).to(dtype)


@dataclass(frozen=True)
class Q15Tensor:
    data: torch.Tensor

    def __post_init__(self) -> None:
        assert_q15(self.data, "Q15Tensor.data")

    @staticmethod
    def from_float(x: torch.Tensor) -> "Q15Tensor":
        return Q15Tensor(to_fixed(x))

    def to_float(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return from_fixed(self.data, dtype=dtype)

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def device(self) -> torch.device:
        return self.data.device
