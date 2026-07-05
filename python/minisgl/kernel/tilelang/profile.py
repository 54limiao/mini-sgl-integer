from __future__ import annotations

import atexit
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch

T = TypeVar("T")


@dataclass
class _Stat:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0


_ENABLED = os.environ.get("MINISGL_CUDA_PROFILE") == "1"
_EVERY = int(os.environ.get("MINISGL_CUDA_PROFILE_EVERY", "200"))
_STATS: dict[str, _Stat] = defaultdict(_Stat)


def reset_cuda_profile() -> None:
    _STATS.clear()


def profile_cuda(name: str, fn: Callable[..., T], *args) -> T:
    if not _ENABLED or torch.cuda.is_current_stream_capturing():
        return fn(*args)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args)
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end)
    stat = _STATS[name]
    stat.count += 1
    stat.total_ms += ms
    stat.max_ms = max(stat.max_ms, ms)
    if _EVERY > 0 and stat.count % _EVERY == 0:
        _print_summary(limit=20)
    return out


def _print_summary(limit: int = 50) -> None:
    if not _STATS:
        return
    rows = sorted(_STATS.items(), key=lambda item: item[1].total_ms, reverse=True)
    print("[MINISGL_CUDA_PROFILE] summary", flush=True)
    for name, stat in rows[:limit]:
        avg = stat.total_ms / stat.count
        print(
            f"[MINISGL_CUDA_PROFILE] {name} count={stat.count} "
            f"total_ms={stat.total_ms:.3f} avg_ms={avg:.3f} max_ms={stat.max_ms:.3f}",
            flush=True,
        )


if _ENABLED:
    atexit.register(_print_summary)
