from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class IntW8A8StaticArtifact:
    path: Path
    config: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "IntW8A8StaticArtifact":
        artifact_path = Path(path)
        config_path = artifact_path / "int_w8a8_static_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing int-w8a8-static config: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(path=artifact_path, config=config)

    def tensor_path(self, name: str) -> Path:
        safe_name = name.replace(".", "__")
        return self.path / "tensors" / f"{safe_name}.safetensors"

    def load_tensor(self, name: str, device: torch.device | str) -> torch.Tensor:
        path = self.tensor_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing quant tensor '{name}': {path}")
        tensors = load_file(str(path), device=str(device))
        if name in tensors:
            return tensors[name]
        if len(tensors) == 1:
            return next(iter(tensors.values()))
        raise KeyError(f"Tensor file {path} does not contain '{name}'")

    def optional_tensor(self, name: str, device: torch.device | str) -> torch.Tensor | None:
        path = self.tensor_path(name)
        if not path.is_file():
            return None
        return self.load_tensor(name, device)
