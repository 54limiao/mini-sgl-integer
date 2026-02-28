import importlib
import os

from .config import ModelConfig

# Environment variable to control integer mode
# Set MINISGL_INTEGER_MODE=1 to use integer-only RMSNorm
_INTEGER_MODE = os.environ.get("MINISGL_INTEGER_MODE", "0") == "1"

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
}

# Integer mode model registry
_INTEGER_MODEL_REGISTRY = {
    "Qwen3ForCausalLM": (".integer.qwen3_integer", "Qwen3ForCausalLMInteger"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    # Use integer model if enabled and available
    if _INTEGER_MODE and model_architecture in _INTEGER_MODEL_REGISTRY:
        module_path, class_name = _INTEGER_MODEL_REGISTRY[model_architecture]
        module = importlib.import_module(module_path, package=__package__)
        model_cls = getattr(module, class_name)
        return model_cls(model_config)

    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]
