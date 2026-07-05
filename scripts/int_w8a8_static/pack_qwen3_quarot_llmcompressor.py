#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from compressed_tensors.transform.utils.hadamard import random_hadamard_matrix
from compressed_tensors.transform import TransformArgs, TransformConfig, TransformScheme, apply_transform_config
from datasets import Dataset
from llmcompressor.modeling import center_embeddings, fuse_norm_linears
from llmcompressor.utils import untie_word_embeddings
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    if (parent / "examples/qwen3_int_only/utils/prepack.py").is_file():
        sys.path.insert(0, str(parent))
        break

from examples.qwen3_int_only.utils import (  # noqa: E402
    ROTATE_SEED,
    Qwen3Config,
    per_channel_i8_weight,
    per_tensor_i8_weight,
    q15_16,
    fast_hadamard,
    rmsnorm_torch,
    rotate_head_input,
    rotate_head_output,
    rope_torch,
    rope_tables,
    static_scale_from_amax,
)
from llmcompressor.args import DatasetArguments  # noqa: E402
from llmcompressor.datasets import get_calibration_dataloader  # noqa: E402


def save_tensor(tensor_dir: Path, name: str, tensor: torch.Tensor) -> None:
    save_file(
        {name: tensor.detach().cpu().contiguous()},
        str(tensor_dir / f"{name.replace('.', '__')}.safetensors"),
    )


def update_head_amax(acc, x):
    cur = x.abs().amax(dim=(0, 2)).to(torch.int64)
    return cur if acc is None else torch.maximum(acc, cur)


def update_tensor_amax(acc, x):
    cur = x.abs().amax().reshape(1).to(torch.int64)
    return cur if acc is None else torch.maximum(acc, cur)


def update_stats_amax(acc, stats):
    out = {} if acc is None else acc
    for name, value in stats.items():
        if name in ("input_qkv_i8", "attn_i8", "post_mlp_i8", "gated_mlp_i8"):
            out[name] = update_tensor_amax(out.get(name), value)
        else:
            out[name] = update_head_amax(out.get(name), value)
    return out


def scales_from_amax(layer):
    return {
        "q_pre_rope_i16": static_scale_from_amax(layer["q_pre_rope_i16"], 32767),
        "k_pre_rope_i16": static_scale_from_amax(layer["k_pre_rope_i16"], 32767),
        "input_qkv_i8": static_scale_from_amax(layer["input_qkv_i8"], 127),
        "q_post_rope_i8": static_scale_from_amax(layer["q_post_rope_i8"], 127),
        "k_post_rope_i8": static_scale_from_amax(layer["k_post_rope_i8"], 127),
        "v_i8": static_scale_from_amax(layer["v_i8"], 127),
        "attn_i8": static_scale_from_amax(layer["attn_i8"], 127),
        "post_mlp_i8": static_scale_from_amax(layer["post_mlp_i8"], 127),
        "gated_mlp_i8": static_scale_from_amax(layer["gated_mlp_i8"], 127),
    }


@torch.no_grad()
def run_calib_segment(x, w, norms, cos, sin, config, prefix_tokens):
    input_norm, post_norm, q_norm, k_norm = norms
    h = rmsnorm_torch(x, input_norm)
    q = h @ w["q_proj"].T
    k = h @ w["k_proj"].T
    v = h @ w["v_proj"].T
    batch, seq_len = x.shape[:2]
    cos_b = cos[None, :, :].expand(batch, seq_len, config.head_dim // 2).reshape(batch * seq_len, config.head_dim // 2)
    sin_b = sin[None, :, :].expand(batch, seq_len, config.head_dim // 2).reshape(batch * seq_len, config.head_dim // 2)
    q = rmsnorm_torch(q.reshape(batch, seq_len, config.num_attention_heads, config.head_dim), q_norm)
    k = rmsnorm_torch(k.reshape(batch, seq_len, config.num_key_value_heads, config.head_dim), k_norm)
    q_rope = rope_torch(q.reshape(batch * seq_len, config.q_size), cos_b, sin_b, config.num_attention_heads, config.head_dim).reshape(batch, seq_len, config.num_attention_heads, config.head_dim)
    k_rope = rope_torch(k.reshape(batch * seq_len, config.kv_size), cos_b, sin_b, config.num_key_value_heads, config.head_dim).reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    q_rope = fast_hadamard(q_rope)
    k_rope = fast_hadamard(k_rope)
    v = v.reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    stats = {
        "input_qkv_i8": q15_16(h[:, prefix_tokens:]),
        "q_pre_rope_i16": q15_16(q).reshape(-1, config.num_attention_heads, config.head_dim),
        "k_pre_rope_i16": q15_16(k).reshape(-1, config.num_key_value_heads, config.head_dim),
        "q_post_rope_i8": q15_16(q_rope).reshape(-1, config.num_attention_heads, config.head_dim),
        "k_post_rope_i8": q15_16(k_rope).reshape(-1, config.num_key_value_heads, config.head_dim),
        "v_i8": q15_16(v).reshape(-1, config.num_key_value_heads, config.head_dim),
    }
    attn = torch.nn.functional.scaled_dot_product_attention(
        q_rope.permute(0, 2, 1, 3),
        k_rope.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        is_causal=True,
        enable_gqa=True,
    )
    attn = attn.permute(0, 2, 1, 3).reshape(batch, seq_len, config.q_size)
    stats["attn_i8"] = q15_16(attn[:, prefix_tokens:])
    x = x + attn @ w["o_proj"].T
    m = rmsnorm_torch(x, post_norm)
    gate = m @ w["gate_proj"].T
    up = m @ w["up_proj"].T
    gated_h = fast_hadamard(torch.nn.functional.silu(gate) * up, config.head_dim)
    stats["post_mlp_i8"] = q15_16(m[:, prefix_tokens:])
    stats["gated_mlp_i8"] = q15_16(gated_h[:, prefix_tokens:])
    return x + gated_h @ w["down_proj"].T, stats


def spinquant_offline_config(head_dim: int) -> TransformConfig:
    return TransformConfig(
        config_groups={
            "R1": TransformScheme(
                type="random-hadamard",
                apply=[
                    TransformArgs(targets=["re:.*embed_tokens$", "re:.*self_attn.o_proj$", "re:.*mlp.down_proj$"], location="weight_output"),
                    TransformArgs(targets=["re:.*self_attn.q_proj$", "re:.*self_attn.k_proj$", "re:.*self_attn.v_proj$", "re:.*mlp.up_proj$", "re:.*mlp.gate_proj$", "lm_head"], location="weight_input", inverse=True),
                ],
            ),
            "R4": TransformScheme(
                type="hadamard",
                head_dim=head_dim,
                apply=[
                    TransformArgs(targets=["re:.*mlp.down_proj$"], location="weight_input", inverse=True),
                ],
            ),
        }
    )


def r2_matrix(head_dim: int, seed: int, device: str = "cuda") -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (random_hadamard_matrix(head_dim, torch.float64, torch.device(device), gen) / (head_dim**0.5)).to(torch.float32)


def calib_ids(args, tokenizer, seq_len: int, total_tokens: int) -> tuple[torch.Tensor, int]:
    text = Path(args.calib_text_file).read_text(encoding="utf-8") if args.calib_text_file else args.calib_text
    dataset = Dataset.from_dict({"text": [text]})
    prefix = tokenizer(args.cache_prompt, add_special_tokens=False).input_ids
    ds_args = DatasetArguments(
        dataset=dataset,
        batch_size=1,
        data_collator="truncation",
        num_calibration_samples=args.calib_batches,
        max_seq_length=max(seq_len - len(prefix), 1),
        pad_to_max_length=True,
        shuffle_calibration_samples=False,
    )
    loader = get_calibration_dataloader(ds_args, tokenizer)
    rows = []
    prefix_t = torch.tensor(prefix, dtype=torch.long)
    for batch in loader:
        ids = batch["input_ids"]
        if ids.ndim == 1:
            ids = ids[None, :]
        for row in ids:
            row = torch.cat((prefix_t, row.cpu().long()), dim=0)[:seq_len]
            if row.numel() < seq_len:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                row = torch.cat((row, torch.full((seq_len - row.numel(),), int(pad_id), dtype=torch.long)))
            rows.append(row)
            if len(rows) * seq_len >= total_tokens:
                return torch.stack(rows).reshape(-1)[:total_tokens].cuda(), len(prefix)
    raise RuntimeError("not enough calibration text")


def module_weight(model, layer: int, owner: str, name: str) -> torch.Tensor:
    mod = model.model.layers[layer]
    block = getattr(mod, owner)
    return getattr(block, name).weight.detach().float()


def save_pack(args) -> Path:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "qwen3_int_only.safetensors"

    cfg = Qwen3Config.from_model_dir(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16).cuda()
    untie_word_embeddings(model)
    center_embeddings(model.model.embed_tokens)
    for layer in model.model.layers:
        fuse_norm_linears(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        fuse_norm_linears(layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj])
    fuse_norm_linears(model.model.norm, [model.lm_head])
    apply_transform_config(model, spinquant_offline_config(cfg.head_dim))
    model.eval()

    ids, prefix_tokens = calib_ids(args, tokenizer, args.calib_seq_len, args.calib_seq_len * args.calib_batches)
    tensors = {}
    embed = model.model.embed_tokens.weight.detach().float()
    lm_head = model.lm_head.weight.detach().float()
    final_norm = model.model.norm.weight.detach().float()
    tensors["model.embed_tokens.weight"] = embed.cpu().contiguous()
    tensors["lm_head.weight"] = lm_head.cpu().contiguous()
    tensors["model.norm.weight"] = final_norm.cpu().contiguous()
    tensors["model.embed_tokens.i8.weight"], tensors["model.embed_tokens.i8.scale"] = [x.cpu().contiguous() for x in per_tensor_i8_weight(embed)]
    tensors["model.embed_tokens.i8.scale"] = tensors["model.embed_tokens.i8.scale"].reshape(1)
    tensors["lm_head.i8.weight"], tensors["lm_head.i8.scale"] = [x.cpu().contiguous() for x in per_channel_i8_weight(lm_head)]
    tensors["quarot.r1"] = torch.empty((0,), dtype=torch.float32)
    tensors["quarot.r4"] = torch.empty((0,), dtype=torch.float32)

    calib_x = embed[ids.reshape(-1, args.calib_seq_len)].float()
    cos, sin, _ = rope_tables(args.calib_seq_len, cfg.head_dim, cfg.rope_theta, "cuda")
    for layer in range(cfg.num_hidden_layers):
        dst = f"layers.{layer}"
        input_norm = model.model.layers[layer].input_layernorm.weight.detach().float()
        post_norm = model.model.layers[layer].post_attention_layernorm.weight.detach().float()
        q_norm = model.model.layers[layer].self_attn.q_norm.weight.detach().float()
        k_norm = model.model.layers[layer].self_attn.k_norm.weight.detach().float()
        tensors[f"{dst}.input_layernorm"] = input_norm.cpu().contiguous()
        tensors[f"{dst}.post_attention_layernorm"] = post_norm.cpu().contiguous()
        tensors[f"{dst}.q_norm"] = q_norm.cpu().contiguous()
        tensors[f"{dst}.k_norm"] = k_norm.cpu().contiguous()

        r2 = r2_matrix(cfg.head_dim, args.rotate_seed, "cuda")
        tensors[f"{dst}.r2"] = r2.cpu().contiguous()
        weights = {}
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[name] = module_weight(model, layer, "self_attn", name)
        for name in ("gate_proj", "up_proj", "down_proj"):
            weights[name] = module_weight(model, layer, "mlp", name)
        weights["v_proj"] = rotate_head_output(weights["v_proj"], cfg.head_dim, r2)
        weights["o_proj"] = rotate_head_input(weights["o_proj"], cfg.head_dim, r2)
        for name, weight in weights.items():
            w, s = per_channel_i8_weight(weight)
            tensors[f"{dst}.{name}.weight"] = w.cpu().contiguous()
            tensors[f"{dst}.{name}.scale"] = s.cpu().contiguous()
        for fused, names in {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}.items():
            w, s = per_channel_i8_weight(torch.cat([weights[n] for n in names], dim=0))
            tensors[f"{dst}.{fused}.weight"] = w.cpu().contiguous()
            tensors[f"{dst}.{fused}.scale"] = s.cpu().contiguous()

        next_x = torch.empty_like(calib_x)
        stats_amax = None
        for start in range(0, calib_x.shape[0], args.calib_micro_batch):
            end = min(start + args.calib_micro_batch, calib_x.shape[0])
            next_x[start:end], stats = run_calib_segment(
                calib_x[start:end],
                weights,
                (input_norm, post_norm, q_norm, k_norm),
                cos,
                sin,
                cfg,
                prefix_tokens,
            )
            stats_amax = update_stats_amax(stats_amax, stats)
        calib_x = next_x
        for name, scale in scales_from_amax(stats_amax).items():
            tensors[f"{dst}.{name}.scale"] = scale.cpu().contiguous()

    final_h = rmsnorm_torch(calib_x, final_norm)
    tensors["model.final_i8.scale"] = static_scale_from_amax(q15_16(final_h[:, prefix_tokens:]).abs().amax().reshape(1), 127).cpu().contiguous()

    save_file(
        tensors,
        str(path),
        metadata={
            "model_dir": args.model_dir,
            "num_hidden_layers": str(cfg.num_hidden_layers),
            "hidden_size": str(cfg.hidden_size),
            "intermediate_size": str(cfg.intermediate_size),
            "num_attention_heads": str(cfg.num_attention_heads),
            "num_key_value_heads": str(cfg.num_key_value_heads),
            "head_dim": str(cfg.head_dim),
            "packed_layers": str(cfg.num_hidden_layers),
            "calib_seq_len": str(args.calib_seq_len),
            "calib_batches": str(args.calib_batches),
            "calib_tokens": str(args.calib_seq_len * args.calib_batches),
            "calib_prefix_tokens": str(prefix_tokens),
            "cache_prompt": args.cache_prompt,
            "use_r1": "1",
            "use_r2": "1",
            "r2_impl": "compressed-tensors-offline-random-shared",
            "r3_impl": "fwht-runtime",
            "r3_quantization": "oblivious",
            "r4_impl": "llmcompressor-offline-exact",
            "weight_scale_dtype": "fp32",
        },
    )
    return path


def save_artifact(pack: Path, artifact_dir: str) -> None:
    from safetensors import safe_open

    output = Path(artifact_dir)
    tensor_dir = output / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    with safe_open(str(pack), framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        num_layers = int(metadata["num_hidden_layers"])

        for name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
            save_tensor(tensor_dir, name, reader.get_tensor(name))

        for name, src in (
            ("lm_head.i8.weight", "lm_head.i8.weight"),
            ("lm_head.i8.weight_scale", "lm_head.i8.scale"),
            ("lm_head.input_scale", "model.final_i8.scale"),
        ):
            save_tensor(tensor_dir, name, reader.get_tensor(src))

        for layer_id in range(num_layers):
            src = f"layers.{layer_id}"
            dst = f"model.layers.{layer_id}"
            for src_name, dst_name in (
                ("input_layernorm", "input_layernorm.weight"),
                ("post_attention_layernorm", "post_attention_layernorm.weight"),
                ("q_norm", "self_attn.q_norm.weight"),
                ("k_norm", "self_attn.k_norm.weight"),
                ("r2", "r2"),
            ):
                save_tensor(tensor_dir, f"{dst}.{dst_name}", reader.get_tensor(f"{src}.{src_name}"))

            for module, input_scale_name in (
                ("self_attn.qkv_proj", "input_qkv_i8"),
                ("self_attn.o_proj", "attn_i8"),
                ("mlp.gate_up_proj", "post_mlp_i8"),
                ("mlp.down_proj", "gated_mlp_i8"),
            ):
                src_module = module.rsplit(".", 1)[-1]
                dst_module = f"{dst}.{module}"
                save_tensor(tensor_dir, f"{dst_module}.weight", reader.get_tensor(f"{src}.{src_module}.weight"))
                save_tensor(
                    tensor_dir,
                    f"{dst_module}.weight_scale",
                    reader.get_tensor(f"{src}.{src_module}.scale"),
                )
                save_tensor(
                    tensor_dir,
                    f"{dst_module}.input_scale",
                    reader.get_tensor(f"{src}.{input_scale_name}.scale"),
                )
                save_tensor(tensor_dir, f"{dst_module}.output_scale", torch.tensor([1.0], dtype=torch.float32))

            for name in ("q_post_rope_i8", "k_post_rope_i8", "v_i8"):
                save_tensor(tensor_dir, f"layers.{layer_id}.{name}.scale", reader.get_tensor(f"{src}.{name}.scale"))

    (output / "int_w8a8_static_config.json").write_text(
        '{"format":"minisgl-int-w8a8-static","version":1}\n',
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--calib-text", default="")
    p.add_argument("--calib-text-file", default="")
    p.add_argument("--calib-seq-len", type=int, default=2048)
    p.add_argument("--calib-batches", type=int, default=32)
    p.add_argument("--calib-micro-batch", type=int, default=1)
    p.add_argument("--cache-prompt", default="你是一个有用而无害的聊天助手。")
    p.add_argument("--rotate-seed", type=int, default=ROTATE_SEED)
    args = p.parse_args()
    pack = save_pack(args)
    save_artifact(pack, args.artifact_dir)
    print(pack)
    print(args.artifact_dir)


if __name__ == "__main__":
    main()
