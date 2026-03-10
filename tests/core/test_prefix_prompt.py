from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Dict, List, cast

from transformers import PreTrainedTokenizerBase

import torch

from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.server.args import parse_args
from minisgl.tokenizer.tokenize import TokenizeManager


class _DummyTokenizer:
    def __init__(self) -> None:
        self.rendered_prompts: List[str] = []
        self.chat_template_calls: List[List[Dict[str, str]]] = []

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.chat_template_calls.append(messages)
        joined = " | ".join(f"{m['role']}:{m['content']}" for m in messages)
        return f"CHAT<{joined}>"

    def encode(self, prompt: str) -> List[int]:
        self.rendered_prompts.append(prompt)
        return [len(prompt)]


def test_launch_prefix_prompt_supports_escaped_newlines(monkeypatch) -> None:
    def _fake_hf_config(_: str) -> SimpleNamespace:
        return SimpleNamespace(dtype="float16", quantization_config=None)

    monkeypatch.setattr("minisgl.utils.cached_load_hf_config", _fake_hf_config)
    server_args, _ = parse_args(["--model", "dummy/model", "--prefix-prompt", "a\\n\\n\\nb"])

    assert server_args.prefix_prompt == "a\n\n\nb"


def test_prefix_prompt_is_prepended_for_text_prompt() -> None:
    tokenizer = _DummyTokenizer()
    manager = TokenizeManager(
        cast(PreTrainedTokenizerBase, tokenizer),
        prefix_prompt="PREFIX: ",
    )
    msgs = [TokenizeMsg(uid=0, text="hello", sampling_params=SamplingParams())]

    result = manager.tokenize(msgs)

    assert tokenizer.rendered_prompts == ["PREFIX: hello"]
    assert torch.equal(result[0], torch.tensor([13], dtype=torch.int32))


def test_prefix_prompt_is_prepended_before_chat_and_system_content() -> None:
    tokenizer = _DummyTokenizer()
    manager = TokenizeManager(
        cast(PreTrainedTokenizerBase, tokenizer),
        prefix_prompt="SYS_PREFIX\n",
    )
    msgs = [
        TokenizeMsg(
            uid=1,
            text=[
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
            sampling_params=SamplingParams(),
        )
    ]

    manager.tokenize(msgs)

    assert tokenizer.chat_template_calls == [msgs[0].text]
    assert tokenizer.rendered_prompts == ["SYS_PREFIX\nCHAT<system:be concise | user:hello>"]


def test_public_api_models_do_not_expose_prefix_prompt() -> None:
    with open("python/minisgl/server/api_server.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    fields_by_model: Dict[str, List[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "GenerateRequest",
            "OpenAICompletionRequest",
        }:
            fields: List[str] = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append(stmt.target.id)
            fields_by_model[node.name] = fields

    assert "prefix_prompt" not in fields_by_model["GenerateRequest"]
    assert "prefix_prompt" not in fields_by_model["OpenAICompletionRequest"]
