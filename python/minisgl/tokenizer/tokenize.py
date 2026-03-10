from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, prefix_prompt: str = "") -> None:
        self.tokenizer = tokenizer
        self.prefix_prompt = prefix_prompt

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            prompt = self.prefix_prompt + prompt
            encoded = self.tokenizer.encode(prompt)
            input_ids = torch.tensor(encoded, dtype=torch.int32)
            results.append(input_ids.view(-1))
        return results
