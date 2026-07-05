from __future__ import annotations

from typing import List, Tuple

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        results: List[Tuple[torch.Tensor, torch.Tensor]] = []
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
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            prefix_ids = self.tokenizer.encode(msg.prefix_text, return_tensors="pt") if msg.prefix_text else torch.empty(0, dtype=torch.int64)
            results.append((input_ids.view(-1).to(torch.int32), prefix_ids.view(-1).to(torch.int32)))
        return results
