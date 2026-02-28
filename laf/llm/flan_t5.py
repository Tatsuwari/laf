from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,TextIteratorStreamer
import torch

import threading

from laf.modes.config import GenConfig
from laf.llm.provider import LLM


@dataclass
class FlanT5Config:
    model_name: str = "google/flan-t5-base"
    device: str = "cpu"


class FlanT5LLM(LLM):
    def __init__(self, cfg: FlanT5Config):
        self.cfg = cfg

        print(f"Loading FLAN-t5: {cfg.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

        if cfg.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def chat(self, system: str, user: str, gen: Optional[GenConfig] = None) -> str:
        gen = gen or GenConfig(max_new_tokens=256, temperature=0.0)

        prompt = f"{system}\nUser: {user}"

        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.cfg.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs, max_new_tokens=gen.max_new_tokens, temperature=gen.temperature
        )

        text = self.tokenizer.decode(outputs[0], skip_speical_tokens=True)
        return text.strip()
    def stream_chat(
        self,
        system: str,
        user: str,
        gen: Optional[GenConfig] = None,
    ):
        gen = gen or GenConfig(max_new_tokens=256, temperature=0.0)

        prompt = f"{system}\nUser: {user}\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.cfg.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            do_sample=gen.temperature > 0,
            streamer=streamer,
        )

        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        for token in streamer:
            yield token