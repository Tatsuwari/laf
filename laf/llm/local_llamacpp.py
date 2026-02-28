# laf/llm/local_llamacpp.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .provider import GenConfig

import os
import sys
import contextlib


@dataclass
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    suppress_logs: bool = True   # <-- new flag


class LlamaCppLLM:
    def __init__(self, cfg: LlamaCppConfig):
        from llama_cpp import Llama
        self.cfg = cfg

        print("Loading model:", self.cfg)

        if cfg.suppress_logs:
            self._llm = self._load_silent(Llama)
        else:
            self._llm = Llama(
                model_path=cfg.model_path,
                n_ctx=cfg.n_ctx,
                n_gpu_layers=cfg.n_gpu_layers,
                n_threads=cfg.n_threads,
                verbose=False,
            )

    def _load_silent(self, Llama):
        """
        Suppress C-level stderr logs from llama.cpp during model load.
        """
        devnull = open(os.devnull, "w")

        with contextlib.redirect_stderr(devnull):
            llm = Llama(
                model_path=self.cfg.model_path,
                n_ctx=self.cfg.n_ctx,
                n_gpu_layers=self.cfg.n_gpu_layers,
                n_threads=self.cfg.n_threads,
                verbose=False,
            )

        devnull.close()
        return llm

    def chat(self, system: str, user: str, gen: Optional[GenConfig] = None) -> str:
        gen = gen or GenConfig(max_new_tokens=256, temperature=0.0)

        prompt = (
            "<|im_start|>system\n"
            f"{system}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        out = self._llm.create_completion(
            prompt=prompt,
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
            stop=["<|im_end|>"],
        )

        text = out["choices"][0]["text"]
        return text.strip() if text else ""
    
    def stream_chat(self, system: str, user: str, gen: Optional[GenConfig] = None):
        gen = gen or GenConfig(max_new_tokens=256, temperature=0.0)

        prompt = (
            "<|im_start|>system\n"
            f"{system}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        stream = self._llm.create_completion(
            prompt=prompt,
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
            stop=["<|im_end|>"],
            stream=True,
        )

        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if token:
                yield token