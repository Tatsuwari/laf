# laf/llm/provider.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol
from laf.modes.config import GenConfig

class LLM(Protocol):
    def chat(self,system: str, user: str, gen: Optional[GenConfig] = None) -> str:
        ...

