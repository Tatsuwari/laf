
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.05
    stop: Optional[list[str]] = None
