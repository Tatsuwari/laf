import re
from typing import List
from .config import SystemConfig

class PlanValidator:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    def normalize(self, s: str) -> str:
        s = s.replace("<|im_end|>", "").replace("<|assistant|>", "").strip()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def validate_steps(self, steps: List[str]) -> List[str]:
        out = []
        for s in steps:
            s = self.normalize(s)
            if not s:
                continue
            if len(s.split()) < self.cfg.min_step_words:
                continue
            out.append(s)
        return out[: self.cfg.max_steps]
