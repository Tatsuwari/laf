from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class GenConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.05

@dataclass
class SystemConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: torch.dtype = torch.float16
    device_map: str = "auto"

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    intent_store_path: Path = Path("data/intent_store.json")
    plugin_dir: Path = Path("plugins_ext")

    # planning safety guard
    max_steps: int = 12
    min_step_words: int = 3

    # routing
    similarity_threshold: float = 0.55  # tune: 0.55-0.65 typical
    top_k_matches: int = 3

    # logging
    intent_log_path: Path = Path("data/intent_logs.jsonl")
    log_if_created_new: bool = True
    log_if_below_threshold: float = 0.55
