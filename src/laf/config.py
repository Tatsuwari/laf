from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class ModelProfile:
    model_name: str
    dtype: torch.dtype = torch.float16
    device_map: str = "auto"


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
    
    # model profile
    model_profiles: dict = field(default_factory=lambda: {
        'fast': ModelProfile(model_name="Qwen/Qwen2.5-3B-Instruct"),
        'medium': ModelProfile(model_name="Qwen/Qwen2.5-7B-Instruct"),
        'heavy': ModelProfile(model_name="Qwen/Qwen2.5-14B-Instruct"),
    })
    active_profile: str = 'fast'

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # storage
    intent_store_path: Path = Path("data/intent_store.json")
    plugin_dir: Path = Path("plugins_ext")

    # planning safety guard
    max_steps: int = 12
    min_step_words: int = 3

    # routing
    similarity_threshold: float = 0.55  # tune: 0.55-0.65 typical
    top_k_matches: int = 3


    # rag
    internal_only: bool = False  # 🔒 If True, never use web_search
    rag_top_k: int = 3
    rag_score_threshold: float = 0.55  # similarity threshold

    # logging
    intent_log_path: Path = Path("data/intent_logs.jsonl")
    log_if_created_new: bool = True
    log_if_below_threshold: float = 0.55

    # pipeline
    trace_enabled: bool = True
    trace_dir: Path = Path("data/traces")
    plan_format: str = "linear"  # linear | tree | dag

    pool_size: int = 1 # number of model instances to load on startup
