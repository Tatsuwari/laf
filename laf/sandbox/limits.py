from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SandboxLimits:
    timeout_sec: float = 10.0
    max_output_bytes: int = 250_000 # cap stdout/stderr read
    allow_network: bool = False
