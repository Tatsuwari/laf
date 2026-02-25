# policies.py
from dataclasses import dataclass

@dataclass
class ExecutionPolicy:
    internal_only: bool = False
    internet_available: bool = True