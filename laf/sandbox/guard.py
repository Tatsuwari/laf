from __future__ import annotations
from dataclasses import dataclass
from .limits import SandboxLimits

@dataclass
class SandboxGuard:
    '''
    Minimal policy/permissions layer. Expand later with rules, packge installs etc.
    '''
    limits: SandboxLimits

    def can_use_network(self) -> bool:
        return bool(self.limits.allow_network)
