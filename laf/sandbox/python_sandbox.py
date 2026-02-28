from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any,Dict, Optional

from .guard import SandboxGuard
from .limits import SandboxLimits


@dataclass
class SandboxResult:
    ok: bool
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None


class PythonSandbox:
    '''
    Runs a tool in a seperate python process eg:
    python -m laf.sandbox.runner
    '''

    def __init__(self,limits:Optional[SandboxLimits]=None):
        self.guard = SandboxGuard(limits or SandboxLimits())

    def run(self,entrypoint: str, args: Dict[str,Any]) -> SandboxResult:
        payload = {
            'entrypoint': entrypoint,
            'args': args,
            'allow_network': self.guard.can_use_network(),
        }

        proc = subprocess.Popen(
            [sys.executable, '-m', 'laf.sandbox.runner'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            out,err = proc.communicate(
                json.dumps(payload),
                timeout=self.guard.limits.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            return SandboxResult(ok=False,error='timeout')

        # cap output reads
        out = (out or '')[: self.guard.limits.max_output_bytes]

        # runner prints JSON on stdout
        try:
            data = json.loads(out.strip() or '{}')
        except Exception:
            return SandboxResult(ok=False,error='invalid_runner_output',traceback=err[:2000])
        if data.get('ok'):
            return SandboxResult(ok=True,result=data.get('result'))
        return SandboxResult(
            ok=False,
            error=str(data.get('error') or 'unknown_error'),
            traceback=data.get('traceback'),
        )
