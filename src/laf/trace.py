from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

def _now_ms() -> int:
    return int(time.time() * 1000)

def _safe_json(x:Any) -> Any:
    '''
    Ensure the payload is JSON-serializable. For unknown objects, stringify them.
    '''
    try:
        json.dumps(x)
        return x
    except Exception:
        if isinstance(x,dict):
            return {str(k): _safe_json(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):
            return [_safe_json(i) for i in x]
        return str(x)
    
def new_task_id(prefix: str = 'task') -> str:
    return f'{prefix}_{uuid.uuid4().hex[:12]}'

@dataclass
class TraceEvent:
    ts_ms: int
    task_id: str
    seq: int
    event: str
    data: Dict[str, Any] = field(default_factory=dict)

class Tracer:
    '''
    - In-memory events during run.
    - Flush to per-task directory on demand.
    '''

    def __init__(
            self,
            base_dir: Path,
            task_id: Optional[str] = None,
            enabled: bool = True,
            ):
        self.base_dir = base_dir
        self.task_id = task_id or new_task_id()
        self.enabled = enabled
        self._events: List[TraceEvent] = []
        self._seq = 0

        self.task_dir = self.base_dir / self.task_id
        self.trace_path = self.task_dir / 'trace.jsonl'

    def emit(self, event: str, **data: Any) -> None:
        if not self.enabled:
            return
        self._seq += 1
        ev = TraceEvent(
            ts_ms=_now_ms(),
            task_id=self.task_id,
            seq=self._seq,
            event=event,
            data=_safe_json(data),
        )
        self._events.append(ev)

    def events(self) -> List[Dict[str, Any]]:
        '''
        Returns JSON-ready list of events for immediate use in responses.
        '''
        return [
            {
                'ts_ms': ev.ts_ms,
                'task_id': ev.task_id,
                'seq': ev.seq,
                'event': ev.event,
                'data': ev.data,
            }
            for ev in self._events
        ]
    def flush(self) -> Path:
        '''
        Write events to per-task jsonl file. Returns path
        '''
        if not self.enabled:
            return self.trace_path
        
        self.task_dir.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open('w',encoding='utf-8') as f:
            for e in self._events:
                json_line = json.dumps({
                    'ts_ms': e.ts_ms,
                    'task_id': e.task_id,
                    'seq': e.seq,
                    'event': e.event,
                    'data': e.data,
                }, ensure_ascii=False)
                f.write(json_line + '\n')
        return self.trace_path
    

    def clear(self) -> None:
        '''
        Clear in-memory events (useful for long sessions if you flush mid-run)
        '''
        self._events.clear()

    def child(self,child_task_id: Optional[str] = None) -> 'Tracer':
        '''
        Create a new tracer in same base dir with new task_id
        '''
        return Tracer(
            base_dir=self.base_dir,
            task_id=child_task_id or new_task_id(),
            enabled=self.enabled,
        )