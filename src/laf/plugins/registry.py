from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional,Tuple
from ..trace import Tracer

@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    params_schema: Optional[dict] = None  # simple JSON schema-like dict

    # optional future fields
    category_path: Optional[list[str]] = None
    returns_schema: Optional[dict] = None
    tags: Optional[list[str]] = None
    constraints: Optional[dict] = None

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def has(self, name: str) -> bool:
        return name in self.tools
    
    def validate_args(self, schema: dict, args: Dict[str, Any]) -> Tuple[bool, str]:
        '''
        Minimal schema validator:
          schema = { 'type':'object', 'properties':{...}, 'required':[...] }
        '''
        if not isinstance(args, dict):
            return False, 'args must be an object'

        stype = schema.get('type')
        if stype and stype != 'object':
            return False, f'schema type must be object (got {stype})'

        required = schema.get('required', []) or []
        if isinstance(required, list):
            for k in required:
                if k not in args:
                    return False, f'missing required field: {k}'

        props = schema.get('properties', {}) or {}
        if isinstance(props, dict):
            for k, ps in props.items():
                if k not in args:
                    continue
                if not isinstance(ps, dict):
                    continue
                expected = ps.get('type')
                if not expected:
                    continue
                v = args[k]
                if expected == 'string' and not isinstance(v, str):
                    return False, f'field {k} must be string'
                if expected == 'number' and not isinstance(v, (int, float)):
                    return False, f'field {k} must be number'
                if expected == 'integer' and not isinstance(v, int):
                    return False, f'field {k} must be integer'
                if expected == 'boolean' and not isinstance(v, bool):
                    return False, f'field {k} must be boolean'
                if expected == 'object' and not isinstance(v, dict):
                    return False, f'field {k} must be object'
                if expected == 'array' and not isinstance(v, list):
                    return False, f'field {k} must be array'
        return True, ''

    def run(self, name: str, args: Dict[str, Any], tracer: Optional['Tracer'] = None) -> Dict[str, Any]:
        
        if tracer:
            tracer.emit('tool.run.start', tool=name, args=args)

        if name not in self.tools:
            out = {'ok': False, 'error': f'Tool not found: {name}'}
            if tracer:
                tracer.emit('tool.run.end', tool=name, ok=False, error=out['error'])
            return out
        
        spec = self.tools[name]
        if spec.params_schema:
            ok,err = self.validate_args(spec.params_schema, args or {})
            if not ok:
                out = {'ok': False, 'tool': name, 'error': f'Args validation failed: {err}', 'args': args}
                if tracer:
                    tracer.emit('tool.run.end', tool=name, ok=False, error=out['error'], args=args)
                return out
        try:
            res = spec.fn(args or {})
            if isinstance(res,dict) and 'ok' in res:
                # allow tool to control its own envelope
                out = res
            else:
                out = {'ok': True, 'tool': name, 'result': res}

            if tracer:
                tracer.emit('tool.run.end', tool=name, ok=bool(out.get('ok', True)))
            return out
        except Exception as e:
            out = {'ok': False, 'tool': name, 'error': str(e)}
            if tracer:
                tracer.emit('tool.run.end', tool=name, ok=False, error=str(e))
            return out