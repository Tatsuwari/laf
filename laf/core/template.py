import re

def resolve_templates(value, memory):
    if isinstance(value, str):
        matches = re.findall(r"\{\{(.*?)\}\}", value)
        for match in matches:
            keys = match.split(".")
            data = memory
            for k in keys:
                if isinstance(data, dict):
                    data = data.get(k)
                else:
                    data = None
                if data is None:
                    break
            if data is not None:
                value = value.replace(f"{{{{{match}}}}}", str(data))
    return value


def resolve_args(args, memory):
    if isinstance(args, dict):
        return {k: resolve_args(v, memory) for k, v in args.items()}
    elif isinstance(args, list):
        return [resolve_args(v, memory) for v in args]
    else:
        return resolve_templates(args, memory)