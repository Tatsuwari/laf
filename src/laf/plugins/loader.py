import importlib.util
from pathlib import Path
from .registry import ToolRegistry

def load_plugins(plugin_dir: Path, registry: ToolRegistry) -> None:
    plugin_dir = Path(plugin_dir)
    plugin_dir.mkdir(parents=True, exist_ok=True)

    for file in plugin_dir.glob("*.py"):
        spec = importlib.util.spec_from_file_location(file.stem, file)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Convention: plugin must define setup(registry)
        if hasattr(mod, "setup") and callable(mod.setup):
            mod.setup(registry)
