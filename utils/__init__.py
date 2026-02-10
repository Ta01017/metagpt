# Re-export helpers from top-level utils.py to avoid name collisions.
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

_utils_path = Path(__file__).resolve().parents[1] / "utils.py"
_spec = spec_from_file_location("_utils_file", _utils_path)
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)

save_checkpoint = _mod.save_checkpoint
load_checkpoint = _mod.load_checkpoint
save_json = _mod.save_json
SimpleLogger = _mod.SimpleLogger
set_seed = _mod.set_seed

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "save_json",
    "SimpleLogger",
    "set_seed",
]
