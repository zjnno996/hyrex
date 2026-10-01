"""Exercise actual commit/retire callback with a storage-manager double."""
import ast
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from types import SimpleNamespace

source = Path(__file__).resolve().parents[2] / "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py"
tree = ast.parse(source.read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_finish_checkpoint_write")
scope = {"logger": logging.getLogger(__name__)}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)

@dataclass(frozen=True)
class Key:
    cache_salt: str
    model_name: str = "model"
    kv_rank: int = 0
    object_group_id: int = 0

class Storage:
    def __init__(self):
        self.present = set()
        self.locked = set()
        self.events = []
    def finish_write(self, keys):
        self.present.update(keys)
        self.events.append(("commit", tuple(keys)))
    def delete_l1_keys(self, keys, force=False):
        assert not force
        self.events.append(("delete", tuple(keys)))
        if any(key in self.locked for key in keys):
            return 0, 1
        deleted = sum(key in self.present for key in keys)
        self.present.difference_update(keys)
        return deleted, 0

storage = Storage()
owner = SimpleNamespace(_ctx=SimpleNamespace(storage_manager=storage),
    _checkpoint_slots={}, _checkpoint_retired=set(), _checkpoint_lock=threading.Lock())
commit = lambda keys: scope[method.name](owner, keys)
old, new, other, full = (Key(s) for s in (
    "sf1:slot:544:old", "sf1:slot:640:new", "sf1:other:640:other", "normal_full_kv"))
commit([old, other, full])
storage.locked.add(old)
commit([new])
assert old in storage.present and new in storage.present
assert old in owner._checkpoint_retired
storage.locked.remove(old)
commit([new])
assert old not in storage.present
assert storage.present == {new, other, full}
assert not owner._checkpoint_retired
first_delete = next(i for i, (event, _) in enumerate(storage.events) if event == "delete")
assert storage.events[first_delete-1] == ("commit", (new,))
print("commit-before-retire, locked retry, and unrelated-object protection passed")
