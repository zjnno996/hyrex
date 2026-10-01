"""Content-addressed movable checkpoints, one entry per coarse prefix slot."""
import base64
import hashlib
import json


def digest(tokens, salt):
    return base64.urlsafe_b64encode(hashlib.sha256(
        json.dumps([salt, tokens], separators=(",", ":")).encode()).digest()).decode().rstrip("=")


def checkpoint_salt(tokens, boundary, salt):
    if not 528 <= boundary <= len(tokens) or boundary % 16:
        raise ValueError("checkpoint must be a complete Full16 prefix with a coarse slot")
    coarse = boundary // 528 * 528
    return f"sf1:{digest(tokens[:coarse], salt)}:{boundary}:{digest(tokens[:boundary], salt)}"


class CheckpointIndex:
    """Hints only: LMCache must still verify object existence and lock it."""
    def __init__(self):
        self.slots = {}

    def record(self, tokens, boundary, salt):
        namespace = checkpoint_salt(tokens, boundary, salt)
        self.slots[namespace.split(":")[1]] = (boundary, namespace)

    def candidates(self, tokens, full_hit, salt):
        matches = []
        for coarse in range(528, min(len(tokens), full_hit) + 1, 528):
            record = self.slots.get(digest(tokens[:coarse], salt))
            if record is None:
                continue
            boundary, namespace = record
            if boundary <= full_hit and namespace == checkpoint_salt(tokens, boundary, salt):
                matches.append(boundary)
        return sorted(matches, reverse=True)
