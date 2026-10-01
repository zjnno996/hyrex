# SPDX-License-Identifier: Apache-2.0
"""Prefix index used by the Marconi baseline.

This is a clean-room, vLLM-facing implementation of the original Marconi
index semantics.  The index keeps compressed radix edges, splits an edge when
a new request branches in the middle, and can evict a node with one child by
absorbing that child into the parent edge.  The latter is important: replacing
it with leaf-only LRU would no longer be a Marconi baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


Path = tuple[str, ...]


@dataclass(slots=True)
class MarconiNode:
    """One compressed radix node."""

    node_id: int
    path: Path
    edge: Path
    parent_id: int | None
    state_kind: str
    token_count: int
    byte_size: int
    last_access_ms: float = 0.0
    compute_savings_ms: float = 0.0
    terminal: bool = False
    ref_count: int = 0
    # Child lookup is by the first component of the compressed edge.
    children: dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return "/".join(self.path)


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if low == high:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


class MarconiIndex:
    """Compressed prefix index with Marconi-style utility eviction."""

    def __init__(self) -> None:
        self._next_id = 1
        self._nodes: dict[int, MarconiNode] = {}
        self._root = MarconiNode(
            node_id=0,
            path=(),
            edge=(),
            parent_id=None,
            state_kind="root",
            token_count=0,
            byte_size=0,
        )

    def _new_node(
        self,
        *,
        path: Path,
        edge: Path,
        parent_id: int,
        state_kind: str,
        token_count: int = 0,
        byte_size: int = 0,
        last_access_ms: float = 0.0,
        compute_savings_ms: float = 0.0,
        terminal: bool = False,
    ) -> MarconiNode:
        node = MarconiNode(
            node_id=self._next_id,
            path=path,
            edge=edge,
            parent_id=parent_id,
            state_kind=state_kind,
            token_count=token_count,
            byte_size=byte_size,
            last_access_ms=last_access_ms,
            compute_savings_ms=compute_savings_ms,
            terminal=terminal,
        )
        self._next_id += 1
        self._nodes[node.node_id] = node
        self._parent(parent_id).children[edge[0]] = node.node_id
        return node

    def _parent(self, node_id: int) -> MarconiNode:
        return self._root if node_id == 0 else self._nodes[node_id]

    def _repath(self, node_id: int) -> None:
        """Refresh full paths after an edge split or merge."""
        node = self._nodes[node_id]
        parent = self._parent(node.parent_id or 0)
        node.path = parent.path + node.edge
        for child_id in node.children.values():
            self._repath(child_id)

    @staticmethod
    def _common_prefix(left: Path, right: Path) -> int:
        length = min(len(left), len(right))
        index = 0
        while index < length and left[index] == right[index]:
            index += 1
        return index

    def insert(
        self,
        segments: Iterable[str],
        *,
        state_kind: str,
        token_count: int,
        byte_size: int,
        last_access_ms: float = 0.0,
        compute_savings_ms: float = 0.0,
    ) -> Path:
        """Insert a sequence, preserving compressed radix split behavior."""
        tokens = tuple(segments)
        if not tokens or any(not token for token in tokens):
            raise ValueError("segments must contain at least one non-empty item")

        current_id = 0
        offset = 0
        while offset < len(tokens):
            current = self._parent(current_id)
            child_id = current.children.get(tokens[offset])
            if child_id is None:
                node = self._new_node(
                    path=current.path + tokens[offset:],
                    edge=tokens[offset:],
                    parent_id=current_id,
                    state_kind=state_kind,
                    token_count=token_count,
                    byte_size=byte_size,
                    last_access_ms=last_access_ms,
                    compute_savings_ms=compute_savings_ms,
                    terminal=True,
                )
                return node.path

            child = self._nodes[child_id]
            remaining = tokens[offset:]
            common = self._common_prefix(child.edge, remaining)
            if common == len(child.edge):
                offset += common
                current_id = child_id
                if offset == len(tokens):
                    child.state_kind = state_kind
                    child.token_count = token_count
                    child.byte_size = byte_size
                    child.last_access_ms = last_access_ms
                    child.compute_savings_ms = compute_savings_ms
                    child.terminal = True
                    return child.path
                continue

            # The new sequence branches inside the existing compressed edge.
            branch_edge = child.edge[:common]
            branch = self._new_node(
                path=current.path + branch_edge,
                edge=branch_edge,
                parent_id=current_id,
                state_kind=state_kind,
                last_access_ms=last_access_ms,
            )
            del current.children[child.edge[0]]
            current.children[branch_edge[0]] = branch.node_id

            child.parent_id = branch.node_id
            child.edge = child.edge[common:]
            branch.children[child.edge[0]] = child.node_id
            self._repath(child.node_id)

            offset += common
            if offset == len(tokens):
                branch.state_kind = state_kind
                branch.token_count = token_count
                branch.byte_size = byte_size
                branch.compute_savings_ms = compute_savings_ms
                branch.terminal = True
                return branch.path

            node = self._new_node(
                path=branch.path + tokens[offset:],
                edge=tokens[offset:],
                parent_id=branch.node_id,
                state_kind=state_kind,
                token_count=token_count,
                byte_size=byte_size,
                last_access_ms=last_access_ms,
                compute_savings_ms=compute_savings_ms,
                terminal=True,
            )
            return node.path

        raise AssertionError("unreachable")

    def lookup(self, segments: Iterable[str]) -> Path:
        """Return the longest complete cached radix prefix."""
        tokens = tuple(segments)
        current = self._root
        offset = 0
        longest = ()
        while offset < len(tokens):
            child_id = current.children.get(tokens[offset])
            if child_id is None:
                break
            child = self._nodes[child_id]
            remaining = tokens[offset:]
            common = self._common_prefix(child.edge, remaining)
            if common != len(child.edge):
                break
            offset += common
            longest = child.path
            current = child
        return longest

    def touch(self, segments: Iterable[str], now_ms: float) -> Path:
        """Touch only the terminal matched node, as in Marconi V2/V3."""
        path = self.lookup(segments)
        node = self.get(path)
        if node is not None:
            node.last_access_ms = now_ms
        return path

    def get(self, path: Path) -> MarconiNode | None:
        path = tuple(path)
        for node in self._nodes.values():
            if node.path == path:
                return node
        return None

    def nodes(self) -> tuple[MarconiNode, ...]:
        return tuple(sorted(self._nodes.values(), key=lambda node: node.path))

    @property
    def materialized_bytes(self) -> int:
        return sum(node.byte_size for node in self._nodes.values())

    def select_evictions(
        self,
        bytes_needed: int,
        now_ms: float,
        *,
        protected: Iterable[Path] = (),
        alpha: float = 1.0,
    ) -> tuple[Path, ...]:
        """Select V2-style leaves or single-child nodes.

        Returned paths are ordered for safe sequential ``remove`` calls.  If
        an intermediate node is selected, it is returned last because its
        merge operation changes the child's compressed path.
        """
        if bytes_needed <= 0:
            return ()
        protected_paths = {tuple(path) for path in protected}
        selected: list[Path] = []
        freed = 0

        while freed < bytes_needed:
            candidates = [
                node
                for node in self._nodes.values()
                if node.path not in selected
                and node.path not in protected_paths
                and node.terminal
                and node.ref_count == 0
                and len(node.children) <= 1
            ]
            if not candidates:
                break

            recencies = _normalize(
                [
                    1.0 / (max(0.0, now_ms - node.last_access_ms) + 1.0)
                    for node in candidates
                ]
            )
            efficiencies = _normalize(
                [
                    node.compute_savings_ms / max(node.byte_size, 1)
                    for node in candidates
                ]
            )
            ranked = sorted(
                zip(candidates, recencies, efficiencies),
                key=lambda item: (
                    item[1] + alpha * item[2],
                    -item[0].byte_size,
                    item[0].key,
                ),
            )
            victim = ranked[0][0]
            selected.append(victim.path)
            freed += victim.byte_size
            # Do not select another descendant after a merge candidate: its
            # path is intentionally changed by the original Marconi merge.
            if victim.children:
                break

        return tuple(selected)

    def remove(self, path: Path) -> None:
        """Remove a node, merging its sole child when required by Marconi."""
        node = self.get(path)
        if node is None:
            return
        if node.ref_count:
            raise ValueError(f"cannot remove referenced node {path!r}")
        parent = self._parent(node.parent_id or 0)

        if len(node.children) == 1:
            child_id = next(iter(node.children.values()))
            child = self._nodes[child_id]
            # Original Marconi drops the intermediate node's state and
            # absorbs the child edge/state into a new compressed node.
            del parent.children[node.edge[0]]
            child.parent_id = parent.node_id
            child.edge = node.edge + child.edge
            self._repath(child.node_id)
            parent.children[child.edge[0]] = child.node_id
            del self._nodes[node.node_id]
            return

        if node.children:
            raise ValueError(f"cannot remove branching node {path!r}")
        del parent.children[node.edge[0]]
        del self._nodes[node.node_id]


__all__ = ["MarconiIndex", "MarconiNode", "Path"]
