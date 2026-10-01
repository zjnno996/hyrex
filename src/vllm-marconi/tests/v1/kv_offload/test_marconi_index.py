# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for the Marconi prefix index."""

from vllm.v1.kv_offload.policies.marconi_index import MarconiIndex


def _insert(index: MarconiIndex, path: tuple[str, ...], **kwargs: object) -> None:
    index.insert(path, state_kind="full_kv", token_count=1024, **kwargs)


def test_common_prefix_is_materialized_once_and_longest_lookup_works():
    index = MarconiIndex()
    _insert(index, ("system",), byte_size=50, last_access_ms=0)
    _insert(index, ("system", "user-a"), byte_size=100, last_access_ms=1)
    _insert(index, ("system", "user-b"), byte_size=100, last_access_ms=2)

    assert index.lookup(("system", "user-a", "tail")) == ("system", "user-a")
    assert index.materialized_bytes == 250
    assert len(index.nodes()) == 3


def test_shared_parent_is_not_evicted_before_descendant():
    index = MarconiIndex()
    _insert(
        index,
        ("system", "old"),
        byte_size=100,
        last_access_ms=1,
        compute_savings_ms=1,
    )
    _insert(
        index,
        ("system", "new"),
        byte_size=100,
        last_access_ms=99,
        compute_savings_ms=100,
    )

    evicted = index.select_evictions(100, now_ms=100)
    assert evicted == (("system", "old"),)
    index.remove(evicted[0])
    assert index.get(("system",)) is not None
    assert index.lookup(("system", "new")) == ("system", "new")


def test_eviction_merges_a_single_child_into_the_parent_edge():
    index = MarconiIndex()
    _insert(
        index,
        ("one", "two"),
        byte_size=100,
        last_access_ms=1,
        compute_savings_ms=1,
    )
    _insert(
        index,
        ("one", "two", "three"),
        byte_size=100,
        last_access_ms=2,
        compute_savings_ms=2,
    )
    victim = index.select_evictions(100, now_ms=10)
    assert victim == (("one", "two"),)
    index.remove(victim[0])
    assert index.lookup(("one", "two", "three")) == ("one", "two", "three")
    assert index.lookup(("one", "two")) == ()
    assert index.materialized_bytes == 100


def test_touch_only_refreshes_the_matched_node():
    index = MarconiIndex()
    _insert(index, ("system",), byte_size=50, last_access_ms=1)
    _insert(index, ("system", "user-a"), byte_size=100, last_access_ms=2)
    index.touch(("system", "user-a", "tail"), now_ms=20)
    assert index.get(("system",)).last_access_ms == 1
    assert index.get(("system", "user-a")).last_access_ms == 20
