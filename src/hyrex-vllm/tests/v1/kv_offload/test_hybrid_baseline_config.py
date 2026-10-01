# SPDX-License-Identifier: Apache-2.0

from benchmarks.reproductions.hybrid_baseline_config import BASELINES, baseline_config


def test_hybrid_baselines_have_isolated_branches_and_explicit_semantics():
    assert set(BASELINES) == {
        "marconi", "tail_replay", "kvpr_hybrid", "cacheflow_hybrid", "hyrex"
    }
    assert len({entry.branch for entry in BASELINES.values()}) == len(BASELINES)
    assert baseline_config("tail_replay")["native_hybrid"] is True
    assert baseline_config("kvpr_hybrid")["native_hybrid"] is False
