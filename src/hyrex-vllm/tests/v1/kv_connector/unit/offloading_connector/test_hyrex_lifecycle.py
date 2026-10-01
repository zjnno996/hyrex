# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    native_cache_event,
)
from vllm.v1.kv_offload.worker.worker import TransferResult


def test_native_load_completion_keeps_request_fence(caplog):
    connector = object.__new__(OffloadingConnectorWorker)
    connector.worker = MagicMock()
    connector.worker.get_finished.return_value = [
        TransferResult(7, True, 1024, 0.002, ("CPU", "GPU"), 0.001)
    ]
    connector.kv_connector_stats = OffloadingConnectorStats()
    connector._connector_worker_meta = OffloadingWorkerMetadata()
    connector._load_jobs = {7: "cmpl-hyrex-0-0"}
    connector._store_jobs = {}
    connector._job_started = {}

    _, finished = connector.get_finished(set())

    assert finished == {"cmpl-hyrex-0-0"}
    assert connector._load_jobs == {}
    assert connector._connector_worker_meta.completed_jobs == {7: 1}
    assert connector._connector_worker_meta.transfer_observations == {
        7: (1024, 2.0, 1.0)
    }
    assert "HYREX_NATIVE_TRANSFER_EVENT" in caplog.text
    assert '"retrieve_bytes": 1024' in caplog.text
    assert '"retrieve_observed_ms": 2.0' in caplog.text
    assert '"retrieve_fence_completed": true' in caplog.text


def test_native_lookup_uses_common_cache_state_schema():
    assert native_cache_event("r0", 528, 1056, "all_load") == {
        "request_id": "r0",
        "gpu_hit_tokens": 528,
        "cpu_hit_tokens": 1584,
        "h2d_tokens": 1056,
        "cache_state": "partial",
        "recovery_action": "load_all",
    }
