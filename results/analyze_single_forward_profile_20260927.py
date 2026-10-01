"""Summarize diagnostic traces; kernel sums are not end-to-end TTFT."""
from collections import Counter
import gzip
import json
from pathlib import Path

root = Path('/root/hyrex_results/single_forward_profile_20260927_v1')
report = {}
for arm in ('baseline', 'deep', 'tail'):
    traces = list((root / arm / 'profile').glob('*.gz'))
    assert len(traces) == 1, (arm, traces)
    events = json.loads(gzip.decompress(traces[0].read_bytes()))['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    counts = Counter(e['name'] for e in events if e.get('cat') == 'cpu_op')
    duration = Counter()
    for e in events:
        if e.get('cat') == 'cpu_op':
            duration[e['name']] += e.get('dur', 0) / 1000
    mm = {e['args']['External id']: e for e in events
          if e.get('cat') == 'cpu_op' and e.get('name') == 'aten::mm'}
    launches = {e['args']['correlation']: e['args'].get('External id') for e in events
                if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in e.get('args', {})}
    shapes = Counter()
    for e in kernels:
        op = mm.get(launches.get(e.get('args', {}).get('correlation')))
        if op is not None:
            shapes[str(op['args'].get('Input Dims'))] += e.get('dur', 0) / 1000
    rows = [json.loads(line) for line in (root / arm / 'online.jsonl').read_text().splitlines()]
    row = rows[5]
    assert row['session_id'] == 'Pr8nMeM_0' and row['turn_index'] == 1 and row['prompt_tokens'] == 831
    if arm != 'baseline':
        assert row['first_text'] == report['baseline']['first_text']
    report[arm] = {
        'first_text': row['first_text'], 'cached_tokens': row['cached_tokens'],
        'kernel_count': len(kernels), 'kernel_ms_sum': sum(e['dur'] for e in kernels)/1000,
        'ops': {name: {'count': counts[name], 'inclusive_cpu_ms': duration[name]} for name in (
            'ChunkGatedDeltaRuleFunction', 'aten::_local_scalar_dense', 'aten::item',
            'aten::is_nonzero', 'aten::clone', 'aten::cat', 'aten::mm', 'aten::copy_')},
        'gemm_kernel_ms_by_shape': dict(shapes),
        'forward_cpu_ranges': [(e['name'], e.get('dur', 0)/1000) for e in events
                               if e.get('cat') == 'user_annotation' and 'execute_context_1' in e['name']],
    }
report['limitations'] = 'Diagnostic single-request traces, profiler overhead and first-profile effects; no additive TTFT decomposition. LMCache-process H2D is not fully captured. CPU operator times are inclusive and overlap.'
(root / 'profile_summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
