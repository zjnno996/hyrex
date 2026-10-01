"""Run fixed source worktrees sequentially; never switch code during a run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from collections import Counter


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--trace', type=Path,
                   default=Path('/root/hyrex_results/motivation_sharegpt_4s10t_trace.jsonl'))
    p.add_argument('--sessions', type=int, default=4)
    p.add_argument('--min-session-turns', type=int, default=10)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    trace = args.trace
    trace_rows = [json.loads(row) for row in trace.read_text().splitlines()]
    counts = Counter(row['session_id'] for row in trace_rows)
    eligible = [sid for sid, count in counts.items() if count >= args.min_session_turns]
    selected = set(eligible[:args.sessions])
    if len(selected) != args.sessions:
        raise ValueError(f'only {len(selected)} eligible sessions in {trace}')
    expected_requests = sum(row['session_id'] in selected for row in trace_rows)
    variants = [('baseline', 'baseline'), ('shallow', 'decoupled'),
                ('deep_qkv', 'decoupled'), ('deep_qonly', 'kv-opt')]
    results = {}
    for arm, version in variants:
        vllm = Path('/root/exp-vllm-' + version)
        lmcache = Path('/root/exp-lmcache-' + version)
        assert not git(vllm, 'status', '--porcelain'), vllm
        assert not git(lmcache, 'status', '--porcelain'), lmcache
        output = args.output_dir / arm
        output.mkdir()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('LMCACHE_HYREX_', 'LMCACHE_TAIL_', 'VLLM_HYREX_'))}
        settings = {} if arm == 'baseline' else {
            'LMCACHE_HYREX_COALESCE_FULL_PAGES': '1',
            'LMCACHE_HYREX_LAST_STATE_ONLY': '1',
            'LMCACHE_HYREX_FULL_LOAD_TO_STATE': '1' if arm == 'shallow' else '0',
            'VLLM_HYREX_Q_ONLY_REPLAY': '1' if arm == 'deep_qonly' else '0',
            'VLLM_HYREX_Q_ONLY_NO_CAT': '1',
            'VLLM_HYREX_MASK_FULL_KV': '1',
            'LMCACHE_HYREX_VERIFY_FULL_H2D': '0',
        }
        env.update(settings)
        # -m otherwise prepends cwd and can import a different vLLM worktree.
        env['PYTHONSAFEPATH'] = '1'
        env['PYTHONPATH'] = str(lmcache) + os.pathsep + str(vllm)
        with (output / 'imports.log').open('w') as log:
            subprocess.run([sys.executable, '-c',
                'import sys,vllm,lmcache; from pathlib import Path; '
                'assert Path(vllm.__file__).resolve().parent.parent == Path(sys.argv[1]); '
                'assert Path(lmcache.__file__).resolve().parent.parent == Path(sys.argv[2]); '
                'print(vllm.__file__); print(lmcache.__file__)', str(vllm), str(lmcache)],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [sys.executable, str(Path(__file__).with_name('audit_real_sharegpt_mp.py')),
                   '--trace', str(trace), '--mode', 'default', '--workflow', 'online',
                   '--sessions', str(args.sessions), '--min-session-turns', str(args.min_session_turns), '--limit', '0',
                   '--round-robin-sessions', '--reset-between-requests', '--gpu', '1',
                   '--cpu-gb', '2', '--first-token-logprobs', '--max-output-tokens', '1',
                   '--vllm-source', str(vllm), '--experimental-lmcache-source', str(lmcache),
                   '--vllm-port', '8761', '--lmcache-port', '8762',
                   '--lmcache-http-port', '8763', '--output-dir', str(output)]
        if arm != 'baseline':
            command += ['--experimental-full-page-size', '16']
        design = {'arm': arm, 'settings': settings, 'command': command,
                  'PYTHONSAFEPATH': '1', 'PYTHONPATH': env['PYTHONPATH'],
                  'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
                  'scope': 'single sequential pass, not statistical significance',
                  'expected_requests': expected_requests,
                  'baseline_patches': 'reset success response; common CUDA fallback stream-order fix',
                  'sources': {str(path): git(path, 'rev-parse', 'HEAD') for path in (vllm, lmcache)}}
        (output / 'design.json').write_text(json.dumps(design, indent=2) + '\n')
        print('START', arm, flush=True)
        with (output / 'runner.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert all(git(path, 'rev-parse', 'HEAD') == commit
                   and not git(path, 'status', '--porcelain')
                   for path, commit in design['sources'].items())
        rows = [json.loads(x) for x in (output / 'online.jsonl').read_text().splitlines()]
        resets = [json.loads(x) for x in (output / 'resets.jsonl').read_text().splitlines()]
        assert len(rows) == expected_requests and len(resets) == expected_requests - 1 and all(x['success'] for x in resets)
        continued = [r for r in rows if r['turn_index'] > 0]
        assert len(continued) == expected_requests - args.sessions
        results[arm] = {'mean_ttft_ms': statistics.mean(r['ttft_ms'] for r in continued),
                        'min_ttft_ms': min(r['ttft_ms'] for r in continued),
                        'max_ttft_ms': max(r['ttft_ms'] for r in continued)}
        kinds = sorted({r.get('request_kind') for r in continued} - {None})
        if kinds:
            results[arm]['by_request_kind'] = {
                kind: statistics.mean(r['ttft_ms'] for r in continued
                                      if r.get('request_kind') == kind)
                for kind in kinds
            }
        (args.output_dir / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print('DONE', arm, results[arm], flush=True)


if __name__ == '__main__':
    main()
