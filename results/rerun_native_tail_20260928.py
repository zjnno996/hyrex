"""Contemporaneous ABBA rerun using the frozen eager recovery harness."""
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

ROOT = Path('/root/hyrex_results/native_tail_abba_20260928_v3')
FROZEN = Path('/root/hyrex_results/single_forward_3arm_4s10t_20260927_v1')
SOURCE = '/root/exp-vllm-short-tail'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def status(state, **details):
    (ROOT / 'status.json').write_text(json.dumps(
        {'state': state, 'updated_unix': time.time(), 'pid': os.getpid(), **details}, indent=2))


def main():
    ROOT.mkdir(exist_ok=False)
    status('running')
    summary = {}
    collected = {'baseline': [], 'tail': []}
    reference = read_rows(FROZEN / 'baseline/online.jsonl')[:40]
    for index, arm in enumerate(('baseline', 'tail', 'tail', 'baseline')):
        # Previous harness owns and shuts down its processes; wait for their ports.
        for attempt in range(60):
            busy = False
            for port in (8761, 8762, 8763):
                with socket.socket() as sock:
                    sock.settimeout(0.2)
                    busy |= sock.connect_ex(('127.0.0.1', port)) == 0
            if not busy:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Experiment ports remain occupied')
        name = f'{index + 1}_{arm}'
        out = ROOT / name
        out.mkdir()
        command = json.loads((FROZEN / arm / 'command.json').read_text())
        changes = {'--output-dir': str(out), '--online-repetitions': '1'}
        if arm == 'tail':
            changes['--vllm-source'] = SOURCE
        for flag, value in changes.items():
            command[command.index(flag) + 1] = value
        env = {k: v for k, v in os.environ.items()
               if 'HYREX' not in k and not k.startswith('LMCACHE_')}
        env.update(PYTHONSAFEPATH='1', CUDA_VISIBLE_DEVICES='1')
        if arm == 'tail':
            env.update(VLLM_HYREX_SINGLE_FORWARD='1',
                       LMCACHE_HYREX_LAST_STATE_ONLY='1',
                       LMCACHE_HYREX_FULL_LOAD_TO_STATE='0',
                       LMCACHE_HYREX_BATCH_FULL_PAGES='1',
                       LMCACHE_HYREX_COALESCE_FULL_PAGES='1',
                       VLLM_HYREX_Q_ONLY_REPLAY='1', VLLM_HYREX_SHORT_TAIL='1')
        source = command[command.index('--vllm-source') + 1]
        cache_source = command[command.index('--experimental-lmcache-source') + 1]
        env['PYTHONPATH'] = f'{cache_source}:{source}'
        metadata = {'command': command, 'environment': {
            k: v for k, v in env.items() if 'HYREX' in k or k in
            ('PYTHONPATH', 'PYTHONSAFEPATH', 'CUDA_VISIBLE_DEVICES')},
            'commits': {p: subprocess.check_output(['git', '-C', p, 'rev-parse', 'HEAD'], text=True).strip()
                        for p in (source, cache_source)}}
        (out / 'command.json').write_text(json.dumps(metadata, indent=2))
        print('Starting', name, flush=True)
        status('running', arm=name)
        with (out / 'runner.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = read_rows(out / 'online.jsonl')
        resets = read_rows(out / 'resets.jsonl')
        assert len(rows) == 40 and len(resets) == 39
        assert all(r['success'] for r in resets)
        assert len(read_rows(out / 'warmup.jsonl')) == 10
        assert all((r['session_id'], r['turn_index'], r['prompt_tokens'], r['first_text']) ==
                   (o['session_id'], o['turn_index'], o['prompt_tokens'], o['first_text'])
                   for r, o in zip(rows, reference)), 'Correctness/trace mismatch'
        log = (out / 'vllm.log').read_text()
        parts = log.split('Successfully reset prefix cache')
        assert len(parts) >= 11
        jit = ''.join(parts[10:]).count('Triton kernel JIT compilation during inference')
        assert jit == 0, 'Formal requests encountered JIT'
        collected[arm].extend(rows)
        summary[name] = {'requests': len(rows), 'first_token_checks_passed': True,
                         'formal_jit_warnings': jit,
                         'mean_ttft_ms': statistics.mean(r['ttft_ms'] for r in rows),
                         'continuation_mean_ttft_ms': statistics.mean(r['ttft_ms'] for r in rows if r['turn_index'] > 0)}
        (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(name, summary[name], flush=True)
    pooled = {arm: statistics.mean(r['ttft_ms'] for r in rows if r['turn_index'] > 0)
              for arm, rows in collected.items()}
    summary['pooled_continuation_mean_ms'] = pooled
    summary['tail_reduction_percent'] = 100 * (1 - pooled['tail'] / pooled['baseline'])
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    status('completed', requests=160)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if ROOT.exists():
            status('failed', error=repr(exc))
        raise
