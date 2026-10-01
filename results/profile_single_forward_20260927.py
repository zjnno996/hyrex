"""Reuse frozen commands; profile the same real second-turn request in each arm."""
import json
import os
from pathlib import Path
import subprocess
import sys

source = Path('/root/hyrex_results/single_forward_3arm_4s10t_20260927_v1')
root = Path('/root/hyrex_results/single_forward_profile_20260927_v1')
root.mkdir(exist_ok=False)
for arm in ('baseline', 'deep', 'tail'):
    command = json.loads((source / arm / 'command.json').read_text())
    out = root / arm
    out.mkdir()
    for flag, value in (('--limit', '8'), ('--online-repetitions', '1'), ('--output-dir', str(out))):
        command[command.index(flag) + 1] = value
    command += ['--profile-request-index', '5']
    env = {k: v for k, v in os.environ.items() if 'HYREX' not in k and not k.startswith('LMCACHE_TAIL_')}
    vllm = command[command.index('--vllm-source') + 1]
    lmcache = command[command.index('--experimental-lmcache-source') + 1]
    env.update(PYTHONSAFEPATH='1', PYTHONPATH=f'{lmcache}:{vllm}', CUDA_VISIBLE_DEVICES='1')
    if arm != 'baseline':
        env.update(VLLM_HYREX_SINGLE_FORWARD='1', LMCACHE_HYREX_LAST_STATE_ONLY='1',
                   LMCACHE_HYREX_FULL_LOAD_TO_STATE='0', LMCACHE_HYREX_BATCH_FULL_PAGES='1',
                   LMCACHE_HYREX_COALESCE_FULL_PAGES='1', VLLM_HYREX_Q_ONLY_REPLAY='1')
    (out / 'command.json').write_text(json.dumps(command, indent=2))
    print('Profiling', arm, flush=True)
    with (out / 'runner.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed', arm, flush=True)
