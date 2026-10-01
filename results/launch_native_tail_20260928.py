"""Launch the recovery comparison directly, independent of the tool terminal."""
import subprocess
from pathlib import Path

root = Path('/root/hyrex_results/native_tail_abba_20260928_v3')
if root.exists():
    raise SystemExit('Output already exists; refusing a duplicate launch')
with Path(str(root) + '.driver.log').open('x') as log:
    process = subprocess.Popen(
        ['/root/hybrid-model-offloading/.venv/bin/python', '-u',
         '/root/hyrex_results/rerun_native_tail_20260928.py'],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, cwd='/root',
    )
print(f'Independent driver PID: {process.pid}', flush=True)
