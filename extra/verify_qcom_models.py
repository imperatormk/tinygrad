"""Same-runner before/after/before verification; keeps all model checks and records host conditions."""
import argparse, hashlib, importlib.metadata, json, os, platform, subprocess, sys, threading, time
from pathlib import Path
from tinygrad.helpers import fetch

LFS_SHA = '51b12d20a25b76fa8a5717c2e5af01f148e469e2'

def command(args):
  try:
    p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {'returncode': p.returncode, 'output': p.stdout}
  except OSError as e: return {'error': str(e)}

def snapshot():
  paths = ['/proc/stat', '/proc/loadavg', '/proc/meminfo', '/proc/pressure/cpu', '/proc/pressure/memory',
           '/sys/fs/cgroup/cpu.stat', '/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.peak']
  return {'monotonic_s': time.monotonic(), **{p: Path(p).read_text() for p in paths if Path(p).exists()}}

def verify(before, after, output, models):
  output.mkdir(parents=True, exist_ok=False)
  metadata = {'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
              'affinity': sorted(os.sched_getaffinity(0)), 'lscpu': command(['lscpu', '--json']),
              'cc': command(['cc', '--version']), 'clang': command(['clang', '--version']),
              'qemu': command(['qemu-aarch64-static', '--version']),
              'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])),
              'numpy_runtime': command([sys.executable, '-c', 'import numpy as np; np.show_runtime(); np.show_config()']),
              'cpuinfo': Path('/proc/cpuinfo').read_text(), 'env': {k: os.environ.get(k) for k in
                ['DEV', 'PARALLEL', 'CCACHE', 'OMP_NUM_THREADS', 'BEAM_MAX_TASKS_PER_CHILD', 'ImageVersion', 'ImageOS']}}
  (output/'host.json').write_text(json.dumps(metadata, indent=2)+'\n')
  # Share immutable downloads, not program caches. Download/extract outside all timed model processes.
  if os.environ.get('DEV', '').endswith(':CL'):
    fetch('https://git.tinygrad.win/tinygrad/images/releases/download/v2/qcomcl.tar.gz', extract=True)
  # Avoid charging the first variant for bytecode creation in newly checked-out worktrees.
  for checkout in dict.fromkeys([before, after]):
    subprocess.run([sys.executable, '-m', 'compileall', '-q', *[str(checkout/p) for p in
                    ['tinygrad', 'examples/openpilot', 'test/mockgpu', 'extra/qcom_model_timing.py']]], check=True)
  results = []
  for model in models:
    path = Path(model) if Path(model).exists() else fetch(
      f'https://huggingface.co/commaai/openpilot-lfs/resolve/{LFS_SHA}/openpilot/selfdrive/modeld/models/{model}.onnx')
    path = path.resolve()
    model_name = Path(model).stem
    model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for label, checkout in [('before1', before), ('after', after), ('before2', before)]:
      dest = output/model_name/label
      dest.mkdir(parents=True)
      env = os.environ | {'PYTHONPATH': str(checkout), 'PARALLEL': '4', 'CCACHE': '0', 'CACHEDB': str(dest/'cache.sqlite3'),
                          'BEAM_MAX_TASKS_PER_CHILD': '256', 'VALIDATE_WITH_CPU': '1', 'IMAGE': '1', 'FLOAT16': '1', 'PYTHONUNBUFFERED': '1'}
      identity = {'revision': command(['git', '-C', str(checkout), 'rev-parse', 'HEAD']),
                  'realize_sha256': hashlib.sha256((checkout/'tinygrad/engine/realize.py').read_bytes()).hexdigest(),
                  'model_sha256': model_hash, 'processes': command(['ps', '-eo', 'pid,ppid,stat,comm'])}
      (dest/'before.json').write_text(json.dumps({'identity': identity, 'snapshot': snapshot()}, indent=2)+'\n')
      done = threading.Event()
      def sample():
        with (dest/'host-samples.jsonl').open('w') as stream:
          while not done.is_set():
            stream.write(json.dumps(snapshot())+'\n')
            stream.flush()
            done.wait(1)
      sampler = threading.Thread(target=sample)
      sampler.start()
      cmd = ['/usr/bin/time', '-f', 'process_wall_s=%e\npeak_rss_kib=%M', '-o', str(dest/'process-time.txt'),
             sys.executable, str(checkout/'examples/openpilot/compile_onnx.py'), str(path), str(dest/'model.pkl'),
             '--benchmark-runs', '1', '--timings', str(dest/'timings.json')]
      print(f'::group::{model_name} {label}', flush=True)
      started = time.monotonic()
      try:
        with (dest/'model.log').open('w') as log:
          with subprocess.Popen(cmd, cwd=checkout, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
              log.write(line)
              if line.startswith('MODEL_TIMING ') or '[0/1]' in line or 'Traceback' in line: print(line, end='', flush=True)
            returncode = proc.wait()
      finally:
        elapsed = time.monotonic()-started
        done.set()
        sampler.join()
        (dest/'after.json').write_text(json.dumps({'snapshot': snapshot(), 'processes': command(['ps', '-eo', 'pid,ppid,stat,comm'])}, indent=2)+'\n')
        print('::endgroup::', flush=True)
      row = {'model': model_name, 'variant': label, 'returncode': returncode, 'process_wall_s': elapsed}
      results.append(row)
      (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
      print('VERIFY_RESULT '+json.dumps(row), flush=True)
      if returncode: raise RuntimeError(f'{model_name} {label} exited {returncode}; see {dest}/model.log')
      timing = json.loads((dest/'timings.json').read_text())
      assert timing['success'] and abs(sum(r['exclusive_s'] for r in timing['costs'])-timing['wall_s']) < 1e-6
  return results

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--before', type=Path, required=True)
  parser.add_argument('--after', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--model', action='append', help='model name or local ONNX path; default: both openpilot models')
  args = parser.parse_args()
  verify(args.before.resolve(), args.after.resolve(), args.output.resolve(), args.model or ['dmonitoring_model', 'driving_supercombo'])
