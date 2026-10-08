"""Compiler/model-free ALU register-read benchmark (normally a few seconds).

  PYTHONPATH=. python extra/bench_qcom_emu_reads.py --baseline /path/to/emu-before.py

Reports time per read INCLUDING amortized writes, with 1 or 8 reads per write.
Each case verifies cached reads against uncached FTZ and checks raw-bit preservation.
Results isolate register handling, not instruction arithmetic or model speed.
"""
import argparse, importlib.util, json, statistics, sys, time
from pathlib import Path
import numpy as np
from test.mockgpu.qcom import emu

def run_case(module, lanes, half, case, target_ms):
  dt, ui = np.dtype('f2' if half else 'f4'), np.dtype('u2' if half else 'u4')
  d = module.Dispatch(b'', np.zeros(4, np.uint32), (lanes,1,1), (1,1,1), 0xfc, 0xfc, 0, 0, [], [], [], False, [], 0,
                      case == 'alias')
  t = module.Threads(d, lanes)
  values = np.ones(lanes, dt).view(ui).copy()
  if case == 'zeros': values[:] = np.resize(np.array([0, 1 << (ui.itemsize*8-1)], ui), lanes)
  if case in ('subnormal', 'mixed', 'masked', 'alias'): values[::1 if case == 'subnormal' else 4] = 1
  if case == 'masked': t.mask = np.arange(lanes) % 2 == 0
  reuse = 1 if case == 'write_each' else 8
  sources = [module.Src('r', 0, half, j % 4) for j in range(reuse)]
  clean = case == 'clean'
  write_half = not half if case == 'alias' else half
  write_values = values.astype(np.uint16 if write_half else np.uint32) if case == 'alias' else values.view(dt) if clean else values
  def write(): t.write(0, write_half, write_values, float_clean=clean)
  def burst():
    write()
    for src in sources: t.read_alu(src, dt)
  # First populate the cache, then rewrite so the check exercises invalidation too.
  burst()
  write()
  before = t.r.tobytes(), b'' if d.merged else t.h.tobytes()
  for src in sources:
    assert t.read_alu(src, dt).tobytes() == module.ftz(t.read(src, dt)).tobytes(), (lanes, half, case, src)
  assert before == (t.r.tobytes(), b'' if d.merged else t.h.tobytes())
  start, count = time.perf_counter_ns(), 0
  while time.perf_counter_ns()-start < target_ms*1e6:
    for _ in range(8): burst()
    count += 8*reuse
  return (time.perf_counter_ns()-start)/count/1000

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path)
  parser.add_argument('--lanes', nargs='+', type=int, default=[1024])
  parser.add_argument('--target-ms', type=float, default=5)
  parser.add_argument('--json', type=Path)
  args = parser.parse_args()
  modules = [('candidate', emu)]
  if args.baseline:
    spec = importlib.util.spec_from_file_location('read_benchmark_baseline', args.baseline)
    assert spec is not None and spec.loader is not None
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    modules = [('baseline', baseline), ('candidate', emu), ('candidate', emu), ('baseline', baseline)]
  results = []
  with np.errstate(all='ignore'):
    for lanes in args.lanes:
      for half in (False, True):
        for case in ('normal', 'zeros', 'subnormal', 'mixed', 'write_each', 'masked', 'alias', 'clean'):
          row = {'lanes':lanes, 'dtype':'f16' if half else 'f32', 'case':case, 'us_per_read':{}}
          for name, module in modules:
            row['us_per_read'].setdefault(name, []).append(run_case(module, lanes, half, case, args.target_ms))
          if args.baseline: row['speedup'] = statistics.median(row['us_per_read']['baseline'])/statistics.median(row['us_per_read']['candidate'])
          results.append(row)
          print(json.dumps(row), flush=True)
  if args.json: args.json.write_text(json.dumps(results, indent=2)+'\n')

if __name__ == '__main__': main()
