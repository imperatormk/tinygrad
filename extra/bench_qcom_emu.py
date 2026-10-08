"""Small, frozen QCOM instruction replays for emulator performance work.

One-time capture (requires a working MOCK+QCOM compiler):
  PYTHONPATH=. DEV=MOCK+QCOM:IR3 python extra/bench_qcom_emu.py capture /tmp/qcom-ir3
  PYTHONPATH=. DEV=MOCK+QCOM:CL PARALLEL=0 python extra/bench_qcom_emu.py capture /tmp/qcom-cl
Fast replay (only Python + numpy; no compiler, GPU, or model download):
  PYTHONPATH=. python extra/bench_qcom_emu.py run /tmp/qcom-ir3 /tmp/qcom-cl --json /tmp/baseline.json
  PYTHONPATH=. python extra/bench_qcom_emu.py run /tmp/qcom-ir3 /tmp/qcom-cl --compare /tmp/baseline.json

Capture keeps the slowest dispatch of each small workload, with its actual
shader, constants, memory inputs and expected memory outputs. Replay resets
memory before each invocation and checks every captured byte afterward, outside
the timed region. Decode gets one untimed warmup. Times include dispatch setup
and execution; they exclude compilation, file I/O, reset and comparison.
These are unweighted microbenchmarks, not an estimate of whole-model speedup.
Goldens preserve baseline emulator behavior; they are not a CPU/hardware oracle.
Fixtures/results are local artifacts; keep the same fixtures for comparisons.
The snapshots use a bounds-checked address resolver, not live GPU mappings.
This measures the instruction emulator, not packet parsing or QCOM runtime.
For less noisy measurements increase --target-ms and --repeats. For profiling:
  PYTHONPATH=. python -m cProfile -o /tmp/emu.prof extra/bench_qcom_emu.py run /tmp/qcom-ir3 --repeats 1 --target-ms 1
"""
import argparse, dataclasses, hashlib, json, math, platform, statistics, sys, time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test.mockgpu.qcom import emu

def digest(arrays):
  h = hashlib.sha256()
  for a in arrays: h.update(memoryview(a).cast('B'))
  return h.hexdigest()

def capture(directory:Path, only:str|None):
  from tinygrad import Tensor, Device, Context, dtypes
  from tinygrad.helpers import DEV
  from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
  if not DEV.interface.startswith('MOCK') or Device.DEFAULT != 'QCOM': raise ValueError('capture requires DEV=MOCK+QCOM:IR3 or :CL')
  directory.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(42)
  def rand(shape): return Tensor(rng.standard_normal(shape).astype(np.float32)).realize()
  def barrier(C, A):
    i, j = UOp.range(8, 0, AxisType.LOOP), UOp.range(8, 1, AxisType.LOCAL)
    return C[i].store(A[i, j].reduce(j, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=()))
  x, w = rand((1, 4, 8, 8)), rand((8, 4, 3, 3))
  dx, dw = rand((1, 8, 8, 8)), rand((8, 1, 3, 3))
  a, b, red, values = rand((16, 32)), rand((32, 16)), rand((16, 128)), rand((4096,))
  indices = Tensor(rng.integers(0, 4096, 256, dtype=np.int32)).realize()
  bar, stream = rand((8, 8)), rand((65536,))
  cases = [('image_conv', lambda: x.conv2d(w, padding=1)),
           ('depthwise', lambda: dx.conv2d(dw, padding=1, groups=8)),
           ('gemm_gelu', lambda: (a @ b).gelu()), ('reduce', lambda: red.sum(axis=1)),
           ('gather', lambda: values[indices]), ('half_cast', lambda: values.cast(dtypes.half)),
           ('stream_alu', lambda: (stream * 1.125 + .25).relu()),
           ('barrier', lambda: Tensor.custom_kernel(Tensor.empty(8), bar, fxn=barrier)[0])]
  if only and only not in dict(cases): raise ValueError(f'unknown case {only}')
  original_run, original_mv = emu.run, emu.to_mv
  for label, build in cases:
    if only and label != only: continue
    target = directory/f'{label}.npz'
    if target.exists(): raise FileExistsError(f'preserving existing fixture {target}')
    candidates = []
    def record(d):
      if sum(sz for _, sz in d.ranges) > 128 << 20: raise ValueError('capture memory budget exceeded')
      before = {addr:np.frombuffer(original_mv(addr, sz), np.uint8).copy() for addr, sz in d.ranges}
      touched = set()
      def tracked_mv(addr, size):
        for base, n in d.ranges:
          if base <= addr and addr+size <= base+n:
            touched.add(base)
            break
        else: raise ValueError(f'unmapped capture access {addr:#x}+{size}')
        return original_mv(addr, size)
      start = time.perf_counter_ns()
      with patch.object(emu, 'to_mv', tracked_mv): original_run(d)
      elapsed = time.perf_counter_ns()-start
      ranges = [(addr, sz) for addr, sz in d.ranges if addr in touched]
      pre = [before[addr] for addr, _ in ranges]
      post = [np.frombuffer(original_mv(addr, sz), np.uint8).copy() for addr, sz in ranges]
      if not any(np.any(a != b) for a, b in zip(pre, post)): return
      candidates.append((elapsed, dataclasses.replace(d, consts=d.consts.copy(), ranges=ranges), pre, post))
    with Context(IMAGE=1, FLOAT16=1), patch.object(emu, 'run', record):
      build().realize()
      Device.default.synchronize()
    if label in ('image_conv', 'depthwise', 'gemm_gelu'): candidates = [c for c in candidates if c[1].textures]
    if not candidates: raise ValueError(f'{label}: no suitable dispatch changed memory (image cases require textures)')
    elapsed, d, pre, post = max(candidates, key=lambda c:c[0])
    instructions = emu.decode(d.image)
    if label in ('image_conv', 'depthwise', 'gemm_gelu') and not any(isinstance(i, emu.Cat5) for i in instructions):
      raise ValueError(f'{label}: selected shader has no texture instructions')
    if label == 'barrier' and not any(i.op == emu.mesa.OPC_BAR for i in instructions if isinstance(i, emu.Inst)):
      raise ValueError('selected barrier shader has no barrier instructions')
    meta = dataclasses.asdict(d)
    meta.pop('image')
    meta.pop('consts')
    for key in ('textures', 'ibos'):
      for img in meta[key]: img['dtype'] = str(img['dtype'])
    meta.update(version=1, name=label, renderer=str(DEV), capture_ms=elapsed/1e6,
                selected_from=len(candidates), expected_sha256=digest(post),
                emulator_sha256=hashlib.sha256(Path(emu.__file__).read_bytes()).hexdigest())
    arrays = {'shader':np.frombuffer(d.image, np.uint8), 'consts':d.consts,
              **{f'before_{i}':v for i,v in enumerate(pre)}, **{f'after_{i}':v for i,v in enumerate(post)}}
    np.savez_compressed(target, metadata=json.dumps(meta), **arrays)
    print(f'{target}: {elapsed/1e6:.2f} ms, {math.prod(d.local_size)*math.prod(d.groups)} lanes, {sum(v.nbytes for v in pre)} bytes', flush=True)

def replay(path:Path, repeats:int, target_ms:float):
  with np.load(path, allow_pickle=False) as z:
    meta = json.loads(str(z['metadata']))
    if meta['version'] != 1: raise ValueError('unsupported fixture version')
    pre = [z[f'before_{i}'].copy() for i in range(len(meta['ranges']))]
    post = [z[f'after_{i}'].copy() for i in range(len(pre))]
    args = {f.name:meta[f.name] for f in dataclasses.fields(emu.Dispatch) if f.name not in ('image', 'consts')}
    for key in ('textures', 'ibos'): args[key] = [emu.Image(**(v | {'dtype':np.dtype(v['dtype'])})) for v in args[key]]
    d = emu.Dispatch(image=z['shader'].tobytes(), consts=z['consts'].copy(), **args)
  instructions = [i for i in emu.decode(d.image) if isinstance(i, emu.Inst)]
  if digest(post) != meta['expected_sha256']: raise ValueError(f'{path}: corrupt expected output')
  work = [v.copy() for v in pre]
  def mapped_mv(addr, size):
    for (base, n), buf in zip(d.ranges, work):
      if base <= addr and addr+size <= base+n: return memoryview(buf)[addr-base:addr-base+size]
    raise ValueError(f'{path}: unmapped replay access {addr:#x}+{size}')
  def once():
    for dst, src in zip(work, pre): np.copyto(dst, src)
    start = time.perf_counter_ns()
    emu.run(d)
    elapsed = time.perf_counter_ns()-start
    for i, (actual, expected) in enumerate(zip(work, post)):
      if not np.array_equal(actual, expected):
        raise AssertionError(f'{path}: buffer {i} changed at {np.count_nonzero(actual != expected)} bytes')
    return elapsed/1e6
  with patch.object(emu, 'to_mv', mapped_mv):
    warm = once()
    iterations = max(1, min(100, math.ceil(target_ms/max(warm, .001))))
    samples = [statistics.mean(once() for _ in range(iterations)) for _ in range(repeats)]
  return dict(name=f"{meta['renderer']}/{meta['name']}", fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              median_ms=statistics.median(samples), samples_ms=samples, iterations=iterations,
              lanes=math.prod(d.local_size)*math.prod(d.groups), shader_bytes=len(d.image), memory_bytes=sum(x.nbytes for x in pre),
              image_instructions=sum(isinstance(i, emu.Cat5) for i in instructions), barriers=sum(i.op == emu.mesa.OPC_BAR for i in instructions))

def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='command', required=True)
  cap = sub.add_parser('capture')
  cap.add_argument('directory', type=Path)
  cap.add_argument('--case', help='capture only this workload')
  run = sub.add_parser('run')
  run.add_argument('directories', nargs='+', type=Path)
  run.add_argument('--repeats', type=int, default=3)
  run.add_argument('--target-ms', type=float, default=15)
  run.add_argument('--json', type=Path)
  run.add_argument('--compare', type=Path)
  args = p.parse_args()
  if args.command == 'capture':
    from tinygrad.helpers import Context
    with Context(IMAGE=1, FLOAT16=1): return capture(args.directory, args.case)
  if args.repeats < 1 or args.target_ms <= 0: p.error('repeats and target-ms must be positive')
  files = sorted(f for directory in args.directories for f in directory.rglob('*.npz'))
  if not files: p.error('no captured .npz fixtures found')
  baseline = json.loads(args.compare.read_text()) if args.compare else None
  if baseline and baseline.get('host') != platform.node(): p.error('baseline host differs; do not compare timings across machines')
  start = time.perf_counter()
  rows = []
  for f in files:
    row = replay(f, args.repeats, args.target_ms)
    if any(r['name'] == row['name'] for r in rows): p.error(f"duplicate case {row['name']}")
    suffix = ''
    if baseline:
      old = next((r for r in baseline['cases'] if r['name'] == row['name']), None)
      if old is None or old['fixture_sha256'] != row['fixture_sha256']: p.error(f'{f}: baseline fixture mismatch')
      row['speedup'] = old['median_ms']/row['median_ms']
      suffix = f"  {row['speedup']:.2f}x baseline"
    print(f"{row['name']:50s} {row['median_ms']:9.3f} ms  exact PASS{suffix}", flush=True)
    rows.append(row)
  if baseline and {r['name'] for r in baseline['cases']} != {r['name'] for r in rows}: p.error('baseline case set differs')
  result = dict(version=1, platform=platform.platform(), machine=platform.machine(), host=platform.node(), python=sys.version.split()[0],
                emulator_sha256=hashlib.sha256(Path(emu.__file__).read_bytes()).hexdigest(),
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                numpy=np.__version__, wall_seconds=time.perf_counter()-start, cases=rows)
  if args.json:
    with args.json.open('x') as f: f.write(json.dumps(result, indent=2)+'\n')
  if baseline: print("Quick-run ratios are noisy; confirm modest gains with --repeats 5 --target-ms 50")
  print(f"{len(rows)} cases, {result['wall_seconds']:.2f}s total (including resets/checks); no model-wide speedup inferred")

if __name__ == '__main__': main()
