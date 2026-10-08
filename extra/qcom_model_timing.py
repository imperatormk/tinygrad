"""Opt-in, parent-process stage accounting for the isolated openpilot CI experiment (Python 3.12+)."""
import inspect, json, os, sys, threading, time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

class ModelTiming:
  def __init__(self, output):
    self.output, self.started, self.owner = output, time.perf_counter(), threading.get_ident()
    self.stack, self.events, self.codes = [], [], {}
    self.stats = defaultdict(lambda: {"calls": 0, "inclusive_s": 0.0, "exclusive_s": 0.0, "max_s": 0.0})
    self.tool = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(self.tool, "qcom-model-timing")
    self._push("total_instrumented")
    self.emit("start", "total_instrumented")
    with self.stage("instrumentation_setup"):
      self.install()

  def emit(self, event, name, **values):
    row = {"event": event, "stage": name, "elapsed_s": time.perf_counter()-self.started,
           "utc": datetime.now(timezone.utc).isoformat(), **values}
    self.events.append(row)
    print("MODEL_TIMING " + json.dumps(row), flush=True)

  def _push(self, name): self.stack.append([name, time.perf_counter(), 0.0])

  def _pop(self):
    name, start, children = self.stack.pop()
    elapsed = time.perf_counter()-start
    if self.stack: self.stack[-1][2] += elapsed
    row = self.stats[tuple(x[0] for x in self.stack)+(name,)]
    row["calls"] += 1
    row["inclusive_s"] += elapsed
    row["exclusive_s"] += elapsed-children
    row["max_s"] = max(row["max_s"], elapsed)
    return elapsed

  @contextmanager
  def stage(self, name):
    self.emit("start", name)
    self._push(name)
    try: yield
    finally: self.emit("end", name, wall_s=self._pop())

  def _start(self, code, offset):
    if threading.get_ident() == self.owner: self._push(self.codes[code])

  def _end(self, code, offset, value):
    if code in self.codes and threading.get_ident() == self.owner:
      assert self.stack[-1][0] == self.codes[code], (self.stack, code)
      self._pop()

  def install(self):
    from tinygrad import Tensor
    from tinygrad.device import Compiler, Buffer
    from tinygrad.engine import realize, jit
    from tinygrad import schedule
    from tinygrad.schedule.memory import memory_plan_rewrite
    from tinygrad.runtime.ops_cpu import CPUProgram
    from test.mockgpu.qcom import emu, emu_native
    functions = [realize.lower_and_compile, realize.compile_linear, realize.link_linear, realize.exec_validate,
                 realize.exec_kernel, realize.exec_copy, realize.exec_hcq, realize.get_runtime, realize.to_program,
                 Compiler.compile_cached, CPUProgram.__call__, schedule.lower_sink_to_linear, memory_plan_rewrite,
                 jit.jit_lower, Tensor.numpy, Buffer.ensure_allocated, emu.run, emu_native.compile_source]
    for fn in functions:
      fn = inspect.unwrap(fn)
      # rewrite_group does not expose __wrapped__; never monitor its code shared by unrelated functions.
      if fn.__qualname__ == 'rewrite_group.<locals>._decorator.<locals>.__wrapper':
        fn = dict(zip(fn.__code__.co_freevars, (cell.cell_contents for cell in fn.__closure__)))['func']
      self.codes[fn.__code__] = fn.__module__ + "." + fn.__qualname__
    for event, callback in ((sys.monitoring.events.PY_START, self._start), (sys.monitoring.events.PY_RETURN, self._end),
                            (sys.monitoring.events.PY_UNWIND, self._end)):
      sys.monitoring.register_callback(self.tool, event, callback)
    sys.monitoring.set_events(self.tool, sys.monitoring.events.PY_UNWIND)
    flags = sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN
    for code in self.codes: sys.monitoring.set_local_events(self.tool, code, flags)

  def finish(self, success):
    for code in self.codes: sys.monitoring.set_local_events(self.tool, code, 0)
    sys.monitoring.set_events(self.tool, 0)
    sys.monitoring.free_tool_id(self.tool)
    assert len(self.stack) == 1, self.stack
    elapsed = self._pop()
    rows = [{"path": list(path), **values} for path, values in self.stats.items()]
    exclusive = sum(row["exclusive_s"] for row in rows)
    assert abs(exclusive-elapsed) < 1e-6, (exclusive, elapsed)
    self.emit("end", "total_instrumented", wall_s=elapsed, exclusive_sum_s=exclusive, success=success)
    # Inclusive rows overlap. Exclusive rows partition the instrumented parent wall time, including waits.
    for row in sorted(rows, key=lambda x: x["exclusive_s"], reverse=True)[:30]:
      print("MODEL_TIMING_COST " + json.dumps(row), flush=True)
    data = {"success": success, "wall_s": elapsed, "exclusive_sum_s": exclusive, "pid": os.getpid(),
            "python": sys.version, "cpu_count": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "env": {k: os.environ.get(k) for k in ("DEV", "PARALLEL", "CCACHE", "BEAM_MAX_TASKS_PER_CHILD", "VALIDATE_WITH_CPU", "GITHUB_SHA")},
            "scope": "Parent main thread only; compiler-worker work is included as parent waiting, not separately attributed. "
                     "Instrumentation overhead is included. Imports before initialization and process shutdown are outside this total.",
            "events": self.events, "costs": rows}
    Path(self.output).write_text(json.dumps(data, indent=2)+"\n")
