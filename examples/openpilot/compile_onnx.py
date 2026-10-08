"""Compile an ONNX model into a TinyJit artifact."""
import argparse, pickle
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from tinygrad import Tensor, Device, TinyJit, Context, dtypes
from tinygrad.dtype import _to_np_dtype
from tinygrad.helpers import fetch, DEBUG
from tinygrad.nn.onnx import OnnxPBParser, OnnxRunner
from tinygrad.engine.realize import lower_and_compile
from examples.openpilot.helpers import allocate_inputs, dump_pickle, load_pickle, make_retargetable, benchmark

def onnx_metadata(path):
  parser = OnnxPBParser(path, load_external_data=False)
  metadata, output_shapes = {}, {}
  for field, wire_type in parser._parse_message(parser.reader.len):
    if field == 14:
      entry = parser._parse_StringStringEntryProto()
      metadata[entry['key']] = entry['value']
    elif field == 7:
      # Read output declarations without parsing graph nodes or weight tensors.
      for field, wire_type in parser._parse_message(parser._decode_end_pos()):
        if field == 12:
          value = parser._parse_ValueInfoProto()
          output_shapes[value['name']] = value['parsed_type'].shape if value['parsed_type'] is not None else ()
        else: parser.reader.skip_field(wire_type)
    else: parser.reader.skip_field(wire_type)
  return metadata, output_shapes

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('onnx')
  parser.add_argument('output')
  parser.add_argument('--benchmark-runs', type=int, default=20)
  parser.add_argument('--out-of-band', action='store_true', help='stream protocol-5 buffers for large models')
  parser.add_argument('--retargetable', action='store_true', help='output retargetable jit (requires recompilation when loading)')
  parser.add_argument('--timings', help='write detailed stage timing JSON (isolated CI diagnostic, Python 3.12+)')
  args = parser.parse_args()
  timing = None
  if args.timings:
    from extra.qcom_model_timing import ModelTiming
    timing = ModelTiming(args.timings)
  stage = timing.stage if timing else lambda _: nullcontext()
  success = False
  try:
    with stage('fetch_model'): path = fetch(args.onnx) if '://' in args.onnx else Path(args.onnx)

    with stage('parse_onnx'): runner = OnnxRunner(path)
    with stage('read_metadata'): properties, output_shapes = onnx_metadata(path)
    metadata = {'metadata': properties, 'input_shapes': {k:v.shape for k,v in runner.graph_inputs.items()}, 'output_shapes': output_shapes}

    def get_specs(d): return {k:(t.shape, np.dtype(_to_np_dtype(t.dtype)).name, Device.DEFAULT) for k,t in d.items()}
    with stage('infer_output_specs'): output_specs = get_specs(runner(allocate_inputs(specs:=get_specs(runner.graph_inputs))))

    def make_inputs(seed):
      rng = np.random.default_rng(seed)
      def initialize(arrays):
        for name, value in arrays.items():
          dtype = runner.graph_inputs[name].dtype
          value[...] = (rng.standard_normal(value.shape) if dtypes.is_float(dtype) else
                        rng.integers(0, 256, value.shape, dtype=np.uint8) if dtype == dtypes.uint8 else
                        rng.integers(0, 2 if dtype == dtypes.bool else 16, value.shape))
      return allocate_inputs(specs, initialize) | {'output_buffers': allocate_inputs(output_specs)}

    @TinyJit(prune=True)
    def run(output_buffers, **inputs):
      outputs = runner({k:v.to(Device.DEFAULT) for k,v in inputs.items()})
      Tensor.realize(*(output_buffers[k].assign(v) for k,v in outputs.items()))

    with stage('prepare_inputs_42'): inputs = make_inputs(42)
    with stage('first_execution'), Context(DEBUG=max(DEBUG.value, 1)): expected = benchmark(run, **inputs)
    for label in ('jit_capture', 'jit_replay'):
      with stage(label): actual = benchmark(run, **inputs)
      with stage(label+'_assert'): np.testing.assert_equal(actual, expected)
    with stage('prepare_inputs_43'): changed_inputs = make_inputs(43)
    with stage('changed_input_execution'): actual = benchmark(run, **changed_inputs)
    with stage('changed_input_assert'):
      with np.testing.assert_raises(AssertionError): np.testing.assert_equal(actual, expected)
    del changed_inputs, actual
    for i in range(args.benchmark_runs):
      with stage(f'warm_benchmark_{i}'):
        actual = benchmark(run, cb=lambda t: print(f"  [{i}/{args.benchmark_runs}] {t*1e3:.2f} ms", flush=True), **inputs)
      with stage(f'warm_benchmark_{i}_assert'): np.testing.assert_equal(actual, expected)

    if args.retargetable: make_retargetable(run)

    artifact = {'metadata': metadata, 'run': run, 'input_specs': specs, 'output_specs': output_specs}

    with stage('pickle_write'):
      if args.out_of_band: dump_pickle(artifact, args.output)
      else:
        with open(args.output, 'wb') as f: pickle.dump(artifact, f)

    # test pickled jit
    with stage('pickle_read'): loaded = load_pickle(args.output, out_of_band=args.out_of_band)
    with stage('retarget_compile'):
      if args.retargetable: loaded['run'].captured._linear = lower_and_compile(loaded['run'].captured._linear)
    with stage('prepare_reload_inputs_42'): reload_inputs = make_inputs(42)
    with stage('reloaded_execution'): actual = benchmark(loaded['run'], **reload_inputs)
    with stage('reloaded_assert'): np.testing.assert_equal(actual, expected)

    success = True
  finally:
    if timing: timing.finish(success)
