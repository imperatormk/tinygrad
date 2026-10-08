import struct, subprocess, unittest
from unittest.mock import patch
import numpy as np
from test.mockgpu.qcom import emu, emu_native as native


def alu(op, dst=0, a=0, b=1, c=2, repeat=0, modifiers=0):
  if op == emu.mesa.OPC_MAD_F32:
    return (3 << 61) | ((op & 127) << 55) | (b << 47) | (repeat << 40) | (dst << 32) | (c << 16) | a | (modifiers << 14)
  return (2 << 61) | ((op & 127) << 53) | (1 << 52) | (repeat << 40) | (dst << 32) | (b << 16) | a | (modifiers << 14)

def image(*words): return struct.pack('<' + 'Q'*len(words), *words)
def dispatch(code, n, merged=False):
  return emu.Dispatch(code, np.zeros(4, np.uint32), (n,1,1), (1,1,1), 0xfc, 0xfc, 0, 0, [], [], [], False, [], 0, merged)


class TestNativeALU(unittest.TestCase):
  def test_bits_masks_aliases_and_repeats(self):
    words = [alu(emu.mesa.OPC_MAD_F32, repeat=3, modifiers=1), alu(emu.mesa.OPC_MUL_F, dst=1, modifiers=3),
             alu(emu.mesa.OPC_ADD_F, dst=0, a=1, modifiers=2)]
    code = image(*words)
    block, fn = native.plan(code, 0)[0], native.kernels(code, 0)[0]
    rng = np.random.default_rng(51)
    edges = np.array([0, 0x80000000, 1, 0x80000001, 0x7fffff, 0x807fffff, 0x800000, 0x80800000,
                      0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7f7fffff, 0x7fc00001, 0xff800001], np.uint32)
    with np.errstate(all='ignore'):
      for merged in (False, True):
        for n in (1, 4, 257):
          raw = rng.integers(0, 2**32, (256,n), dtype=np.uint32)
          raw[:8] = rng.choice(edges, (8,n))
          for mode in ('all', 'partial', 'none'):
            with self.subTest(merged=merged, n=n, mode=mode):
              states = [emu.Threads(dispatch(code, n, merged), n) for _ in range(2)]
              for t in states:
                t.r[:] = raw
                if mode != 'all': t.mask = np.arange(n) % 2 == 0 if mode == 'partial' else np.zeros(n, bool)
                for reg in range(12): t.read_alu(emu.Src('r', reg, True), np.dtype('f2'))
              for i,k in block.ops: emu.exec_alu(states[0], i, k)
              block.execute(states[1], fn)
              np.testing.assert_array_equal(states[0].r, states[1].r)
              for reg in range(12):
                s = emu.Src('r', reg, True)
                self.assertEqual(states[1].read_alu(s, np.dtype('f2')).tobytes(), emu.ftz(states[0].read(s, np.dtype('f2'))).tobytes())

  def test_mad_rounding(self):
    code = image(alu(emu.mesa.OPC_MAD_F32, dst=3))
    block, fn = native.plan(code, 0)[0], native.kernels(code, 0)[0]
    n, rng = 65537, np.random.default_rng(72)
    with np.errstate(all='ignore'):
      for mode in ('bits', 'cancellation', 'subnormal_product'):
        t = emu.Threads(dispatch(code, n), n)
        t.r[:3] = rng.integers(0, 2**32, (3,n), dtype=np.uint32)
        if mode == 'cancellation':
          t.r[:2] = rng.uniform(.5, 2, (2,n)).astype(np.float32).view(np.uint32)
          t.r[2] = (-(t.r[0].view(np.float32) * t.r[1].view(np.float32))).view(np.uint32)
        if mode == 'subnormal_product':
          t.r[:2] = np.full((2,n), 2**-65, np.float32).view(np.uint32)
          t.r[2] = np.full(n, 2**-126, np.float32).view(np.uint32)
        a,b,c = [emu.ftz(v.view(np.float32)) for v in t.r[:3]]
        expected = emu.canonical_nan(emu.ftz(emu.ftz(a*b)+c)).view(np.uint32)
        block.execute(t, fn)
        np.testing.assert_array_equal(t.r[3], expected)

  def test_plan_and_divergent_entry(self):
    # Half the lanes branch directly into the middle of a contiguous ALU sequence.
    code = image((1 << 55) | 2, alu(emu.mesa.OPC_ADD_F), alu(emu.mesa.OPC_MUL_F), emu.mesa.OPC_END << 55)
    self.assertEqual({pc:b.end for pc,b in native.plan(code, 0).items()}, {1:2, 2:3})
    self.assertIn(2, native.plan(image(alu(emu.mesa.OPC_ADD_F), alu(emu.mesa.OPC_ADD_F), alu(emu.mesa.OPC_ADD_F)), 2))
    original, results = emu.Threads, []
    def threads(d,n):
      t = original(d,n)
      t.r[0], t.r[1] = np.full(n, 2, np.float32).view(np.uint32), np.full(n, 3, np.float32).view(np.uint32)
      t.r[emu.P0] = np.arange(n) % 2
      results.append(t)
      return t
    with patch.object(emu, 'Threads', threads):
      emu.run(dispatch(code, 9))
      with patch.object(native, 'plan', return_value={}): emu.run(dispatch(code, 9))
    np.testing.assert_array_equal(results[0].r, results[1].r)

  def test_support_and_compile_failure(self):
    for word in (alu(emu.mesa.OPC_ADD_F) | (1 << 42), alu(emu.mesa.OPC_ADD_F) | (1 << 46),
                 alu(emu.mesa.OPC_ADD_F) & ~(1 << 52), alu(emu.mesa.OPC_ADD_F, dst=emu.A0), alu(emu.mesa.OPC_ADD_F) | (1 << 12)):
      self.assertFalse(native.supported(emu.decode_inst(0, word)))
    with patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'cc')):
      with self.assertRaises(subprocess.CalledProcessError): native.compile_source('/* deliberate compile failure */')


if __name__ == '__main__': unittest.main()
