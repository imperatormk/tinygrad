import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from test.mockgpu.qcom.emu import ftz, Threads, Dispatch, Src, Image, exec_isam, A0

class TestEmulatorFloat(unittest.TestCase):
  def test_flush_bits(self):
    rng = np.random.default_rng(42)
    for dt, ui, fraction, exponent in [(np.float16, np.uint16, 10, 5), (np.float32, np.uint32, 23, 8), (np.float64, np.uint64, 52, 11)]:
      sign, exp = 1 << (fraction + exponent), ((1 << exponent) - 1) << fraction
      edges = [0, 1, (1 << fraction) - 1, 1 << fraction, (1 << fraction) + 1, exp - 1, exp, exp + 1, exp + (1 << (fraction - 1))]
      bits = np.arange(65536, dtype=ui) if dt == np.float16 else np.concatenate((
        np.array(edges + [v | sign for v in edges], dtype=ui), rng.integers(0, np.iinfo(ui).max, 4096, dtype=ui)))
      expected = bits.copy()
      subnormal = ((bits & ui(exp)) == 0) & ((bits & ui((1 << fraction) - 1)) != 0)
      expected[subnormal] &= ui(sign)
      for stride in (1, 3, -1):
        with self.subTest(dtype=dt, stride=stride), np.errstate(all='ignore'):
          values = bits.view(dt)[::stride]
          values.flags.writeable = False
          before = values.tobytes()
          actual = ftz(values)
          self.assertEqual(actual.dtype, values.dtype)
          self.assertEqual(actual.shape, values.shape)
          self.assertEqual(actual.tobytes(), expected[::stride].tobytes())
          self.assertEqual(values.tobytes(), before)

  def test_empty_and_integer(self):
    for dt in (np.float16, np.float32, np.float64, np.int32):
      values = np.empty(0, dt)
      self.assertEqual(ftz(values).tobytes(), values.tobytes())
    values = np.array([-1, 0, 1], np.int32)
    np.testing.assert_array_equal(ftz(values), values)

  def test_float_reads_after_overlapping_writes(self):
    rng = np.random.default_rng(19)
    for merged in (False, True):
      d = Dispatch(b'', np.zeros(4, np.uint32), (32,1,1), (1,1,1), 0xfc, 0xfc, 0, 0, [], [], [], False, [], 0, merged)
      t = Threads(d, 32)
      with np.errstate(all='ignore'):
        for step in range(400):
          half, reg = bool(rng.integers(2)), int(rng.choice([0, 1, 2, 3, 4, 5, A0]))
          dt, ui = np.dtype('f2' if half else 'f4'), np.dtype('u2' if half else 'u4')
          raw = rng.integers(0, np.iinfo(ui).max, 32, dtype=ui)
          raw[:4] = [0, 1, 1 << (dt.itemsize * 8 - 1), (1 << (dt.itemsize * 8 - 1)) + 1]
          clean = step % 3 == 0
          value = ftz(raw.view(dt)) if clean else raw
          t.mask = t.everyone if step % 2 else rng.integers(2, size=32).astype(bool)
          t.write(reg, half, value, float_clean=clean)
          before = t.r.tobytes(), b'' if merged else t.h.tobytes()
          for read_half in (False, True):
            for read_reg in (reg, reg // 2, 2 * reg if reg < 6 else reg):
              for absneg in range(4):
                src = Src('r', read_reg, read_half, absneg)
                dtype = np.dtype('f2' if read_half else 'f4')
                self.assertEqual(t.read_alu(src, dtype).tobytes(), ftz(t.read(src, dtype)).tobytes(), (merged, step, src))
          self.assertEqual(before, (t.r.tobytes(), b'' if merged else t.h.tobytes()))

  def test_cached_float_bits_and_modifiers(self):
    for half in (False, True):
      dt, ui = np.dtype('f2' if half else 'f4'), np.dtype('u2' if half else 'u4')
      raw = np.arange(65536, dtype=ui) if half else np.random.default_rng(23).integers(0, 2**32, 65536, dtype=ui)
      d = Dispatch(b'', np.zeros(4, np.uint32), (len(raw),1,1), (1,1,1), 0xfc, 0xfc, 0, 0, [], [], [], False, [], 0, False)
      t = Threads(d, len(raw))
      t.write(0, half, raw)
      with np.errstate(all='ignore'):
        for _ in range(2):
          for absneg in range(4):
            src = Src('r', 0, half, absneg)
            self.assertEqual(t.read_alu(src, dt).tobytes(), ftz(t.read(src, dt)).tobytes())
      self.assertEqual(t.read(Src('r', 0, half), ui).tobytes(), raw.tobytes())

  def test_texture_rows_and_live_memory(self):
    for dtype in (np.dtype('f2'), np.dtype('f4')):
      for row_elements in (12, 13, 16): # tightly packed, non-pixel-aligned pitch, padded
        for wrmask in (5, 15):
          img = Image(4096, 3, 3, row_elements * dtype.itemsize, dtype)
          d = Dispatch(b'', np.zeros(4, np.uint32), (6,1,1), (1,1,1), 0xfc, 0xfc, 0, 0, [], [img], [], False, [True], 0, True)
          t, memory = Threads(d, 6), np.arange(row_elements * 3, dtype=dtype)
          x, y = np.array([-1, 0, 1, 2, 3, 0], np.int32), np.array([0, 0, 1, 2, 0, 3], np.int32)
          channels = [c for c in range(4) if wrmask >> c & 1]
          inst = SimpleNamespace(tex=0, s2en=False, samp=0, type=0 if dtype.itemsize == 2 else 1, src1=0, dst=0, wrmask=wrmask)
          with patch('test.mockgpu.qcom.emu.to_mv', lambda addr, size: memoryview(memory).cast('B')):
            for increment in (0, 10):
              memory += increment # cached views must observe writes between texture instructions
              t.r[0], t.r[1] = x.view(np.uint32), y.view(np.uint32)
              expected = np.array([[memory[yy * row_elements + xx * 4 + c] if 0 <= xx < 3 and 0 <= yy < 3 else 0
                                    for c in channels] for xx, yy in zip(x, y)], dtype=dtype)
              exec_isam(t, inst, 0) # destination overlaps the coordinate registers, including merged half aliases
              for n in range(len(channels)):
                actual = (t.h if dtype.itemsize == 2 else t.r)[n].view(dtype)
                self.assertEqual(actual.tobytes(), expected[:, n].tobytes())

if __name__ == '__main__': unittest.main()
