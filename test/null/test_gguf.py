import struct, unittest
from tinygrad import dtypes, Tensor, Device
from tinygrad.llm.gguf import ggml_data_to_tensor, gguf_shard, _bulk_load
supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

@unittest.skipUnless(dtypes.uint8 in supported_dtypes and dtypes.half in supported_dtypes, "Backend must support uint8 and half")
class TestGGUF(unittest.TestCase):
  def test_expected_failure_unknown_type(self):
    with self.assertRaises(ValueError):
      ggml_data_to_tensor(Tensor.empty(512, dtype=dtypes.uint8), 256, 1337)

  def test_gguf_shard_bulk_load(self):
    vals_a, vals_b, vals_c = [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [5.0, 6.0, 7.0, 8.0]
    raw = Tensor(b"\x00" * 16 + struct.pack("<4f4f", *vals_a, *vals_b) + b"\x00" + struct.pack("<4f", *vals_c), device="CPU").realize()
    a, b, c, empty = (raw[16:32], (4,), 0), (raw[32:48], (4,), 0), (raw[49:65], (4,), 0), (raw[48:48], (0,), 0)
    for entries, exp, bulk in (({"a": a, "b": b}, {"a": vals_a, "b": vals_b}, True),
                               ({"a": a, "b": b, "c": c}, {"a": vals_a, "b": vals_b, "c": vals_c}, False),
                               ({"a": a, "c": c}, {"a": vals_a, "c": vals_c}, False),
                               ({"a": a, "b": b, "e": empty}, {"a": vals_a, "b": vals_b, "e": []}, False)):
      self.assertEqual(bool(_bulk_load(entries, "CPU")), bulk)
      self.assertEqual({k: v.tolist() for k, v in gguf_shard(entries, ("CPU",)).items()}, exp)
    self.assertEqual(gguf_shard({}, ("CPU",)), {})

if __name__ == '__main__':
  unittest.main()
