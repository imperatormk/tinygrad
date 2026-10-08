import unittest
from unittest.mock import patch
from tinygrad import Tensor, Context, Variable, Device
from test.helpers import needs_second_gpu

class TestValidateWithCPU(unittest.TestCase):
  def setUp(self):
    self.ctx = Context(VALIDATE_WITH_CPU=1)
    self.ctx.__enter__()
  def tearDown(self): self.ctx.__exit__(None, None, None)

  def test_add(self): self.assertListEqual((Tensor([1.,2,3])+Tensor([4.,5,6])).tolist(), [5.0, 7.0, 9.0])
  def test_mul(self): self.assertListEqual((Tensor([1.,2,3])*Tensor([4.,5,6])).tolist(), [4.0, 10.0, 18.0])
  def test_sum(self): self.assertEqual(Tensor([1.,2,3,4]).sum().item(), 10.0)
  def test_reduce_then_op(self): self.assertEqual((Tensor([1.,2,3,4]).sum() * 2).item(), 20.0)

  def test_assign(self):
    a = Tensor([1.,2,3]).realize()
    a.assign(a + 1).realize()
    self.assertListEqual(a.tolist(), [2.0, 3.0, 4.0])

  def test_buffer_view(self):
    self.assertListEqual((Tensor([1.,2,3,4,5,6,7,8])[2:6] + 1).tolist(), [4.0, 5.0, 6.0, 7.0])

  def test_symbolic(self):
    i = Variable('i', 1, 10)
    ones = Tensor.ones(10).contiguous()
    self.assertListEqual((ones[:i.bind(5)] + 1).contiguous()[:5].tolist(), [2.0]*5)

  def test_multi_kernel(self):
    a = (Tensor([1.,2,3]) + 1).contiguous()
    b = (a * 2).contiguous()
    self.assertListEqual((b - 1).tolist(), [3.0, 5.0, 7.0])

  def test_references_precompiled(self):
    import numpy as np
    from tinygrad.engine import realize
    from tinygrad.uop.ops import Ops
    from tinygrad.codegen import to_program_key, to_program_cache
    from tinygrad.engine.worker import terminate_worker_pool
    for workers in (0, 2):
      with self.subTest(workers=workers), Context(PARALLEL=workers):
        to_program_cache.clear()
        a = Tensor([1.,2,3]).realize()
        out = (a + 7).contiguous()
        linear = realize.compile_linear(out.schedule_linear(), validate=True)
        validators = [c for c in linear.toposort(enter_calls=False)
                      if c.op is Ops.CALL and c.body.op is Ops.CUSTOM_FUNCTION and c.body.arg.name == "validate"]
        self.assertTrue(validators)
        for c in validators:
          self.assertIn(to_program_key(c.body.src[0], Device["CPU"].renderer), to_program_cache)
        linear = realize.link_linear(linear)
        original = realize.to_program
        def cached_only(ast, renderer):
          self.assertIn(to_program_key(ast, renderer), to_program_cache)
          return original(ast, renderer)
        try:
          with patch.object(realize, 'to_program', side_effect=cached_only), \
               patch.object(np.testing, 'assert_allclose', wraps=np.testing.assert_allclose) as check:
            realize.run_linear(linear, jit=True)
            self.assertTrue(check.called)
          self.assertEqual(out.tolist(), [8.,9.,10.])
        finally: terminate_worker_pool()

  @needs_second_gpu
  def test_sharded(self):
    t = Tensor([1.,2,3,4]).shard((f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"), axis=0)
    self.assertListEqual((t + 1).tolist(), [2.0, 3.0, 4.0, 5.0])

if __name__ == "__main__":
  unittest.main()
