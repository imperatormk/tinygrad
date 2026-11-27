import functools, itertools, math
from dataclasses import dataclass
from typing import Callable
import numpy as np
from tinygrad.helpers import Context, getbits
from tinygrad.runtime.autogen import mesa
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import DType, dtypes, AddrSpace
from tinygrad.device import Device
from tinygrad.engine.realize import get_runtime
from tinygrad.codegen import to_program

NREGS, A0, P0, MAX_LOCAL = 64 * 4, 61 * 4, 62 * 4, 1024
HALF_TYPES = (0, 2, 4, 6, 7) # f2, f4, u2, u4, i2, i4, u1, i1
FLUT = [0.0, 0.5, 1.0, 2.0, math.e, math.pi, 1/math.pi, 1/math.log2(math.e), math.log2(math.e), 1/math.log2(10), math.log2(10), 4.0]
REG_INIT = 0

def sext(x:int, n:int) -> int: return x - (1 << n) if x & (1 << (n - 1)) else x

class Field:
  def __init__(self, lo:int, hi:int|None=None): self.lo, self.hi = lo, lo if hi is None else hi
  def __get__(self, i, owner=None) -> int: return getbits(i.word, self.lo, self.hi)

@dataclass(frozen=True)
class Src: kind:str; val:int; half:bool = False; absneg:int = 0; r:bool = False # noqa: E702

def multisrc(x:int, half:bool, r:int) -> Src:
  absneg = getbits(x, 14, 15)
  if getbits(x, 11, 13) == 0b000: return Src("r", getbits(x, 0, 7), half, absneg, bool(r))
  if getbits(x, 10, 13) == 0b0011 and not half: return Src("c<", sext(getbits(x, 0, 9), 10), half, absneg)
  if getbits(x, 11, 12) == 0b10: return Src("c", getbits(x, 0, 10), half, absneg, bool(r))
  if getbits(x, 11, 13) == 0b100: return Src("i", sext(getbits(x, 0, 10), 11), half, absneg)
  if getbits(x, 11, 13) == 0b101:
    v = FLUT[getbits(x, 0, 9)]
    return Src("i", int(np.float16(v).view(np.uint16)) if getbits(x, 10, 10) else int(np.float32(v).view(np.uint32)), half, absneg)
  raise NotImplementedError(f"multisrc encoding {x:#x}")

def cat3src(x:int, half:bool, immed:int, r:int, neg:int) -> Src:
  if getbits(x, 11, 12) == 0b00: return Src("r", getbits(x, 0, 7), half, neg, bool(r))
  if getbits(x, 12, 12): return Src("i", getbits(x, 0, 11), half, neg) if immed else Src("c", getbits(x, 0, 10), half, neg, bool(r))
  raise NotImplementedError(f"cat3 source encoding {x:#x}")

class Inst:
  cat, opc, repeat = Field(61, 63), Field(55, 58), 0
  srcs:list[Src]
  def __init__(self, pc:int, word:int):
    self.pc, self.word = pc, word
    self.op, self.iterations = self.cat << 7 | self.opc, self.repeat + 1
  def error(self): return NotImplementedError(f"pc {self.pc}: {mesa.opc_t.get(self.op, self.op)} is not emulated {self.word:#x}")

class Cat0(Inst):
  immed, brtype, inv2, comp2 = Field(0, 31), Field(37, 39), Field(45), Field(46, 47)
  opc_hi, inv1, comp1 = Field(49), Field(52), Field(53, 54)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.op = mesa.OPC_BR + self.brtype if (self.opc_hi, self.opc) == (0, 1) else self.opc_hi << 4 | self.opc
    self.target = pc + sext(self.immed, 32)
    if self.op not in CTRL_IDX: raise self.error()

class Cat1(Inst):
  dst, repeat, src_r, dst_type, dst_rel = Field(32, 39), Field(40, 41), Field(43), Field(46, 48), Field(49)
  src_type, src_mode, round, multi = Field(50, 52), Field(53, 54), Field(55, 56), Field(57, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    half = self.src_type in HALF_TYPES
    if (self.multi, self.dst_rel) == (0b10, 0):
      self.op, self.iterations = mesa.OPC_SWZ + getbits(word, 40, 41), 1
      if self.op != mesa.OPC_SWZ: raise self.error()
      self.srcs, self.dsts = [Src("r", getbits(word, 0, 7), half), Src("r", getbits(word, 8, 15), half)], [self.dst, getbits(word, 16, 23)]
      return
    self.op = mesa.OPC_MOV
    if self.dst_rel or self.multi or self.src_mode == 0b11 or self.round > 1: raise self.error()
    if self.src_mode == 0b00:
      if getbits(word, 11, 11):
        if not getbits(word, 10, 10) or half: raise self.error()
        self.srcs = [Src("c<", sext(getbits(word, 0, 9), 10))]
      else: self.srcs = [Src("r", getbits(word, 0, 7), half, r=bool(self.src_r))]
    elif self.src_mode == 0b01: self.srcs = [Src("c", getbits(word, 0, 10), half, r=bool(self.src_r))]
    else: self.srcs = [Src("i", getbits(word, 0, 31), half)]

class Cat2(Inst):
  dst, repeat, sat, src1_r, dst_conv, ei = Field(32, 39), Field(40, 41), Field(42), Field(43), Field(46), Field(47)
  cond, src2_r, full, opc = Field(48, 50), Field(51), Field(52), Field(53, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if self.op not in ALU_OPS or (self.op in CMPS and self.cond >= 6): raise self.error()
    if self.ei and (self.op != mesa.OPC_ADD_U or not self.full): raise self.error()
    if self.sat and self.op not in CMPS and self.op not in SAT_INT and self.op not in FLOAT_OPS: raise self.error()
    self.dst_half = (not self.full) ^ bool(self.dst_conv)
    self.srcs = [multisrc(getbits(word, 0, 15), not self.full, self.src1_r)]
    if self.op not in CAT2_1SRC: self.srcs.append(multisrc(getbits(word, 16, 31), not self.full, self.src2_r))

class Cat3(Inst):
  src1, alt, src1_neg, src2_r = Field(0, 12), Field(13), Field(14), Field(15)
  src3, src3_r, src2_neg, src3_neg = Field(16, 28), Field(29), Field(30), Field(31)
  dst, repeat, sat_full, src1_r, dst_conv, src2 = Field(32, 39), Field(40, 41), Field(42), Field(43), Field(46), Field(47, 54)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.op += 8 * self.alt
    if self.op not in ALU_OPS: raise self.error()
    half = not self.sat_full if self.alt else self.op in CAT3_HALF
    self.sat, self.dst_half = 0 if self.alt else self.sat_full, half ^ bool(self.dst_conv)
    if self.op == mesa.OPC_MAD_U16: self.dst_half = not self.dst_conv
    if self.sat and self.op not in FLOAT_OPS: raise self.error()
    self.srcs = [cat3src(self.src1, half, self.alt, self.src1_r, self.src1_neg), Src("r", self.src2, half, self.src2_neg, bool(self.src2_r)),
                 cat3src(self.src3, half, self.alt, self.src3_r, self.src3_neg)]

class Cat4(Inst):
  dst, repeat, sat, src_r, dst_conv, full, opc = Field(32, 39), Field(40, 41), Field(42), Field(43), Field(46), Field(52), Field(53, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if self.op not in SFU_IDX: raise self.error()
    self.dst_half, self.srcs = (not self.full) ^ bool(self.dst_conv), [multisrc(getbits(word, 0, 15), not self.full, self.src_r)]

class Cat5(Inst):
  full, src1, samp, tex, dst = Field(0), Field(1, 8), Field(21, 24), Field(25, 31), Field(32, 39)
  src3, desc_mode, wrmask, type, flags, opc = Field(21, 28), Field(29, 31), Field(40, 43), Field(44, 46), Field(48, 52), Field(54, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.s2en = self.flags == 0b01000 and self.desc_mode == 0
    if self.op != mesa.OPC_ISAM or (self.flags and not self.s2en) or not self.full: raise self.error()

class Load(Inst):
  off, addr, size, dst, type, opc = Field(1, 13), Field(14, 21), Field(24, 31), Field(32, 39), Field(49, 51), Field(54, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.offset = sext(self.off, 13)

class Store(Inst):
  val, off_hi, size, off_lo, dst_off = Field(1, 8), Field(9, 13), Field(24, 31), Field(32, 39), Field(40)
  addr, type, opc = Field(41, 48), Field(49, 51), Field(54, 58)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.offset = sext(self.off_hi << 8 | self.off_lo, 13) if self.dst_off else 0

def reg_offset(i:Load|Store, word:int, src2:int) -> tuple[int, int, int]:
  if getbits(word, 11, 11): raise i.error()
  i.offset = 0
  return src2, getbits(word, 12, 13), getbits(word, 9, 10)

class Ldg(Load):
  size = Field(24, 26)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if not getbits(word, 0, 0) or getbits(word, 52, 53): raise self.error()
    self.reg_off = reg_offset(self, word, getbits(word, 1, 8)) if getbits(word, 22, 22) else None

class Stg(Store):
  size = Field(24, 26)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if not getbits(word, 23, 23) or getbits(word, 53, 53): raise self.error()
    self.reg_off = reg_offset(self, word, self.off_lo) if getbits(word, 52, 52) else None

class Ibo(Inst):
  mode, bindless, d_minus_one, typed, type_size_minus_one, opc = Field(6, 7), Field(8), Field(9, 10), Field(11), Field(12, 13), Field(14, 19)
  has_offset, coord, val, ssbo, type = Field(23), Field(24, 31), Field(32, 39), Field(41, 48), Field(49, 51)
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    self.ncomp = self.type_size_minus_one + 1
    if self.op != mesa.OPC_STIB or self.mode or self.bindless or self.has_offset or not self.typed or self.d_minus_one != 1:
      raise self.error()

class Cat7(Inst):
  def __init__(self, pc:int, word:int):
    super().__init__(pc, word)
    if self.op not in (mesa.OPC_BAR, mesa.OPC_FENCE): raise self.error()

CMPS = {mesa.OPC_CMPS_F, mesa.OPC_CMPS_U, mesa.OPC_CMPS_S, mesa.OPC_CMPV_F, mesa.OPC_CMPV_U, mesa.OPC_CMPV_S}
CMPV = {mesa.OPC_CMPV_F, mesa.OPC_CMPV_U, mesa.OPC_CMPV_S}
CAT2_1SRC = {mesa.OPC_SIGN_F, mesa.OPC_ABSNEG_F, mesa.OPC_FLOOR_F, mesa.OPC_TRUNC_F, mesa.OPC_ABSNEG_S, mesa.OPC_NOT_B, mesa.OPC_CLZ_B,
             mesa.OPC_CLZ_S, mesa.OPC_SETRM}
BITWISE = {mesa.OPC_AND_B, mesa.OPC_OR_B, mesa.OPC_XOR_B, mesa.OPC_NOT_B}
SAT_INT = {mesa.OPC_ADD_U, mesa.OPC_ADD_S, mesa.OPC_SUB_U, mesa.OPC_SUB_S}
CAT3_HALF = {mesa.OPC_MAD_F16, mesa.OPC_SEL_B16, mesa.OPC_SEL_S16}
CATS = {0: Cat0, 1: Cat1, 2: Cat2, 3: Cat3, 4: Cat4, 5: Cat5, 7: Cat7}
CAT6 = {mesa.OPC_LDG: Ldg, mesa.OPC_STG: Stg, mesa.OPC_LDL: Load, mesa.OPC_LDP: Load, mesa.OPC_STL: Store, mesa.OPC_STP: Store}

def decode_inst(pc:int, word:int) -> Inst:
  cat = getbits(word, 61, 63)
  if cat == 6 and getbits(word, 52, 53) == 0b10 and getbits(word, 20, 22) == 0b110: return Ibo(pc, word)
  cls = CAT6.get(6 << 7 | getbits(word, 54, 58)) if cat == 6 else CATS.get(cat)
  if cls is None: raise NotImplementedError(f"pc {pc}: cat{cat} is not emulated {word:#x}")
  return cls(pc, word)

# VM bytecode: REC_LEN uint32 per instruction; opcode IDs are indices into the category tables below
REC_LEN = 11
K_UNDEF, K_FMAD32, K_FMAD16, K_FMISC, K_IALU32, K_IMISC, K_SFU, K_MOV, K_MEM, K_TEX, K_CTRL = range(11)
SRC_KIND = {"r": 1, "c": 2, "c<": 3, "i": 4}
ERR_MSGS = {2: "relative const out of range", 3: "out of bounds local access", 4: "out of bounds private access",
            5: "texture is not bound", 6: "IBO is not bound", 7: "call stack overflow", 8: "ret without call"}
FMAD_OPS = {mesa.OPC_MAD_F32, mesa.OPC_MAD_F16, mesa.OPC_ADD_F, mesa.OPC_MUL_F}
SIGNED_IOPS = {mesa.OPC_ADD_S, mesa.OPC_SUB_S, mesa.OPC_MIN_S, mesa.OPC_MAX_S, mesa.OPC_ABSNEG_S, mesa.OPC_CLZ_S,
               mesa.OPC_ASHR_B, mesa.OPC_SEL_S32, mesa.OPC_SEL_S16, mesa.OPC_SAD_S32, mesa.OPC_CMPS_S, mesa.OPC_CMPV_S}
def _idx(groups:tuple[tuple[int, ...], ...]) -> dict[int, int]: return {op: i for i, g in enumerate(groups) for op in g}
FMISC_IDX = _idx(((mesa.OPC_MIN_F,), (mesa.OPC_MAX_F,), (mesa.OPC_SIGN_F,), (mesa.OPC_ABSNEG_F,), (mesa.OPC_FLOOR_F,),
                  (mesa.OPC_TRUNC_F,), (mesa.OPC_SEL_F32,), (mesa.OPC_CMPS_F, mesa.OPC_CMPV_F)))
# Shared integer op indices: 0..24 are fast 32-bit ops (K_IALU32), 25..35 are misc/sat/cmp ops (K_IMISC handles all 36)
INT_IDX = _idx((
  (mesa.OPC_ADD_U,), (mesa.OPC_ADD_S,), (mesa.OPC_SUB_U,), (mesa.OPC_SUB_S,),
  (mesa.OPC_AND_B,), (mesa.OPC_OR_B,), (mesa.OPC_XOR_B,), (mesa.OPC_NOT_B,),
  (mesa.OPC_SHL_B,), (mesa.OPC_SHR_B,), (mesa.OPC_ASHR_B,), (mesa.OPC_GETBIT_B,),
  (mesa.OPC_MUL_S24,), (mesa.OPC_MUL_U24,), (mesa.OPC_MULL_U,),
  (mesa.OPC_MADSH_M16, mesa.OPC_MADSH_U16), (mesa.OPC_MAD_U16,), (mesa.OPC_MAD_S24,),
  (mesa.OPC_SEL_B32, mesa.OPC_SEL_B16), (mesa.OPC_SEL_S32, mesa.OPC_SEL_S16),
  (mesa.OPC_SHRM,), (mesa.OPC_SHRG,), (mesa.OPC_SHLG,), (mesa.OPC_SHLM,), (mesa.OPC_ANDG,),
  (mesa.OPC_MIN_U,), (mesa.OPC_MIN_S,), (mesa.OPC_MAX_U,), (mesa.OPC_MAX_S,), (mesa.OPC_ABSNEG_S,),
  (mesa.OPC_CLZ_B,), (mesa.OPC_CLZ_S,), (mesa.OPC_SETRM,), (mesa.OPC_SAD_S32,),
  (mesa.OPC_CMPS_U, mesa.OPC_CMPV_U), (mesa.OPC_CMPS_S, mesa.OPC_CMPV_S)))
SFU_IDX = _idx(((mesa.OPC_RCP,), (mesa.OPC_RSQ, mesa.OPC_HRSQ), (mesa.OPC_LOG2, mesa.OPC_HLOG2),
                (mesa.OPC_EXP2, mesa.OPC_HEXP2), (mesa.OPC_SIN,), (mesa.OPC_SQRT,)))
MEM_IDX = _idx(((mesa.OPC_LDG,), (mesa.OPC_STG,), (mesa.OPC_LDL,), (mesa.OPC_STL,), (mesa.OPC_LDP,), (mesa.OPC_STP,)))
CTRL_IDX = _idx(((mesa.OPC_NOP, mesa.OPC_PREDE, mesa.OPC_FENCE), (mesa.OPC_END,), (mesa.OPC_JUMP,), (mesa.OPC_BR,), (mesa.OPC_BRAO,),
                 (mesa.OPC_BRAA,), (mesa.OPC_PREDT,), (mesa.OPC_PREDF,), (mesa.OPC_CALL,), (mesa.OPC_RET,), (mesa.OPC_BAR,)))
FLOAT_OPS, ALU_OPS = FMAD_OPS | FMISC_IDX.keys(), FMAD_OPS | FMISC_IDX.keys() | INT_IDX.keys()

def _rec(*xs:int) -> tuple[int, ...]: return xs + (0,) * (REC_LEN - len(xs))
def pack_src(s:Src) -> tuple[int, int]:
  return SRC_KIND[s.kind] | (int(s.half) << 3) | (s.absneg << 4) | (int(s.r) << 6), s.val & 0xFFFFFFFF
def _pack3(srcs:list[Src]) -> tuple[int, ...]:
  ms, vs = zip(*(pack_src(s) for s in srcs)) if srcs else ((), ())
  return (*ms, *(0,) * (3 - len(ms)), *vs, *(0,) * (3 - len(vs)))

def encode_inst(i:Inst, pred_targets:dict[int, int]) -> tuple[int, ...]:
  if isinstance(i, Cat0):
    if i.op in (mesa.OPC_PREDT, mesa.OPC_PREDF):
      return _rec(K_CTRL, CTRL_IDX[i.op], pred_targets[i.pc]) if i.pc in pred_targets else _rec(K_UNDEF)
    return _rec(K_CTRL, CTRL_IDX[i.op], i.target, i.comp1, i.inv1 | (i.inv2 << 1) | (i.comp2 << 4))
  if isinstance(i, Cat7): return _rec(K_CTRL, CTRL_IDX[i.op])
  if isinstance(i, Cat1):
    if i.op == mesa.OPC_SWZ:
      (m0, v0), (m1, v1) = pack_src(i.srcs[0]), pack_src(i.srcs[1])
      return (K_MOV, 1, i.dst, 1, int(i.dst_type in HALF_TYPES), m0, m1, 0, v0, v1, i.dsts[1])
    flags = int(i.dst_type in HALF_TYPES) | (int(i.round == 1) << 2) | (i.src_type << 4) | (i.dst_type << 8)
    return _rec(K_MOV, 0 if i.src_type == i.dst_type and i.src_type not in (6, 7) else 2, i.dst, i.iterations, flags, *_pack3(i.srcs))
  if isinstance(i, (Cat2, Cat3)):
    src_half = i.srcs[0].half
    if i.op in FMAD_OPS:
      srcs = [i.srcs[0], Src("i", 0x3C00 if src_half else 0x3F800000, src_half), i.srcs[1]] if i.op == mesa.OPC_ADD_F else \
             [i.srcs[0], i.srcs[1], Src("i", 0x8000 if src_half else 0x80000000, src_half)] if i.op == mesa.OPC_MUL_F else i.srcs
      return _rec(K_FMAD16 if src_half else K_FMAD32, 0, i.dst, i.iterations, int(i.dst_half) | (int(i.sat) << 1), *_pack3(srcs))
    is_bw, ei = i.op in BITWISE, i.ei if isinstance(i, Cat2) else 0
    cond = i.cond if isinstance(i, Cat2) and i.op in CMPS else 0
    flags = int(i.dst_half) | (int(i.sat) << 1) | (int(i.op in CMPV) << 2) | (int(ei) << 3) | (cond << 4) | \
            (int(is_bw) << 8) | (int(src_half) << 9) | (int(i.op in SIGNED_IOPS) << 10)
    if i.op in FMISC_IDX: return _rec(K_FMISC, FMISC_IDX[i.op], i.dst, i.iterations, flags, *_pack3(i.srcs))
    iop = INT_IDX[i.op]
    fast_i = not src_half and not i.dst_half and not i.sat and not ei and iop < 25 and (is_bw or all(s.absneg == 0 for s in i.srcs))
    return _rec(K_IALU32 if fast_i else K_IMISC, iop, i.dst, i.iterations, flags, *_pack3(i.srcs))
  if isinstance(i, Cat4):
    return _rec(K_SFU, SFU_IDX[i.op], i.dst, i.iterations, int(i.dst_half) | (int(i.sat) << 1) | (int(i.srcs[0].half) << 2), *_pack3(i.srcs))
  if isinstance(i, (Load, Store)):
    ro = getattr(i, "reg_off", None)
    r_src2, r_sh, r_off = ro if ro is not None else (0, 0, 0)
    flags = int(i.type in HALF_TYPES) | (int(ro is not None) << 2) | (i.type << 4) | (r_sh << 8) | (r_off << 12)
    return (K_MEM, MEM_IDX[i.op], i.dst if isinstance(i, Load) else i.val, i.size, flags, 0, 0, 0, i.addr, i.offset & 0xFFFFFFFF, r_src2)
  if isinstance(i, Cat5):
    return _rec(K_TEX, 0, i.dst, i.wrmask, int(i.s2en) | (i.type << 4), 0, 0, 0, i.src1, i.src3 if i.s2en else i.tex)
  assert isinstance(i, Ibo)
  return _rec(K_TEX, 1, i.ssbo, i.ncomp, i.type << 4, 0, 0, 0, i.coord, i.val)

def _max_reg(prog:list[Inst|None]) -> int:
  hi = 0
  for i in prog:
    if i is None: continue
    for s in getattr(i, "srcs", ()):
      if s.kind == "r" and s.val < A0: hi = max(hi, s.val + i.iterations)
    if hasattr(i, "dst") and not isinstance(i, Ibo) and i.dst < A0:
      hi = max(hi, i.dst + (4 if isinstance(i, Cat5) else i.size if isinstance(i, Load) else i.iterations))
    if isinstance(i, Cat1) and i.op == mesa.OPC_SWZ: hi = max(hi, *(r + 1 for r in i.dsts if r < A0), 0)
    if isinstance(i, (Load, Store)):
      if i.addr < A0: hi = max(hi, i.addr + 2)
      if isinstance(i, Store) and i.val < A0: hi = max(hi, i.val + i.size)
      if (ro := getattr(i, "reg_off", None)) is not None and ro[0] < A0: hi = max(hi, ro[0] + 1)
    if isinstance(i, Cat5):
      if i.src1 < A0: hi = max(hi, i.src1 + 2)
      if i.s2en and i.src3 < A0: hi = max(hi, i.src3 + 1)
    if isinstance(i, Ibo):
      if i.coord < A0: hi = max(hi, i.coord + 2)
      if i.val < A0: hi = max(hi, i.val + i.ncomp)
  return min(hi, A0)

@functools.cache
def decode(image:bytes) -> tuple[np.ndarray, bool, bool, int]:
  prog:list[Inst|None] = []
  for pc, word in enumerate(np.frombuffer(image, np.uint64).tolist()):
    try: prog.append(decode_inst(pc, word))
    except NotImplementedError: prog.append(None)
  preds = [i for i in prog if isinstance(i, Cat0) and i.op in (mesa.OPC_PREDT, mesa.OPC_PREDF, mesa.OPC_PREDE)]
  pred_targets:dict[int, int] = {}
  for idx, i in enumerate(preds):
    if i.op == mesa.OPC_PREDE: continue
    nxt, after = preds[idx + 1] if idx + 1 < len(preds) else None, preds[idx + 2] if idx + 2 < len(preds) else None
    if nxt is not None and nxt.op == mesa.OPC_PREDE: pred_targets[i.pc] = nxt.pc
    elif nxt is not None and nxt.op != i.op and after is not None and after.op == mesa.OPC_PREDE: pred_targets[i.pc] = nxt.pc + 1
  recs = [encode_inst(i, pred_targets) if i is not None else _rec(K_UNDEF) for i in prog]
  ops = {i.op for i in prog if isinstance(i, (Load, Store))}
  return np.array(recs, np.uint32).reshape(-1), bool(ops & {mesa.OPC_LDL, mesa.OPC_STL}), bool(ops & {mesa.OPC_LDP, mesa.OPC_STP}), _max_reg(prog)

@dataclass(frozen=True)
class Image: addr:int; width:int; height:int; pitch:int; dtype:np.dtype # noqa: E702

@dataclass(frozen=True)
class Dispatch:
  image:bytes; consts:np.ndarray; local_size:tuple[int, ...]; groups:tuple[int, ...]; localid_reg:int; wgid_reg:int # noqa: E702
  lmem_size:int; pvt_size:int; textures:list[Image]; ibos:list[Image]; demote:bool; entry:int # noqa: E702

def _c(v:int|float|bool, dt:DType=dtypes.uint32) -> UOp: return UOp.const(v, dt)
def _sel(idx:UOp, vals:list[UOp]) -> UOp:
  return functools.reduce(lambda acc, iv: idx.eq(_c(iv[0])).where(iv[1], acc), reversed(list(enumerate(vals[:-1]))), vals[-1])
def _ftz_u(u:UOp, half:bool) -> UOp:
  sign, tiny, dt = (0x8000, 0x0400, dtypes.uint16) if half else (0x80000000, 0x00800000, dtypes.uint32)
  return ((u & _c(sign - 1, dt)) < _c(tiny, dt)).where(u & _c(sign, dt), u)
def _ftz(f:UOp, half:bool) -> UOp:
  return _ftz_u(f.bitcast(dtypes.uint16 if half else dtypes.uint32), half).bitcast(dtypes.half if half else dtypes.float32)
def _cnan_u(u:UOp, half:bool) -> UOp:
  sign, inf, qnan, dt = (0x8000, 0x7C00, 0x7E00, dtypes.uint16) if half else (0x80000000, 0x7F800000, 0x7FC00000, dtypes.uint32)
  return ((u & _c(sign - 1, dt)) > _c(inf, dt)).where(_c(qnan, dt), u)
def _isnan(f:UOp, half:bool) -> UOp:
  sign, inf, dt = (0x8000, 0x7C00, dtypes.uint16) if half else (0x80000000, 0x7F800000, dtypes.uint32)
  return (f.bitcast(dt) & _c(sign - 1, dt)) > _c(inf, dt)
def _cov_to_float(v_f64:UOp, dst_half:bool, even:UOp) -> UOp:
  fdt, udt = (dtypes.half, dtypes.uint16) if dst_half else (dtypes.float32, dtypes.uint32)
  r = v_f64.cast(fdt)
  ru = r.bitcast(udt)
  rtz = ((even.eq(_c(0))) & (r.cast(dtypes.float64).abs() > v_f64.abs())).where(ru - _c(1, udt), ru)
  return _ftz_u(rtz, True).bitcast(fdt) if dst_half else rtz.bitcast(fdt)
def _clz_u(u:UOp, bits:int, trailing:bool=False) -> UOp:
  dt, x, n = dtypes.uint16 if bits == 16 else dtypes.uint32, u, _c(0, dtypes.uint16 if bits == 16 else dtypes.uint32)
  for s in ([8, 4, 2, 1] if bits == 16 else [16, 8, 4, 2, 1]):
    m = ((1 << s) - 1) if trailing else (((1 << s) - 1) << (bits - s))
    z = (x & _c(m, dt)).eq(_c(0, dt))
    n, x = z.where(n + _c(s, dt), n), z.where(x >> _c(s, dt) if trailing else x << _c(s, dt), x)
  return u.eq(_c(0, dt)).where(_c(0xFFFF if bits == 16 else 0xFFFFFFFF, dt), n)
def _fminmax(a:UOp, b:UOp, half:bool, is_max:bool) -> UOp:
  fdt, udt = (dtypes.half, dtypes.uint16) if half else (dtypes.float32, dtypes.uint32)
  z = _c(0.0, fdt)
  zres = ((a.bitcast(udt) & b.bitcast(udt)) if is_max else (a.bitcast(udt) | b.bitcast(udt))).bitcast(fdt)
  return _isnan(a, half).where(b, _isnan(b, half).where(a, (a.eq(z) & b.eq(z)).where(zres, ((a > b) if is_max else (a < b)).where(a, b))))
def _sat_f(fval:UOp, half:bool) -> UOp:
  fdt = dtypes.half if half else dtypes.float32
  z, one = _c(0.0, fdt), _c(1.0, fdt)
  return _isnan(fval, half).where(fval, (fval < z).where(z, (fval > one).where(one, fval)))

@functools.cache
def vm_kernel() -> Callable[..., None]:
  I32, U32, U16, U8, U64, I64 = dtypes.int32, dtypes.uint32, dtypes.uint16, dtypes.uint8, dtypes.uint64, dtypes.int64
  F32, F64, F16 = dtypes.float32, dtypes.float64, dtypes.half
  def _ci(v:int) -> UOp: return UOp.const(v, I32)
  def _not(b:UOp) -> UOp: return b.eq(UOp.const(False, dtypes.bool))
  ax = itertools.count().__next__
  prog, consts = UOp.param(0, U32, 1 << 22, name="prog"), UOp.param(1, U32, 4096, name="consts")
  vmem = UOp.param(2, U8, 1 << 46, name="vmem")
  r_buf, h_buf = UOp.param(3, U32, (NREGS + 3) * MAX_LOCAL, name="r"), UOp.param(4, U16, (NREGS + 3) * MAX_LOCAL, name="h")
  lmem, pvt = UOp.param(5, U8, 1 << 20, name="lmem"), UOp.param(6, U8, 1 << 26, name="pvt")
  tex_m, ibo_m = UOp.param(7, U64, 1024 * 5, name="tex_m"), UOp.param(8, U64, 1024 * 5, name="ibo_m")
  err, p_i = UOp.param(9, I32, 2, name="err"), UOp.param(10, I32, 16, name="p_i")
  vmem16, vmem32 = UOp.param(11, U16, 1 << 45, name="vmem16"), UOp.param(12, U32, 1 << 44, name="vmem32")
  st_pc, st_done = UOp.alloc((MAX_LOCAL,), I32, 0, AddrSpace.REG), UOp.alloc((MAX_LOCAL,), U8, 1, AddrSpace.REG)
  st_blk, st_dep = UOp.alloc((MAX_LOCAL,), U8, 2, AddrSpace.REG), UOp.alloc((MAX_LOCAL,), I32, 3, AddrSpace.REG)
  st_calls = UOp.alloc((16 * MAX_LOCAL,), I32, 4, AddrSpace.REG)
  st_uni, st_cur = UOp.alloc((1,), I32, 5, AddrSpace.REG), UOp.alloc((1,), I32, 6, AddrSpace.REG)
  st_go = UOp.alloc((1,), U8, 7, AddrSpace.REG)
  tmp, tmp_f64 = UOp.alloc((MAX_LOCAL,), U32, 8, AddrSpace.REG), UOp.alloc((MAX_LOCAL,), F64, 9, AddrSpace.REG)

  n_local, n_groups, lx, ly, gx, gy, lid_reg, wgid_reg = [p_i.index(j).load() for j in range(8)]
  lmem_sz, pvt_sz, n_tex, n_ibo, demote, entry, reg_init_i, n_init = [p_i.index(j).load() for j in range(8, 16)]
  reg_init = reg_init_i.cast(U32)
  init_err = UOp.group(err.index(0).store(_ci(0)), err.index(1).store(_ci(0)))

  wg = UOp.range(n_groups, ax())
  r_i, s_i = UOp.range(n_init * n_local, ax()), UOp.range((NREGS - A0) * n_local, ax())
  lm_i, pv_i = UOp.range(lmem_sz, ax()), UOp.range(pvt_sz * n_local, ax())
  init_r = UOp.group(
    UOp.group(r_buf.after(init_err, wg).index(r_i).store(reg_init),
              h_buf.after(init_err, wg).index(r_i).store((reg_init & _c(0xFFFF)).cast(U16))).end(r_i),
    r_buf.after(init_err, wg).index(_ci(A0) * n_local + s_i).store(reg_init).end(s_i))
  l0 = UOp.range(n_local, ax())
  g_xyz = [(wg % gx).cast(U32), ((wg // gx) % gy).cast(U32), (wg // (gx * gy)).cast(U32)]
  l_xyz = [(l0 % lx).cast(U32), ((l0 // lx) % ly).cast(U32), (l0 // (lx * ly)).cast(U32)]
  has_lid, has_wg = lid_reg.ne(_ci(0xfc)), wgid_reg.ne(_ci(0xfc))
  init_ids = UOp.group(
    *[r_buf.after(init_r).index(((lid_reg + _ci(j)) * n_local + l0).valid(has_lid)).store(l_xyz[j]) for j in range(3)],
    *[r_buf.after(init_r).index(((wgid_reg + _ci(j)) * n_local + l0).valid(has_wg)).store(g_xyz[j]) for j in range(3)],
    st_pc.after(init_r).index(l0).store(entry), st_done.after(init_r).index(l0).store(_c(0, U8)),
    st_blk.after(init_r).index(l0).store(_c(0, U8)), st_dep.after(init_r).index(l0).store(_ci(0))).end(l0)
  init_wg = UOp.group(lmem.after(init_err, wg).index(lm_i).store(_c(0, U8)).end(lm_i),
                      pvt.after(init_err, wg).index(pv_i).store(_c(0, U8)).end(pv_i),
                      init_ids, st_uni.after(init_r).index(0).store(entry), st_go.after(init_r).index(0).store(_c(1, U8)))

  loop = UOp.loop(ax()).replace(src=(UOp(Ops.NOOP).after(init_wg),))
  uni = st_uni.after(loop).index(0).load()
  is_uni = uni >= _ci(0)

  g_div, l_min = UOp.range(_not(is_uni).cast(I32), ax()), UOp.range(n_local, ax())
  d_l, b_l = st_done.after(loop, g_div).index(l_min).load(), st_blk.after(loop, g_div).index(l_min).load()
  p_l = st_pc.after(loop, g_div).index(l_min).load()
  any_undone = d_l.eq(_c(0, U8)).cast(I32).reduce(l_min, arg=Ops.MAX)
  min_pc = -(d_l.eq(_c(0, U8)) & b_l.eq(_c(0, U8))).where(-p_l, _ci(-0x7FFFFFFF)).reduce(l_min, arg=Ops.MAX)
  dead = min_pc.eq(_ci(0x7FFFFFFF)) & any_undone.ne(_ci(0))
  pick_div = UOp.group(st_cur.after(loop, g_div).index(0).store(min_pc),
                       st_go.after(loop, g_div).index(0).store((min_pc < _ci(0x7FFFFFFF)).cast(U8)),
                       err.after(loop, g_div).index(0).store(dead.where(_ci(9), err.after(loop, g_div).index(0).load()))).end(g_div)
  g_uni = UOp.range(is_uni.cast(I32), ax())
  picked = UOp.group(pick_div, st_cur.after(loop, g_uni).index(0).store(uni).end(g_uni))

  go_now = st_go.after(picked).index(0).load().ne(_c(0, U8)) & err.after(picked).index(0).load().eq(_ci(0))
  cur = st_cur.after(picked).index(0).load()
  rec = [prog.after(picked).index((cur * _ci(REC_LEN) + _ci(j)).valid(go_now)).load() for j in range(REC_LEN)]
  kind, op, dst, iters, flags, s_meta, s_val = rec[0], rec[1], rec[2].cast(I32), rec[3].cast(I32), rec[4], rec[5:8], rec[8:11]

  def lane_active(lane:UOp, dep:UOp) -> UOp:
    return is_uni.where(_c(True, dtypes.bool), st_pc.after(dep).index(lane).load().eq(cur) & st_done.after(dep).index(lane).load().eq(_c(0, U8)))
  def set_err(dep:UOp, cond:UOp, code:int) -> UOp:
    e0 = err.after(dep).index(0).load()
    do_set = cond & e0.eq(_ci(0))
    return UOp.group(err.after(dep).index(0).store(do_set.where(_ci(code), e0)),
                     err.after(dep).index(1).store(do_set.where(cur, err.after(dep).index(1).load())))
  def write_reg(dep:UOp, lane:UOp, active:UOp, reg:UOp, dst_half:UOp, val_u32:UOp) -> UOp:
    use_h, idx = dst_half.ne(_c(0)) & (reg < _ci(A0)), reg.clamp(0, NREGS - 1) * n_local + lane
    return UOp.group(h_buf.after(dep).index(idx.valid(active & use_h)).store((val_u32 & _c(0xFFFF)).cast(U16)),
                     r_buf.after(dep).index(idx.valid(active & _not(use_h))).store(val_u32))
  def commit_tmp(dep:UOp, reg:UOp, dst_half:UOp) -> UOp:
    use_h, base_r = dst_half.ne(_c(0)) & (reg < _ci(A0)), reg.clamp(0, NREGS - 1) * n_local
    g_ur, l_ur = UOp.range((is_uni & _not(use_h)).cast(I32), ax()), UOp.range(n_local, ax())
    g_uh, l_uh = UOp.range((is_uni & use_h).cast(I32), ax()), UOp.range(n_local, ax())
    g_dv, l_dv = UOp.range(_not(is_uni).cast(I32), ax()), UOp.range(n_local, ax())
    return UOp.group(
      r_buf.after(dep, g_ur).index(base_r + l_ur).store(tmp.after(dep, g_ur).index(l_ur).load()).end(l_ur).end(g_ur),
      h_buf.after(dep, g_uh).index(base_r + l_uh).store((tmp.after(dep, g_uh).index(l_uh).load() & _c(0xFFFF)).cast(U16)).end(l_uh).end(g_uh),
      write_reg(UOp.group(dep, g_dv), l_dv, lane_active(l_dv, g_dv), reg, dst_half, tmp.after(dep, g_dv).index(l_dv).load()).end(l_dv).end(g_dv))

  # Resolve a source ONCE before `for (lane)` into (dep, (base_row, and_mask, xor_mask)); handles r/c/c</i/demote/half>=A0
  def prep_src(dep:UOp, slot:int, meta:UOp, val:UOp, k:UOp, is_float:bool, half:bool, bw_neg:UOp|None=None,
               demote_f16:UOp|bool|None=None) -> tuple[UOp, tuple[UOp, UOp, UOp]]:
    udt = U16 if half else U32
    full_m, sign_m, abs_m = _c(0xFFFF if half else 0xFFFFFFFF, udt), _c(0x8000 if half else 0x80000000, udt), _c(0x7FFF if half else 0x7FFFFFFF, udt)
    sk = meta & _c(7)
    r_inc = ((meta >> _c(6)) & _c(1)).cast(I32) * k
    reg, c_idx = (val.cast(I32) + r_inc).clamp(0, NREGS - 1), (val.cast(I32) + r_inc).clamp(0, 8191)
    c_full = consts.after(dep).index(c_idx.clamp(0, 4095)).load()
    if half:
      cw = consts.after(dep).index((c_idx >> _ci(1)).clamp(0, 4095)).load()
      df16 = is_float if demote_f16 is None else demote_f16
      c_f16 = c_full.bitcast(F32).cast(F16).bitcast(U16).cast(U32) if df16 is True else \
              (c_full & _c(0xFFFF)) if df16 is False else df16.where(c_full.bitcast(F32).cast(F16).bitcast(U16).cast(U32), c_full & _c(0xFFFF))
      vc = demote.ne(_ci(0)).where(c_f16, (cw >> ((c_idx & _ci(1)).cast(U32) * _c(16))) & _c(0xFFFF))
    else: vc = c_full
    need_rel, need_r2h = sk.eq(_c(3)), _c(half, dtypes.bool) & sk.eq(_c(1)) & (reg >= _ci(A0))
    g_sp, l_sp = UOp.range((need_rel | need_r2h).cast(I32), ax()), UOp.range(n_local, ax())
    rel_idx = r_buf.after(dep, g_sp).index(_ci(A0) * n_local + l_sp).load().bitcast(I32) + val.bitcast(I32)
    rel_oob = lane_active(l_sp, UOp.group(dep, g_sp)) & need_rel & ((rel_idx < _ci(0)) | (rel_idx >= _ci(4096)))
    vrel = consts.after(dep, g_sp).index(rel_idx.clamp(0, 4095)).load()
    sp_idx = _ci(NREGS + slot) * n_local + l_sp
    sp_val = need_rel.where(vrel, r_buf.after(dep, g_sp).index(reg * n_local + l_sp).load())
    sp_st = h_buf.after(dep, g_sp).index(sp_idx).store((sp_val & _c(0xFFFF)).cast(U16)) if half else r_buf.after(dep, g_sp).index(sp_idx).store(vrel)
    sp = UOp.group(sp_st, set_err(UOp.group(dep, g_sp), rel_oob, 2)).end(l_sp).end(g_sp)
    is_reg = sk.eq(_c(1)) | need_rel
    row = (need_rel | need_r2h).where(_ci(NREGS + slot), sk.eq(_c(1)).where(reg, _ci(0)))
    abs_b, neg_b = ((meta >> _c(5)) & _c(1)).ne(_c(0)), ((meta >> _c(4)) & _c(1)).ne(_c(0))
    m_mod, x_mod = (abs_b.where(abs_m, full_m), neg_b.where(sign_m, _c(0, udt))) if is_float else \
                   (full_m, (neg_b & bw_neg.ne(_c(0))).where(full_m, _c(0, udt))) if bw_neg is not None else (full_m, _c(0, udt))
    sv = (sk.eq(_c(2)).where(vc, val).cast(udt) & m_mod) ^ x_mod
    return sp, (row * n_local, is_reg.where(m_mod, _c(0, udt)), is_reg.where(x_mod, sv))

  def prep_3(dep:UOp, k:UOp, is_float:bool, half:bool, bw_neg:UOp|None=None) -> tuple[UOp, list[tuple[UOp, UOp, UOp]]]:
    ps = [prep_src(dep, j, s_meta[j], s_val[j], k, is_float, half, bw_neg) for j in range(3)]
    return UOp.group(*[p[0] for p in ps]), [p[1] for p in ps]
  def ld_src(buf:UOp, sp:UOp, lane:UOp, spec:tuple[UOp, UOp, UOp]) -> UOp:
    return (buf.after(sp).index(spec[0] + lane).load() & spec[1]) ^ spec[2]
  def _fconv(fv:UOp, half:bool, dst_half:UOp) -> UOp:
    return dst_half.ne(_c(0)).where(fv.bitcast(U16).cast(U32), fv.cast(F32).bitcast(U32)) if half else \
           dst_half.ne(_c(0)).where(_cov_to_float(fv.cast(F64), True, _c(0)).bitcast(U16).cast(U32), fv.bitcast(U32))
  def _ialu_25(au:UOp, bu:UOp, cu:UOp, bits:int, udt:DType, sdt:DType) -> list[UOp]:
    ai, bi, all1 = au.bitcast(sdt), bu.bitcast(sdt), _c(0xFFFF if bits == 16 else 0xFFFFFFFF, udt)
    sh, sha = bu & _c(bits - 1, udt), au & _c(bits - 1, udt)
    a32, b32, c32 = au.cast(U32), bu.cast(U32), cu.cast(U32)
    ms24 = (((a32 & _c(0xFFFFFF)) ^ _c(0x800000)) - _c(0x800000)) * (((b32 & _c(0xFFFFFF)) ^ _c(0x800000)) - _c(0x800000))
    return [
      (au + bu).cast(U32), (au + bu).cast(U32), (au - bu).cast(U32), (au - bu).cast(U32),
      (au & bu).cast(U32), (au | bu).cast(U32), (au ^ bu).cast(U32), (au ^ all1).cast(U32),
      (au << sh).cast(U32), (au >> sh).cast(U32), (ai >> sh.bitcast(sdt)).bitcast(udt).cast(U32), ((au >> sh) & _c(1, udt)).cast(U32),
      ms24, (a32 & _c(0xFFFFFF)) * (b32 & _c(0xFFFFFF)), (a32 & _c(0xFFFF)) * (b32 & _c(0xFFFF)),
      (((a32 & _c(0xFFFF)) * (b32 >> _c(16))) << _c(16)) + c32, ((a32 & _c(0xFFFF)) * (b32 & _c(0xFFFF))) + c32, ms24 + c32,
      bu.ne(_c(0, udt)).where(au, cu).cast(U32), (bi >= _c(0, sdt)).where(au, cu).cast(U32),
      ((bu >> sha) & cu).cast(U32), ((bu >> sha) | cu).cast(U32), ((bu << sha) | cu).cast(U32),
      ((bu << sha) & cu).cast(U32), ((bu & au) | cu).cast(U32)]

  # 1. K_UNDEF
  g_und = UOp.range((go_now & kind.eq(_c(K_UNDEF))).cast(I32), ax())
  s_und = set_err(g_und, _c(True, dtypes.bool), 1).end(g_und)

  # 2. K_FMAD32 & K_FMAD16 (MAD/ADD/MUL — vectorized SIMD via tmp[])
  def emit_fmad(k_tag:int, half:bool) -> UOp:
    fdt, udt, buf = (F16, U16, h_buf) if half else (F32, U32, r_buf)
    g, k = UOp.range((go_now & kind.eq(_c(k_tag))).cast(I32), ax()), UOp.range(iters, ax())
    sp, ss = prep_3(UOp.group(g, k), k, True, half)
    lane = UOp.range(n_local, ax())
    a, b, c = [_ftz(ld_src(buf, sp, lane, ss[j]).bitcast(fdt), half) for j in range(3)]
    comp = tmp.after(sp).index(lane).store(_cnan_u(_ftz_u((_ftz(a * b, half) + c).bitcast(udt), half), half).cast(U32)).end(lane)
    sat, dst_half = (flags >> _c(1)) & _c(1), flags & _c(1)
    g_p, l_p = UOp.range((sat.ne(_c(0)) | dst_half.ne(_c(int(half)))).cast(I32), ax()), UOp.range(n_local, ax())
    fv = tmp.after(comp, g_p).index(l_p).load().cast(udt).bitcast(fdt)
    fv = sat.ne(_c(0)).where(_sat_f(fv, half), fv)
    post = tmp.after(comp, g_p).index(l_p).store(_fconv(fv, half, dst_half)).end(l_p).end(g_p)
    return commit_tmp(UOp.group(comp, post), dst + k, dst_half).end(k).end(g)
  s_fmad = UOp.group(emit_fmad(K_FMAD32, False), emit_fmad(K_FMAD16, True))

  # 3. K_FMISC (MIN/MAX/SIGN/ABSNEG/FLOOR/TRUNC/SEL/CMP)
  def emit_fmisc(half:bool) -> UOp:
    fdt, udt, buf = (F16, U16, h_buf) if half else (F32, U32, r_buf)
    g = UOp.range((go_now & kind.eq(_c(K_FMISC)) & ((flags >> _c(9)) & _c(1)).eq(_c(int(half)))).cast(I32), ax())
    k = UOp.range(iters, ax())
    sp, ss = prep_3(UOp.group(g, k), k, True, half)
    lane = UOp.range(n_local, ax())
    a, b, c = [_ftz(ld_src(buf, sp, lane, ss[j]).bitcast(fdt), half) for j in range(3)]
    z, one, a64 = _c(0.0, fdt), _c(1.0, fdt), a.cast(F64)
    tr64 = a64.trunc()
    sign_f = _isnan(a, half).where(z, (a > z).where(one, (a < z).where(_c(-1.0, fdt), a)))
    fout = _sel(op, [_fminmax(a, b, half, False), _fminmax(a, b, half, True), sign_f, a,
                     (a64 < tr64).where(tr64 - _c(1.0, F64), tr64).cast(fdt), tr64.cast(fdt), ((b > z) | b.eq(z)).where(a, c)])
    fu = _ftz_u(fout.bitcast(udt), half)
    sat, dst_half = (flags >> _c(1)) & _c(1), flags & _c(1)
    fc = op.eq(_c(6)).where(fu, _cnan_u(fu, half)).bitcast(fdt)
    out_f = _fconv(sat.ne(_c(0)).where(_sat_f(fc, half), fc), half, dst_half)
    cmp_u = _sel((flags >> _c(4)) & _c(7), [a < b, (a < b) | a.eq(b), a > b, (a > b) | a.eq(b), a.eq(b), a.ne(b)]).cast(udt)
    cmp_u = sat.ne(_c(0)).where(cmp_u ^ _c(1, udt), cmp_u)
    cmp_u = ((flags >> _c(2)) & _c(1)).ne(_c(0)).where(_c(0, udt) - cmp_u, cmp_u)
    comp = tmp.after(sp).index(lane).store(op.eq(_c(7)).where(cmp_u.cast(U32), out_f)).end(lane)
    return commit_tmp(comp, dst + k, dst_half).end(k).end(g)
  s_fmisc = UOp.group(emit_fmisc(False), emit_fmisc(True))

  # 4. K_IALU32 (fast 32-bit int/bitwise/shift/mul/mad/sel — vectorized SIMD via tmp[])
  g_ia, k_ia = UOp.range((go_now & kind.eq(_c(K_IALU32))).cast(I32), ax()), UOp.range(iters, ax())
  sp_ia, ss_ia = prep_3(UOp.group(g_ia, k_ia), k_ia, False, False, (flags >> _c(8)) & _c(1))
  l_ia = UOp.range(n_local, ax())
  a_i, b_i, c_i = [ld_src(r_buf, sp_ia, l_ia, ss_ia[j]) for j in range(3)]
  s_ialu32 = commit_tmp(tmp.after(sp_ia).index(l_ia).store(_sel(op, _ialu_25(a_i, b_i, c_i, 32, U32, I32))).end(l_ia),
                        dst + k_ia, _c(0)).end(k_ia).end(g_ia)

  # 5. K_IMISC (16-bit int ops + sat/ei/cmp/clz/ctz/min/max/absneg/sad/signed-mod)
  def emit_imisc(half:bool) -> UOp:
    bits, udt, sdt, buf = (16, U16, dtypes.int16, h_buf) if half else (32, U32, I32, r_buf)
    all1 = _c(0xFFFF if half else 0xFFFFFFFF, udt)
    g = UOp.range((go_now & kind.eq(_c(K_IMISC)) & ((flags >> _c(9)) & _c(1)).eq(_c(int(half)))).cast(I32), ax())
    k = UOp.range(iters, ax())
    sp, ss = prep_3(UOp.group(g, k), k, False, half)
    lane = UOp.range(n_local, ax())
    is_bw, is_s = ((flags >> _c(8)) & _c(1)).ne(_c(0)), ((flags >> _c(10)) & _c(1)).ne(_c(0))
    def imod(j:int) -> UOp:
      u = ld_src(buf, sp, lane, ss[j])
      u = (is_s & ((s_meta[j] >> _c(5)) & _c(1)).ne(_c(0)) & (u.bitcast(sdt) < _c(0, sdt))).where(_c(0, udt) - u, u)
      return ((s_meta[j] >> _c(4)) & _c(1)).ne(_c(0)).where(is_bw.where(u ^ all1, _c(0, udt) - u), u)
    au, bu, cu = imod(0), imod(1), imod(2)
    ai, bi = au.bitcast(sdt), bu.bitcast(sdt)
    sat, ei = ((flags >> _c(1)) & _c(1)).ne(_c(0)), ((flags >> _c(3)) & _c(1)).ne(_c(0))
    umax = _c(0xFFFF if half else 0xFFFFFFFF, I64)
    smin, smax = _c(-32768 if half else -2147483648, I64), _c(32767 if half else 2147483647, I64)
    add_u = ei.where(((au.cast(U64) + bu.cast(U64)) >> _c(1, U64)).cast(udt),
            sat.where((au.cast(I64) + bu.cast(I64)).clamp(0, umax).cast(udt), au + bu))
    sub_u = sat.where((au.cast(I64) - bu.cast(I64)).clamp(0, umax).cast(udt), au - bu)
    add_s = sat.where((ai.cast(I64) + bi.cast(I64)).clamp(smin, smax).cast(sdt).bitcast(udt), au + bu)
    sub_s = sat.where((ai.cast(I64) - bi.cast(I64)).clamp(smin, smax).cast(sdt).bitcast(udt), au - bu)
    cond = (flags >> _c(4)) & _c(7)
    cu_b = _sel(cond, [au < bu, au <= bu, au > bu, au >= bu, au.eq(bu), au.ne(bu)])
    cs_b = _sel(cond, [ai < bi, ai <= bi, ai > bi, ai >= bi, ai.eq(bi), ai.ne(bi)])
    cmp_v = op.eq(_c(35)).where(cs_b, cu_b).cast(udt)
    cmp_v = sat.where(cmp_v ^ _c(1, udt), cmp_v)
    cmp_v = ((flags >> _c(2)) & _c(1)).ne(_c(0)).where(_c(0, udt) - cmp_v, cmp_v)
    out32 = _sel(op, [add_u.cast(U32), add_s.cast(U32), sub_u.cast(U32), sub_s.cast(U32)] + _ialu_25(au, bu, cu, bits, udt, sdt)[4:] + [
      (au < bu).where(au, bu).cast(U32), (ai < bi).where(au, bu).cast(U32),
      (au > bu).where(au, bu).cast(U32), (ai > bi).where(au, bu).cast(U32), au.cast(U32),
      _clz_u(au, bits).cast(U32), _clz_u((ai < _c(0, sdt)).where(au ^ all1, au), bits).cast(U32),
      _clz_u(au, bits, True).cast(U32), (au + bu + cu).cast(U32), cmp_v.cast(U32), cmp_v.cast(U32)])
    dst_half = flags & _c(1)
    if half: out32 = (dst_half.eq(_c(0)) & is_s & op.ne(_c(35))).where(out32.cast(U16).bitcast(dtypes.int16).cast(I32).bitcast(U32), out32)
    return commit_tmp(tmp.after(sp).index(lane).store(out32).end(lane), dst + k, dst_half).end(k).end(g)
  s_imisc = UOp.group(emit_imisc(False), emit_imisc(True))

  # 6. K_SFU (Cat4 — shared f64 prologue/epilogue, each transcendental in its own gate)
  g_sfu, k_sfu = UOp.range((go_now & kind.eq(_c(K_SFU))).cast(I32), ax()), UOp.range(iters, ax())
  dep_sfu, src_hs = UOp.group(g_sfu, k_sfu), ((flags >> _c(2)) & _c(1)).ne(_c(0))
  g_sf0, g_sf1 = UOp.range(_not(src_hs).cast(I32), ax()), UOp.range(src_hs.cast(I32), ax())
  spf, sf_f = prep_src(UOp.group(dep_sfu, g_sf0), 0, s_meta[0], s_val[0], k_sfu, True, False)
  sph, sf_h = prep_src(UOp.group(dep_sfu, g_sf1), 0, s_meta[0], s_val[0], k_sfu, True, True)
  lf0, lf1 = UOp.range(n_local, ax()), UOp.range(n_local, ax())
  ld_sfu = UOp.group(tmp_f64.after(spf).index(lf0).store(_ftz(ld_src(r_buf, spf, lf0, sf_f).bitcast(F32), False).cast(F64)).end(lf0).end(g_sf0),
                     tmp_f64.after(sph).index(lf1).store(_ftz(ld_src(h_buf, sph, lf1, sf_h).bitcast(F16), True).cast(F64)).end(lf1).end(g_sf1))
  sfu_ops = [lambda x: _c(1.0, F64) / x, lambda x: _c(1.0, F64) / x.sqrt(),
             lambda x: x.log2(), lambda x: x.exp2(), lambda x: x.sin(), lambda x: x.sqrt()]
  sfu_eval = []
  for sop, fn in enumerate(sfu_ops):
    gs, ls = UOp.range(op.eq(_c(sop)).cast(I32), ax()), UOp.range(n_local, ax())
    sfu_eval.append(tmp_f64.after(ld_sfu, gs).index(ls).store(fn(tmp_f64.after(ld_sfu, gs).index(ls).load())).end(ls).end(gs))
  ev_sfu = UOp.group(*sfu_eval)
  l_se = UOp.range(n_local, ax())
  y32 = tmp_f64.after(ev_sfu).index(l_se).load().cast(F32)
  sat_s, dst_hs = ((flags >> _c(1)) & _c(1)).ne(_c(0)), flags & _c(1)
  y16 = _cnan_u(_cov_to_float(y32.cast(F64), True, _c(0)).bitcast(U16), True).bitcast(F16)
  y32c = _cnan_u(_ftz_u(y32.bitcast(U32), False), False).bitcast(F32)
  out_sfu = src_hs.where(_fconv(sat_s.where(_sat_f(y16, True), y16), True, dst_hs), _fconv(sat_s.where(_sat_f(y32c, False), y32c), False, dst_hs))
  s_sfu = commit_tmp(tmp.after(ev_sfu).index(l_se).store(out_sfu).end(l_se), dst + k_sfu, dst_hs).end(k_sfu).end(g_sfu)

  # 7. K_MOV (op==0: fast same-type copy; op==1: SWZ; op==2: converting COV)
  def emit_mov_same(half:bool) -> UOp:
    buf = h_buf if half else r_buf
    g = UOp.range((go_now & kind.eq(_c(K_MOV)) & op.eq(_c(0)) & (flags & _c(1)).eq(_c(int(half)))).cast(I32), ax())
    k, l_ms = UOp.range(iters, ax()), UOp.range(n_local, ax())
    sp, spec = prep_src(UOp.group(g, k), 0, s_meta[0], s_val[0], k, False, half, demote_f16=((flags >> _c(4)) & _c(7)).eq(_c(0)))
    return commit_tmp(tmp.after(sp).index(l_ms).store(ld_src(buf, sp, l_ms, spec).cast(U32)).end(l_ms), dst + k, _c(int(half))).end(k).end(g)

  def emit_swz(half:bool) -> UOp:
    buf = h_buf if half else r_buf
    g = UOp.range((go_now & kind.eq(_c(K_MOV)) & op.eq(_c(1)) & ((s_meta[0] >> _c(3)) & _c(1)).eq(_c(int(half)))).cast(I32), ax())
    sp0, s0 = prep_src(g, 0, s_meta[0], s_val[0], _ci(0), False, half)
    sp1, s1 = prep_src(g, 1, s_meta[1], s_val[1], _ci(0), False, half)
    sp, l_s = UOp.group(sp0, sp1), UOp.range(n_local, ax())
    v0, v1 = ld_src(buf, sp, l_s, s0).cast(U32), ld_src(buf, sp, l_s, s1).cast(U32)
    rd, act = UOp.group(v0, v1), lane_active(l_s, sp)
    return UOp.group(write_reg(rd, l_s, act, dst, flags & _c(1), v0),
                     write_reg(rd, l_s, act, s_val[2].cast(I32), flags & _c(1), v1)).end(l_s).end(g)

  g_cov, k_m = UOp.range((go_now & kind.eq(_c(K_MOV)) & op.eq(_c(2))).cast(I32), ax()), UOp.range(iters, ax())
  dep_m = UOp.group(g_cov, k_m)
  src_t, dst_t, even_m = (flags >> _c(4)) & _c(7), (flags >> _c(8)) & _c(7), (flags >> _c(2)) & _c(1)
  src_hm = src_t.eq(_c(0)) | src_t.eq(_c(2)) | src_t.eq(_c(4)) | src_t.eq(_c(6)) | src_t.eq(_c(7))
  g_m0, g_m1 = UOp.range(_not(src_hm).cast(I32), ax()), UOp.range(src_hm.cast(I32), ax())
  spm_f, sm_f = prep_src(UOp.group(dep_m, g_m0), 0, s_meta[0], s_val[0], k_m, False, False)
  spm_h, sm_h = prep_src(UOp.group(dep_m, g_m1), 0, s_meta[0], s_val[0], k_m, False, True, demote_f16=src_t.eq(_c(0)))
  lm0, lm1 = UOp.range(n_local, ax()), UOp.range(n_local, ax())
  ld_cov = UOp.group(tmp.after(spm_f).index(lm0).store(ld_src(r_buf, spm_f, lm0, sm_f)).end(lm0).end(g_m0),
                     tmp.after(spm_h).index(lm1).store(ld_src(h_buf, spm_h, lm1, sm_h).cast(U32)).end(lm1).end(g_m1))
  l_m = UOp.range(n_local, ax())
  raw_m = tmp.after(ld_cov).index(l_m).load()
  # cov from u1 (6) and i1 (7) both sign-extend from int8
  s_f64 = _sel(src_t, [raw_m.cast(U16).bitcast(F16).cast(F64), raw_m.bitcast(F32).cast(F64), (raw_m & _c(0xFFFF)).cast(F64), raw_m.cast(F64),
                       raw_m.cast(U16).bitcast(dtypes.int16).cast(F64), raw_m.bitcast(I32).cast(F64), raw_m.cast(U8).bitcast(dtypes.int8).cast(F64)])
  s_i64 = _sel(src_t, [_c(0, I64), _c(0, I64), (raw_m & _c(0xFFFF)).cast(I64), raw_m.cast(I64),
                       raw_m.cast(U16).bitcast(dtypes.int16).cast(I64), raw_m.bitcast(I32).cast(I64), raw_m.cast(U8).bitcast(dtypes.int8).cast(I64)])
  f_clean = s_f64.ne(s_f64).where(_c(0.0, F64), s_f64)
  dmin = _sel(dst_t, [_c(v, F64) for v in (0.0, 0.0, 0.0, 0.0, -32768.0, -2147483648.0, 0.0, -128.0)])
  dmax = _sel(dst_t, [_c(v, F64) for v in (0.0, 0.0, 65535.0, 4294967295.0, 32767.0, 2147483647.0, 255.0, 127.0)])
  f_cl = (f_clean < dmin).where(dmin, (f_clean > dmax).where(dmax, f_clean.trunc()))
  f2i = dst_t.eq(_c(3)).where(f_cl.cast(U32), f_cl.cast(I64).cast(U32))
  to_f = dst_t.eq(_c(0)).where(_cov_to_float(s_f64, True, even_m).bitcast(U16).cast(U32), _cov_to_float(s_f64, False, even_m).bitcast(U32))
  out_m = src_t.eq(dst_t).where(raw_m, (dst_t < _c(2)).where(to_f, (src_t < _c(2)).where(f2i, s_i64.cast(U32))))
  out_m = dst_t.eq(_c(6)).where(out_m & _c(0xFF), dst_t.eq(_c(7)).where(out_m.cast(U8).bitcast(dtypes.int8).cast(U16).cast(U32), out_m))
  s_cov = commit_tmp(tmp.after(ld_cov).index(l_m).store(out_m).end(l_m), dst + k_m, flags & _c(1)).end(k_m).end(g_cov)
  s_mov = UOp.group(emit_mov_same(False), emit_mov_same(True), emit_swz(False), emit_swz(True), s_cov)

  # 8. K_MEM (Cat6 LDG/STG/LDL/STL/LDP/STP)
  g_mem, l_mem = UOp.range((go_now & kind.eq(_c(K_MEM))).cast(I32), ax()), UOp.range(n_local, ax())
  act_mem, m_type, m_half = lane_active(l_mem, g_mem), (flags >> _c(4)) & _c(7), flags & _c(1)
  esz = (m_type >= _c(6)).where(_ci(1), (m_type.eq(_c(0)) | m_type.eq(_c(2)) | m_type.eq(_c(4))).where(_ci(2), _ci(4)))
  areg, moff = s_val[0].cast(I32).clamp(0, NREGS - 1), s_val[1].bitcast(I32)
  g_hi = r_buf.after(g_mem).index((areg + _ci(1)).clamp(0, NREGS - 1) * n_local + l_mem).load().cast(U64) << _c(32, U64)
  gaddr = (r_buf.after(g_mem).index(areg * n_local + l_mem).load().cast(U64) | g_hi) + moff.cast(I64).bitcast(U64)
  r_src2, r_sh, r_off = s_val[2].cast(I32).clamp(0, NREGS - 1), ((flags >> _c(8)) & _c(3)).cast(U64), ((flags >> _c(12)) & _c(3)).cast(U64)
  roff = ((r_buf.after(g_mem).index(r_src2 * n_local + l_mem).load().cast(U64) << r_sh) + r_off) * esz.cast(U64)
  gaddr = ((flags >> _c(2)) & _c(1)).ne(_c(0)).where(gaddr + roff, gaddr)
  lp_off = r_buf.after(g_mem).index(areg * n_local + l_mem).load().bitcast(I32) + moff
  is_g, is_l, is_p, is_ld = op < _c(2), (op.eq(_c(2)) | op.eq(_c(3))), op >= _c(4), (op & _c(1)).eq(_c(0))
  l_ok, p_ok = (lp_off >= _ci(0)) & (lp_off + esz * iters <= lmem_sz), (lp_off >= _ci(0)) & (lp_off + esz * iters <= pvt_sz)
  mem_ok = act_mem & is_g.where(_c(True, dtypes.bool), is_l.where(l_ok, p_ok))
  c_m = UOp.range(iters, ax())
  c_off = c_m * esz
  def ld_b(b:int) -> UOp:
    ok = mem_ok & is_ld & (_ci(b) < esz)
    gv = vmem.after(g_mem).index((gaddr + (c_off + _ci(b)).cast(U64)).cast(I64).valid(ok & is_g)).load()
    lv = lmem.after(g_mem).index((lp_off + c_off + _ci(b)).valid(ok & is_l)).load()
    pv = pvt.after(g_mem).index((l_mem * pvt_sz + lp_off + c_off + _ci(b)).valid(ok & is_p)).load()
    return ok.where(is_g.where(gv, is_l.where(lv, pv)).cast(U32) << _c(b * 8), _c(0))
  ld_val = ld_b(0) | ld_b(1) | ld_b(2) | ld_b(3)
  ld_val = m_type.eq(_c(7)).where(ld_val.cast(U8).bitcast(dtypes.int8).cast(U16).cast(U32), ld_val)
  st_idx = (dst + c_m).clamp(0, NREGS - 1) * n_local + l_mem
  st_val = m_half.ne(_c(0)).where(h_buf.after(g_mem).index(st_idx).load().cast(U32), r_buf.after(g_mem).index(st_idx).load())
  def st_b(b:int) -> UOp:
    ok, bv = mem_ok & _not(is_ld) & (_ci(b) < esz), ((st_val >> _c(b * 8)) & _c(0xFF)).cast(U8)
    return UOp.group(vmem.after(g_mem).index((gaddr + (c_off + _ci(b)).cast(U64)).cast(I64).valid(ok & is_g)).store(bv),
                     lmem.after(g_mem).index((lp_off + c_off + _ci(b)).valid(ok & is_l)).store(bv),
                     pvt.after(g_mem).index((l_mem * pvt_sz + lp_off + c_off + _ci(b)).valid(ok & is_p)).store(bv))
  s_mem = UOp.group(write_reg(UOp.group(g_mem, gaddr, lp_off, ld_val), l_mem, act_mem & is_ld, dst + c_m, m_half, ld_val),
                    *[st_b(b) for b in range(4)], set_err(g_mem, act_mem & is_l & _not(l_ok), 3),
                    set_err(g_mem, act_mem & op.eq(_c(5)) & _not(p_ok), 4)).end(c_m).end(l_mem).end(g_mem)

  # 9. K_TEX (Cat5 ISAM + Ibo STIB using direct uint16/uint32 loads/stores)
  g_is = UOp.range((go_now & kind.eq(_c(K_TEX)) & op.eq(_c(0))).cast(I32), ax())
  tex_hi = (n_tex - _ci(1)).clamp(0, 1023)
  s2en, tb_s = (flags & _c(1)).ne(_c(0)), s_val[1].cast(I32).clamp(0, tex_hi) * _ci(5)
  ts_meta = [tex_m.after(g_is).index(tb_s + _ci(j)).load() for j in range(5)]
  l_is = UOp.range(n_local, ax())
  act_is = lane_active(l_is, g_is)
  t_s2 = h_buf.after(g_is).index((s_val[1].cast(I32).clamp(0, NREGS - 1) * n_local + l_is).valid(s2en)).load().cast(I32)
  t_idx = s2en.where(t_s2, s_val[1].cast(I32))
  t_oob = act_is & ((t_idx < _ci(0)) | (t_idx >= n_tex))
  tb_d = t_idx.clamp(0, tex_hi) * _ci(5)
  t_addr, t_w, t_h, t_pitch, t_f16 = [s2en.where(tex_m.after(g_is).index((tb_d + _ci(j)).valid(s2en)).load(), ts_meta[j]) for j in range(5)]
  tx = r_buf.after(g_is).index(s_val[0].cast(I32).clamp(0, NREGS - 1) * n_local + l_is).load().cast(U64)
  ty = r_buf.after(g_is).index((s_val[0].cast(I32) + _ci(1)).clamp(0, NREGS - 1) * n_local + l_is).load().cast(U64)
  t_ok, t_eb = act_is & _not(t_oob) & (tx < t_w) & (ty < t_h), ty * t_pitch + tx * _c(4, U64)
  is_dst_f16, is_i16 = ((flags >> _c(4)) & _c(7)).eq(_c(0)), t_f16.ne(_c(0, U64))
  t_vals = []
  for ch in range(4):
    has_ch = ((iters.cast(U32) >> _c(ch)) & _c(1)).ne(_c(0))
    ok_ch, eb = t_ok & has_ch, t_eb + _c(ch, U64)
    u16 = vmem16.after(g_is).index(((t_addr >> _c(1, U64)) + eb).cast(I64).valid(ok_ch & is_i16)).load()
    u32 = vmem32.after(g_is).index(((t_addr >> _c(2, U64)) + eb).cast(I64).valid(ok_ch & _not(is_i16))).load()
    fv = ok_ch.where(is_i16.where(u16.bitcast(F16).cast(F32), u32.bitcast(F32)), _c(0.0, F32))
    t_vals.append((has_ch, is_dst_f16.where(fv.cast(F16).bitcast(U16).cast(U32), fv.bitcast(U32))))
  rd_is = UOp.group(*[v for _, v in t_vals])
  s_is = UOp.group(set_err(g_is, t_oob, 5), *[
    write_reg(rd_is, l_is, act_is & hc & _not(t_oob),
              dst + sum(((iters >> _ci(p)) & _ci(1) for p in range(ch)), _ci(0)), is_dst_f16.cast(U32), oc)
    for ch, (hc, oc) in enumerate(t_vals)]).end(l_is).end(g_is)

  g_ib = UOp.range((go_now & kind.eq(_c(K_TEX)) & op.eq(_c(1))).cast(I32), ax())
  ibb = dst.clamp(0, (n_ibo - _ci(1)).clamp(0, 1023)) * _ci(5)
  i_addr, i_w, i_h, i_pitch, i_f16 = [ibo_m.after(g_ib).index(ibb + _ci(j)).load() for j in range(5)]
  l_ib, c_ib = UOp.range(n_local, ax()), UOp.range(iters, ax())
  act_ib = lane_active(l_ib, g_ib)
  ib_oob = act_ib & ((dst < _ci(0)) | (dst >= n_ibo))
  ix = r_buf.after(g_ib).index(s_val[0].cast(I32).clamp(0, NREGS - 1) * n_local + l_ib).load().cast(U64)
  iy = r_buf.after(g_ib).index((s_val[0].cast(I32) + _ci(1)).clamp(0, NREGS - 1) * n_local + l_ib).load().cast(U64)
  ib_ok, is_img_f16 = act_ib & _not(ib_oob) & (ix < i_w) & (iy < i_h), i_f16.ne(_c(0, U64))
  vreg = (s_val[1].cast(I32) + c_ib).clamp(0, NREGS - 1) * n_local + l_ib
  vf_h, vf_f = h_buf.after(g_ib).index(vreg).load().bitcast(F16).cast(F32), r_buf.after(g_ib).index(vreg).load().bitcast(F32)
  vf = ((flags >> _c(4)) & _c(7)).eq(_c(0)).where(vf_h, vf_f)
  ieb = iy * i_pitch + ix * _c(4, U64) + c_ib.cast(U64)
  s_ib = UOp.group(
    vmem16.after(g_ib).index(((i_addr >> _c(1, U64)) + ieb).cast(I64).valid(ib_ok & is_img_f16)).store(vf.cast(F16).bitcast(U16)),
    vmem32.after(g_ib).index(((i_addr >> _c(2, U64)) + ieb).cast(I64).valid(ib_ok & _not(is_img_f16))).store(vf.bitcast(U32)),
    set_err(g_ib, ib_oob, 6)).end(c_ib).end(l_ib).end(g_ib)

  exec_done = UOp.group(s_und, s_fmad, s_fmisc, s_ialu32, s_imisc, s_sfu, s_mov, s_mem, s_is, s_ib)

  # Advance PC for non-CTRL instructions + K_CTRL (NOP/END/JUMP/BR/BRAO/BRAA/PREDT/PREDF/CALL/RET/BAR)
  is_non_ctrl, is_ctrl = go_now & kind.ne(_c(K_CTRL)) & kind.ne(_c(K_UNDEF)), go_now & kind.eq(_c(K_CTRL))
  g_adv_u = UOp.range((is_non_ctrl & is_uni).cast(I32), ax())
  g_adv_d, l_ad = UOp.range((is_non_ctrl & _not(is_uni)).cast(I32), ax()), UOp.range(n_local, ax())
  g_unop = UOp.range((is_ctrl & is_uni & op.eq(_c(0))).cast(I32), ax())
  g_cf = UOp.range((is_ctrl & _not(is_uni & op.eq(_c(0)))).cast(I32), ax())
  l_bar, l_cf = UOp.range(n_local, ax()), UOp.range(n_local, ax())
  bar_other = (_not(is_uni) & st_done.after(exec_done, g_cf).index(l_bar).load().eq(_c(0, U8)) &
               st_pc.after(exec_done, g_cf).index(l_bar).load().ne(cur)).cast(I32).reduce(l_bar, arg=Ops.MAX)
  dep_cf = UOp.group(exec_done, g_cf, bar_other)
  act_cf, dep_l = lane_active(l_cf, dep_cf), st_dep.after(dep_cf).index(l_cf).load()
  c1 = r_buf.after(dep_cf).index((_ci(P0) + (iters & _ci(3))) * n_local + l_cf).load().ne(_c(0)) ^ (flags & _c(1)).ne(_c(0))
  p2_reg = _ci(P0) + ((flags >> _c(4)) & _c(3)).cast(I32)
  c2 = r_buf.after(dep_cf).index(p2_reg * n_local + l_cf).load().ne(_c(0)) ^ ((flags >> _c(1)) & _c(1)).ne(_c(0))
  p0_nz = r_buf.after(dep_cf).index(_ci(P0) * n_local + l_cf).load().ne(_c(0))
  br_go = _sel(op, [_c(False, dtypes.bool), _c(False, dtypes.bool), _c(True, dtypes.bool), c1, c1 | c2, c1 & c2, _not(p0_nz), p0_nz])
  nxt_pc = cur + _ci(1)
  ret_pc = st_calls.after(dep_cf).index(((dep_l - _ci(1)).clamp(0, 15)) * n_local + l_cf).load()
  new_pc = ((op >= _c(2)) & (op <= _c(7))).where(br_go.where(dst, nxt_pc), op.eq(_c(8)).where(dst, op.eq(_c(9)).where(ret_pc,
           op.eq(_c(10)).where(bar_other.ne(_ci(0)).where(cur, nxt_pc), nxt_pc))))
  call_ovf, ret_udf = act_cf & op.eq(_c(8)) & (dep_l >= _ci(16)), act_cf & op.eq(_c(9)) & (dep_l <= _ci(0))
  s_cf_lanes = UOp.group(
    st_pc.after(dep_cf).index(l_cf.valid(act_cf)).store(new_pc),
    st_done.after(dep_cf).index(l_cf.valid(act_cf)).store(op.eq(_c(1)).where(_c(1, U8), st_done.after(dep_cf).index(l_cf).load())),
    st_blk.after(dep_cf).index(l_cf.valid(act_cf)).store(op.eq(_c(10)).where(bar_other.ne(_ci(0)).cast(U8), st_blk.after(dep_cf).index(l_cf).load())),
    st_dep.after(dep_cf).index(l_cf.valid(act_cf)).store(op.eq(_c(8)).where(dep_l + _ci(1), op.eq(_c(9)).where(dep_l - _ci(1), dep_l))),
    st_calls.after(dep_cf).index((dep_l.clamp(0, 15) * n_local + l_cf).valid(act_cf & op.eq(_c(8)) & _not(call_ovf))).store(nxt_pc),
    set_err(dep_cf, call_ovf, 7), set_err(dep_cf, ret_udf, 8)).end(l_cf)
  l_rc, pc0 = UOp.range(n_local, ax()), st_pc.after(s_cf_lanes).index(0).load()
  any_diff = (st_done.after(s_cf_lanes).index(l_rc).load().ne(_c(0, U8)) | st_blk.after(s_cf_lanes).index(l_rc).load().ne(_c(0, U8)) |
              st_pc.after(s_cf_lanes).index(l_rc).load().ne(pc0)).cast(I32).reduce(l_rc, arg=Ops.MAX)
  any_left = st_done.after(s_cf_lanes).index(l_rc).load().eq(_c(0, U8)).cast(I32).reduce(l_rc, arg=Ops.MAX)
  step = UOp.group(
    st_uni.after(exec_done, g_adv_u).index(0).store(nxt_pc).end(g_adv_u),
    st_pc.after(exec_done, g_adv_d).index(l_ad.valid(lane_active(l_ad, g_adv_d))).store(nxt_pc).end(l_ad).end(g_adv_d),
    st_uni.after(exec_done, g_unop).index(0).store(nxt_pc).end(g_unop),
    UOp.group(st_uni.after(s_cf_lanes).index(0).store(any_diff.eq(_ci(0)).where(pc0, _ci(-1))),
              st_go.after(s_cf_lanes).index(0).store(any_left.cast(U8))).end(g_cf))
  wg_done = step.backedge(loop, st_go.after(step).index(0).load().ne(_c(0, U8)) & err.after(step).index(0).load().eq(_ci(0))).end(wg)

  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", TRANSCENDENTAL=1, CAPTURE_PROCESS_REPLAY=0):
    prg = to_program(UOp.sink(wg_done, arg=KernelInfo(name="qcom_a630_vm")), Device["CPU"].renderer)
  rt, slots = get_runtime("CPU", prg), [p.arg.slot for p in prg.kernel_params]
  return lambda *bufs: rt(*[bufs[s] for s in slots])

def _pack_images(imgs:list[Image]) -> np.ndarray:
  if not imgs: return np.zeros(5, np.uint64)
  return np.array([[i.addr, i.width, i.height, i.pitch // i.dtype.itemsize, int(i.dtype == np.float16)] for i in imgs], np.uint64).reshape(-1)

def run(d:Dispatch):
  n_local, n_groups = math.prod(d.local_size), math.prod(d.groups)
  assert n_local <= MAX_LOCAL, f"local size {n_local} exceeds {MAX_LOCAL}"
  prog, uses_lmem, uses_pvt, max_r = decode(d.image)
  n_init = min(A0, max(max_r, d.localid_reg + 3 if d.localid_reg != 0xfc else 0, d.wgid_reg + 3 if d.wgid_reg != 0xfc else 0))
  r_buf, h_buf = np.empty((NREGS + 3) * n_local, np.uint32), np.empty((NREGS + 3) * n_local, np.uint16)
  lmem_sz, pvt_sz = d.lmem_size if uses_lmem else 0, d.pvt_size if uses_pvt else 0
  lmem, pvt = np.empty(max(lmem_sz, 1), np.uint8), np.empty(max(pvt_sz * n_local, 1), np.uint8)
  tex_m, ibo_m, err = _pack_images(d.textures), _pack_images(d.ibos), np.zeros(2, np.int32)
  params = np.array([n_local, n_groups, d.local_size[0], d.local_size[1], d.groups[0], d.groups[1],
                     d.localid_reg, d.wgid_reg, lmem_sz, pvt_sz, len(d.textures), len(d.ibos),
                     int(d.demote), d.entry, np.int32(np.uint32(REG_INIT)), n_init], np.int32)
  vm_kernel()(prog.ctypes.data, d.consts.ctypes.data, 0, r_buf.ctypes.data, h_buf.ctypes.data,
              lmem.ctypes.data, pvt.ctypes.data, tex_m.ctypes.data, ibo_m.ctypes.data, err.ctypes.data, params.ctypes.data, 0, 0)
  code, epc = int(err[0]), int(err[1])
  if code == 1: raise decode_inst(epc, int(np.frombuffer(d.image, np.uint64)[epc])).error()
  if code: raise RuntimeError("every thread is waiting at a bar" if code == 9 else f"pc {epc}: {ERR_MSGS[code]}")
