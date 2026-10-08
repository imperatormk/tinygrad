"""Native straight-line FP32 ALU blocks; instruction eligibility is fixed during decoding."""
import ctypes, functools, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard
from test.mockgpu.qcom import emu

HEADER = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <float.h>
#if FLT_EVAL_METHOD != 0
#error Float evaluation must round to the operand type
#endif
static inline uint32_t bits(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static inline float value(uint32_t u) { float x; memcpy(&x,&u,4); return x; }
static inline uint32_t flush(uint32_t u) { return (u&0x7fffffffu)<0x00800000u ? u&0x80000000u : u; }
static inline uint32_t finish(float x) { uint32_t u=flush(bits(x)); return (u&0x7fffffffu)>0x7f800000u ? 0x7fc00000u : u; }
static inline uint32_t mad(float a,float b,float c) { float p=a*b; return finish(value(flush(bits(p)))+c); }

typedef uint32_t U __attribute__((vector_size(16)));
typedef float F __attribute__((vector_size(16)));
static inline U vbroadcast(uint32_t x) { return (U){x,x,x,x}; }
static inline U vbits(F x) { U u; memcpy(&u,&x,16); return u; }
static inline F vvalue(U u) { F x; memcpy(&x,&u,16); return x; }
static inline U vflush(U u) {
 U m=(U)((u&vbroadcast(0x7fffffffu))<vbroadcast(0x00800000u));
 return (u&~m)|((u&vbroadcast(0x80000000u))&m);
}
static inline U vfinish(F x) {
 U u=vflush(vbits(x)), m=(U)((u&vbroadcast(0x7fffffffu))>vbroadcast(0x7f800000u));
 return (u&~m)|(vbroadcast(0x7fc00000u)&m);
}
static inline U vmad(F a,F b,F c) { F p=a*b; return vfinish(vvalue(vflush(vbits(p)))+c); }
"""

@dataclass(frozen=True)
class Block:
  end:int
  ops:tuple[tuple[emu.Cat2|emu.Cat3, int], ...]
  dsts:tuple[int, ...]

  def execute(self, t:emu.Threads, fn):
    fn(t.r.ctypes.data, len(t.mask), None if t.mask is t.everyone else t.mask.ctypes.data)
    for reg in self.dsts:
      t.float_reads[reg] = None
      t.float_clean[reg] = t.mask is t.everyone or t.float_clean[reg]
      if t.d.merged and reg < emu.NREGS // 2:
        t.float_reads[emu.NREGS + 2*reg] = t.float_reads[emu.NREGS + 2*reg + 1] = None
        t.float_clean[emu.NREGS + 2*reg] = t.float_clean[emu.NREGS + 2*reg + 1] = False

def supported(i) -> TypeGuard[emu.Cat2|emu.Cat3]:
  if not isinstance(i, (emu.Cat2, emu.Cat3)) or i.op not in (emu.mesa.OPC_ADD_F, emu.mesa.OPC_MUL_F, emu.mesa.OPC_MAD_F32): return False
  return not (i.sat or i.dst_conv or i.dst_half) and i.dst + i.iterations <= emu.A0 and all(
    s.kind == 'r' and not s.half and s.val < emu.A0 for srcs in i.repeat_srcs for s in srcs)

@functools.cache
def plan(image:bytes, entry:int) -> dict[int, Block]:
  prog = emu.decode(image)
  boundaries = {entry} | {i.target for i in prog if isinstance(i, emu.Cat0)}
  blocks:dict[int, Block] = {}
  pc = 0
  while pc < len(prog):
    if not supported(prog[pc]):
      pc += 1
      continue
    start = pc
    ops:list[tuple[emu.Cat2|emu.Cat3, int]] = []
    while pc < len(prog) and supported(i := prog[pc]):
      if pc != start and (pc in boundaries or len(ops) + i.iterations > 64): break
      ops.extend((i, k) for k in range(i.iterations))
      pc += 1
    blocks[start] = Block(pc, tuple(ops), tuple(sorted({i.dst + k for i,k in ops})))
  return blocks

def body(block:Block, vector:bool) -> str:
  prefix = 'v' if vector else ''
  regs = sorted({s.val for i,k in block.ops for s in i.repeat_srcs[k]} | set(block.dsts))
  lines = [f'U r{r}; memcpy(&r{r},r+{r}*n+lane,16);' if vector else f'uint32_t r{r}=r[{r}*n+lane];' for r in regs]
  for i,k in block.ops:
    args = []
    for s in i.repeat_srcs[k]:
      value = f'r{s.val}'
      if s.absneg & 2: value = f'({value}&0x7fffffffu)'
      if s.absneg & 1: value = f'({value}^0x80000000u)'
      args.append(f'{prefix}value({prefix}flush({value}))')
    value = f'{prefix}mad({",".join(args)})' if i.op == emu.mesa.OPC_MAD_F32 else (
      f'{prefix}finish({args[0]} {"+" if i.op == emu.mesa.OPC_ADD_F else "*"} {args[1]})')
    lines.append(f'r{i.dst+k}={value};')
  lines += [f'memcpy(r+{r}*n+lane,&r{r},16);' if vector else f'r[{r}*n+lane]=r{r};' for r in block.dsts]
  return '\n'.join(lines)

def source(blocks:dict[int, Block]) -> str:
  return HEADER + '\n'.join(f'''void block_{pc}(uint32_t *r,size_t n,const uint8_t *mask) {{
size_t lane=0;
if (!mask) {{ for (;lane+4<=n;lane+=4) {{
{body(block, True)}
}} }}
for (;lane<n;lane++) {{
if (mask && !mask[lane]) continue;
{body(block, False)}
}}
}}''' for pc,block in blocks.items())

@functools.cache
def compile_source(code:str):
  # Keep source and compiler diagnostics available when an experimental native build fails.
  directory = Path(tempfile.mkdtemp(prefix='qcom-emu-native-'))
  (src := directory/'block.c').write_text(code)
  so = directory/'block.so'
  subprocess.run(['cc', '-O3', '-std=c11', '-shared', '-fPIC', '-fno-fast-math', '-ffp-contract=off',
                  '-fexcess-precision=standard', str(src), '-o', str(so)], check=True)
  return ctypes.CDLL(str(so))

@functools.cache
def kernels(image:bytes, entry:int):
  blocks = plan(image, entry)
  lib = compile_source(source(blocks))
  result = {}
  for pc in blocks:
    fn = getattr(lib, f'block_{pc}')
    fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p], None
    result[pc] = fn
  return result
