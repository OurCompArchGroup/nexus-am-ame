---
name: xuantie-to-zames
description: Migrate RISC-V matrix-extension assembly from the Xuantie 0.6 dialect (e.g. mlae8, mmacc.w.b, msyncreset, llvm-mc -mattr=+zame,+matrix-xuantie-0.6) to the Zames proposal-14 form (mla/mlb/mlc, single mmacc, msetcfg/mcfg, msyncregreset sync0..sync15, -march=...zames_zmasync via clang). Use when porting matrix tests under nexus-am/apps or nexus-am/tests, or when an old-dialect file fails to assemble with the current toolchain.
---

# Xuantie 0.6 → Zames proposal-14 migration

The Zames spec drops type encoding from instruction mnemonics. Element width and
signedness are programmed once into `mcfgN` configuration registers and shared
across loads, stores, and `mmacc`. The accompanying Zmasync extension renames
sync primitives and swaps `macquire` operand order.

## Toolchain

| Aspect | Xuantie 0.6 | Zames |
|---|---|---|
| llvm-mc attrs | `+zame,+matrix-xuantie-0.6` | `+zames,+zmasync` |
| Driving the assembler | custom `llvm-mc` rule in Makefile | `clang` (set `MARCH = rv64gc..._zames_zmasync`) |
| MARCH suffix | n/a (used llvm-mc directly) | append `_zames_zmasync` |

Drop the `llvm-mc-rule` block in the per-app Makefile and let `Makefile.app`
drive `clang -c matop.S`. This also fixes the `cannot link object files with
different floating-point ABI` linker error caused by llvm-mc emitting an
ABI-less object next to `lp64d` C objects.

## Mnemonic rename

| Xuantie 0.6 | Zames | Notes |
|---|---|---|
| `msettilem/n/k`, `msettile{m,n,k}i` | unchanged | |
| `mlae8` / `mlbe8` / `mlce32` | `mla` / `mlb` / `mlc` | width comes from `mcfg(md)` |
| `msce32` (and friends) | `msc` (or `msa`/`msb`) | width comes from `mcfg(ms3)` |
| transposed loads (if any) | `mlat`/`mlbt`/`mlct`, `msat`/`msbt`/`msct` | |
| whole-register | `mla.whole` etc. | save/restore raw register |
| `mmacc.w.b`, `mmaccu.w.b`, `mmaccsu.w.b`, `mmaccus.w.b` | `mmacc md, ms2, ms1` | sign/width from `mcfg(ms1)` & `mcfg(ms2)`; **note operand order: ms2 before ms1** |
| `mzero` family | `mzero md` | no type suffix |
| n/a | `minit` | sets `mstatus.MS = 01` (initial) |

`mmacc md, ms2, ms1` semantics: `md = md + ms1 * ms2`. When porting
`mmacc.w.b acc, trA, trB` (acc += trA * trB), write `mmacc acc, trB, trA`.

## mcfg programming

Every tile and accumulator register used by a load/store/`mmacc` must have its
`mcfgN` programmed first. `msetcfg mcfgN, rs1` writes `x[rs1]` into the
selected mcfg. The 3-bit `mcfg_d1` selector reuses the matrix-register
numbering: `mcfg0..mcfg3` configure `tr0..tr3`, `mcfg4..mcfg7` configure
`acc0..acc3`. `mgetcfg rd, mcfgN` reads it back.

`type_code[3:0]` (table_sel = 0):

| code | type | code | type |
|---|---|---|---|
| 0000 | int4   | 1000 | fp8e4m3 |
| 0001 | uint4  | 1001 | fp16    |
| 0010 | int8   | 1010 | bf16    |
| 0011 | uint8  | 1011 | tf32    |
| 0100 | int32  | 1100 | fp32    |
| 0101 | nvfp4  | 1101 | fp2pack4 |
| 0110 | mxfp4  | 1110 | fp2pack5 |
| 0111 | fp8e5m2 | 1111 | Reserved |

Boilerplate (assembly):

```asm
.equ MCFG_INT8,  0x02
.equ MCFG_UINT8, 0x03
.equ MCFG_INT32, 0x04

.macro SETCFG mcfg_sel, type_code
    li      t6, \type_code
    msetcfg \mcfg_sel, t6
.endm

    SETCFG  mcfg0, MCFG_INT8        # tr0 = int8
    SETCFG  mcfg1, MCFG_INT8        # tr1 = int8
    SETCFG  mcfg4, MCFG_INT32       # acc0 = int32
```

For sign-mixed multiplies the **operand mcfg** dictates signedness, not the
mnemonic. To translate the four old mmacc variants when A is in tr_a and B is
in tr_b (so ms1 = tr_a, ms2 = tr_b):

| old mnemonic | mcfg(tr_a) | mcfg(tr_b) | new |
|---|---|---|---|
| `mmacc.w.b`    | int8  | int8  | `mmacc acc, tr_b, tr_a` |
| `mmaccu.w.b`   | uint8 | uint8 | `mmacc acc, tr_b, tr_a` |
| `mmaccsu.w.b`  | int8  | uint8 | `mmacc acc, tr_b, tr_a` |
| `mmaccus.w.b`  | uint8 | int8  | `mmacc acc, tr_b, tr_a` |

Since `tr_a`/`tr_b` reuse the same tile registers across operations with
different signs, `msetcfg` must be re-issued before each operation. The
accumulator mcfg can be set once per region.

## Zmasync sync primitives

| Xuantie | Zames Zmasync | Notes |
|---|---|---|
| `msyncreset tok1` | `msyncregreset sync1` | rename + register naming `sync0..sync15` |
| `mrelease tok1` | `mrelease sync1` | rename only |
| `macquire a0, tok1` | `macquire sync1, a0` | **operand order swapped**: sync register first, then rs1 threshold |
| n/a | `mfence` | new — visibility barrier from non-matrix code into the matrix engine |

Encodings live under `func3=100` in the custom-1 opcode space (`0101011`).
LLVM requires `+zmasync` in addition to `+zames`.

## mstatus / MS bit

Unchanged from Xuantie: MS occupies bits [26:25] in mstatus/sstatus and accepts
the same 4 states (Off / Initial / Clean / Dirty). The Xuantie matrix_init
boilerplate (`lui a0, 8194; addiw a0, a0, 512; csrs mstatus, a0`) still works
and additionally toggles FS and VS so vector/FP state is also enabled. Zames
also adds the `minit` instruction as a shorthand for forcing MS = Initial.

## Capability discovery

`misa` no longer carries Zames. Discover via the device tree:

```
riscv,isa-extensions = "zames";
riscv,matrix-isa-caps = "i8i8i32", "bf16bf16fp32", "fp8e4m3fp8e4m3fp32";
```

The capability string follows `<input_a><input_b><output_c>`. Integer tokens
are `i<W>` / `u<W>`; float tokens use direct names (`bf16`, `fp32`, `fp8e4m3`,
…). Programming an unsupported `mcfg` raises an illegal instruction exception
at `msetcfg` time.

## Reference port

A worked example sits at `nexus-am/apps/ame-mmacc-zames/`. It mirrors the
original `ame-mmacc` test (same data, same expected results — 131, 4145283,
-16253, -32637) but uses the new spec end-to-end. Diff against the original
`ame-mmacc/matop.S` to see every change pattern in context.

### NEMU-matrix mcfg signedness (fixed 2026-05)

`NEMU-matrix/src/isa/riscv64/instr/rvmatrix/mcompute.h` previously hard-coded
`is_signed=true` for all `get_mreg` calls in the integer mmacc path. This was
fixed by deriving `s1_signed = is_signed_int_mtype(s1mcfg.type_code)` and
`s2_signed = is_signed_int_mtype(s2mcfg.type_code)` and passing them to
`get_mreg`. All four sign combinations (ss, uu, su, us) now produce correct
results. `ame-mmacc-zames` MMACC0–4 all pass on NEMU after this change.

## Quick conversion checklist

1. Replace the custom `llvm-mc-rule` in the Makefile with `MARCH = ..._zames_zmasync`.
2. For every typed load/store, drop the type suffix: `mlae8 → mla`, etc.
3. Insert `msetcfg mcfgN, rsN` for each tile/acc used, before the first access.
4. Collapse the four `mmacc*` mnemonics into `mmacc md, ms2, ms1` (swap operand
   order) and encode signedness in `mcfg(ms1)` / `mcfg(ms2)`.
5. `msyncreset → msyncregreset`; sync tokens are `sync0..sync15`; swap
   `macquire` operand order to `macquire sync, rs1`.
6. `make ARCH=riscv64-xs TOOLCHAIN=LLVM` and grep the disassembly to confirm
   the new mnemonics resolved.
