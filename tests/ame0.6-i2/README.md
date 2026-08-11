# ame0.6-i2 Proposal-12 Test

This test validates signed i2 * signed i8 -> signed i32 GEMM on the proposal-12
instruction set.

The instruction stream uses normal Zames/proposal-12 instructions:

- `msetcfg mcfg0, 0x0d` for `tr0`, where `fp2pack4` is treated as signed int2.
- `msetcfg mcfg1, 0x02` for signed int8 `tr1`.
- `msetcfg mcfg4, 0x04` for signed int32 `acc0`.
- `mla`, `mlb`, `mlc`, `mmacc acc0, tr1, tr0`, and `msc`.

A is packed as four signed 2-bit lanes per byte in low-bit lane order. B is
stored as B-transposed with element `B[n * K + k]`.

Run:

```bash
make run-nemu
```
