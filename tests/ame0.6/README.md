# ame0.6 Proposal-12 Test

This directory keeps the old `ame0.6` test location, but the implementation has
been ported to AME proposal-12/Zames instructions.

It validates `ggml_ame_gemm_tile_i8_i32_bT()` against a scalar reference for a
single 128x64 by 128x64 signed-int8 GEMM tile:

```text
C(128x128) = A(128x64) * B^T(128x64)
```

Run:

```bash
make run-nemu
```

The Makefile forces `ARCH=riscv64-xs`, `TOOLCHAIN=LLVM`, and uses
`zames/zmasync` for both assembly and disassembly.
