# AME Proposal-14 GEMM Fuzz Test

This test uses the current Zames proposal-14 instruction syntax and compares
each matrix result with a scalar reference implementation.

It validates `ggml_ame_gemm_tile_i8_i32_bT()` for a single 128x64 by 128x64
signed-int8 GEMM tile:

```text
C(128x128) = A(128x64) * B^T(128x64)
```

Run:

```bash
make run-nemu
```

The Makefile forces `ARCH=riscv64-xs`, `TOOLCHAIN=LLVM`, and uses
`zames/zmasync` for both assembly and disassembly.
