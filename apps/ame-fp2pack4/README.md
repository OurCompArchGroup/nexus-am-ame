# AME fp2pack4 directed test

This app validates one full `DefaultMatrixConfig` operation:

```text
C[128][128] += A_i8[128][64] * B_i2[128][64]^T
```

`B` is a dense 2 KiB panel with a 16-byte row stride. Within each byte,
`B[k+0]` occupies bits 1:0 and `B[k+3]` occupies bits 7:6. Codes are signed
two's-complement i2 (`0`, `1`, `-2`, `-1`), and every packed byte contains all
four codes. The A rows repeat a K-dimensional identity matrix: the first 64
rows use `+1` and the remaining 64 use `-3`. Every expanded B lane is therefore
multiplied by both positive and negative signed-i8 activations. The scalar check
verifies the complete i32 output tile, including its nonzero initial
accumulator values.

Unsupported packed-load shapes and placements are outside this directed test's
defined behavior (UB); the processor and CUTE matrix path must not expose a
matrix-specific exception interface for them. The directed test covers only the
supported normal packed-B operation and its numerical result.

Build or run with the shared workspace environment:

```bash
make
make run-nemu
```
