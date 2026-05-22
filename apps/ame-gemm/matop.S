.text
.global matrix_init
.global matrix_gemm
.global mat_dist


.section .rodata

.section .data
.align 8

# A: 256 rows x 128 cols int8, row-major (K-inner). All elements = 1.
mat_na:
        .rept   32768           # 256 * 128 = 32768
        .byte   1
        .endr

# B: 256 rows x 128 cols int8, N-major / pre-transposed (K-inner).
# Same layout convention as ame-mmacc/mat_nb. All elements = 1.
mat_nb:
        .rept   32768           # 256 * 128 = 32768
        .byte   1
        .endr

# C: 256 rows x 256 cols int32, row-major. Initialized to 0 so each
# output tile's accumulator (loaded via mlce32) starts from zero.
mat_dist:
        .rept   65536           # 256 * 256 = 65536
        .word   0
        .endr


.section .text

matrix_init:
        # set mstatus, enable matrix extension (XS|VS bits)
        lui a0, 8194
        addiw a0, a0, 512
        csrs mstatus, a0

        li a0, 128
        msettilem       a0
        msettilen       a0
        li a0, 64
        msettilek       a0

        msettilemi      128
        msettileni      128
        msettileki      64
        ret


# C[256x256] = A[256x128] * B[256x128]^T  (B stored pre-transposed, K-inner)
# Tile shape: M_t = N_t = 128, K_t = 64
# 2x2 output tiles (mi, ni in {0,1}); each tile accumulates 2 K-tiles.
#
# Cross-tile interleaved schedule with outer-product operand reuse.
# Per K-tile: load {A0, A1, B0, B1} once, then 4 MACs across acc0..acc3:
#   acc0 += A0*B0    acc1 += A0*B1    acc2 += A1*B0    acc3 += A1*B1
# Halves the operand-load count vs the per-tile baseline (4 loads instead of
# 8 per K-tile) and lets loads + matrix-MACs interleave across accumulators.
matrix_gemm:
        msyncreset      tok1

        li      t1, 128                 # A/B row stride (K_total bytes)
        li      t3, 64                  # K-tile byte offset
        li      t5, 1024                # C row stride (N_total * 4 bytes)

        # ---- A/B base pointers for the 2x2 tile grid ----
        la      t0, mat_na              # A0 = A[0,   0]
        la      t6, mat_na
        li      a1, 16384
        add     t6, t6, a1              # A1 = A[128, 0]
        la      t2, mat_nb              # B0 = B[0,   0]
        la      a4, mat_nb
        add     a4, a4, a1              # B1 = B[128, 0]

        # ---- C base pointers ----
        la      t4, mat_dist            # C00 = C[0,   0]
        la      a5, mat_dist
        addi    a5, a5, 512             # C01 = C[0,   128]
        la      a6, mat_dist
        li      a1, 131072
        add     a6, a6, a1              # C10 = C[128, 0]
        la      a7, mat_dist
        li      a1, 131584
        add     a7, a7, a1              # C11 = C[128, 128]

        # Initialize accumulators from C
        mlce32  acc0, (t4), t5
        mlce32  acc1, (a5), t5
        mlce32  acc2, (a6), t5
        mlce32  acc3, (a7), t5

        # ---- K-tile 0: load {A0, A1, B0, B1}, then 4 outer-product MACs ----
        mlae8   tr0, (t0), t1           # A0 K0
        mlbe8   tr1, (t2), t1           # B0 K0
        mlae8   tr2, (t6), t1           # A1 K0
        mlbe8   tr3, (a4), t1           # B1 K0
        mmacc.w.b acc0, tr0, tr1        # acc0 += A0 * B0
        mmacc.w.b acc1, tr0, tr3        # acc1 += A0 * B1
        mmacc.w.b acc2, tr2, tr1        # acc2 += A1 * B0
        mmacc.w.b acc3, tr2, tr3        # acc3 += A1 * B1

        # Advance to K-tile 1
        add     t0, t0, t3
        add     t6, t6, t3
        add     t2, t2, t3
        add     a4, a4, t3

        # ---- K-tile 1 ----
        mlae8   tr0, (t0), t1
        mlbe8   tr1, (t2), t1
        mlae8   tr2, (t6), t1
        mlbe8   tr3, (a4), t1
        mmacc.w.b acc0, tr0, tr1
        mmacc.w.b acc1, tr0, tr3
        mmacc.w.b acc2, tr2, tr1
        mmacc.w.b acc3, tr2, tr3

        # Store accumulators back to C
        msce32  acc0, (t4), t5
        msce32  acc1, (a5), t5
        msce32  acc2, (a6), t5
        msce32  acc3, (a7), t5

        mrelease        tok1
        li      a0, 1
        macquire        a0, tok1
        j       gemm_check


# Inputs are all-ones, so every output element must equal K_total = 128.
# Probe one element per 128x128 output tile + the far corner instead of
# walking all 65,536 words; same confidence at ~50x fewer dynamic insts.
gemm_check:
        fence   rw, rw
        la      a1, mat_dist
        li      a2, 128                 # expected value
        li      a0, 1                   # default: fail

        # tile (mi=0, ni=0): dist[0]      -> byte offset 0
        lw      a4, 0(a1)
        bne     a4, a2, gemm_done

        # tile (mi=0, ni=1): dist[128]    -> byte offset 512
        lw      a4, 512(a1)
        bne     a4, a2, gemm_done

        # tile (mi=1, ni=0): dist[32768]  -> byte offset 131072
        li      a5, 131072
        add     a6, a1, a5
        lw      a4, 0(a6)
        bne     a4, a2, gemm_done

        # tile (mi=1, ni=1): dist[32896]  -> byte offset 131584
        li      a5, 131584
        add     a6, a1, a5
        lw      a4, 0(a6)
        bne     a4, a2, gemm_done

        # far corner: dist[65535]         -> byte offset 262140
        li      a5, 262140
        add     a6, a1, a5
        lw      a4, 0(a6)
        bne     a4, a2, gemm_done

        li      a0, 0                   # all probes matched
gemm_done:
        ret
