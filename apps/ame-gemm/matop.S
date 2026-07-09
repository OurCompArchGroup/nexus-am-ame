.text
.global matrix_init_128
.global matrix_gemm_128
.global matrix_init_64
.global matrix_gemm_64
.global mat_dist

// mcfg type_code constants (table_sel = 0)
.equ MCFG_INT8,  0x02
.equ MCFG_INT32, 0x04

.macro SETCFG mcfg_sel, type_code
    li      t6, \type_code
    msetcfg \mcfg_sel, t6
.endm

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
# output tile's accumulator (loaded via mlc) starts from zero.
mat_dist:
        .rept   65536           # 256 * 256 = 65536
        .word   0
        .endr


.section .text

matrix_init_128:
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


matrix_init_64:
        # set mstatus, enable matrix extension (XS|VS bits)
        lui a0, 8194
        addiw a0, a0, 512
        csrs mstatus, a0

        li a0, 64
        msettilem       a0
        msettilen       a0
        msettilek       a0

        msettilemi      64
        msettileni      64
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
#
# Register map:
#   tr0 = A0 (mi=0), tr1 = A1 (mi=1), tr2 = B0 (ni=0), tr3 = B1 (ni=1)
#   acc0..acc3 = C[mi,ni] for (mi,ni) in {(0,0),(0,1),(1,0),(1,1)}
#   t0/a1 = A0/A1 row pointer at current K offset
#   t2/a2 = B0/B1 row pointer at current K offset
#   t4/a4/a3/a5 = C00/C01/C10/C11 base pointers
#   t1=t3=128 (A,B row stride)   t5=1024 (C row stride)   a7=64 (K-tile byte offset)
matrix_gemm_128:
        msyncregreset   sync1

        # Program mcfg for all tile/acc registers used.
        # tr0-tr3 = int8 (mcfg0-3), acc0-acc3 = int32 (mcfg4-7)
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        SETCFG  mcfg5, MCFG_INT32
        SETCFG  mcfg6, MCFG_INT32
        SETCFG  mcfg7, MCFG_INT32

        li      t1, 128                 # A row stride (K_total bytes)
        li      t3, 128                 # B row stride (K_total bytes)
        li      t5, 1024                # C row stride (N_total * 4 bytes)
        li      a7, 64                  # K-tile byte offset

        # ---- A0 / A1 base pointers ----
        la      t0, mat_na              # A0 = A[0,0]
        la      a1, mat_na
        li      a6, 16384               # 128 rows * 128 K bytes
        add     a1, a1, a6              # A1 = A[128,0]

        # ---- B0 / B1 base pointers (B is pre-transposed, ni-major) ----
        la      t2, mat_nb              # B0 = B[0,0]   (ni=0)
        la      a2, mat_nb
        add     a2, a2, a6              # B1 = B[128,0] (ni=1)

        # ---- C00 / C01 / C10 / C11 base pointers ----
        la      t4, mat_dist            # C00 = C[0,0]
        la      a4, mat_dist
        addi    a4, a4, 512             # C01 = C[0,128]   (ni=1: 128*4 bytes)
        la      a3, mat_dist
        li      a6, 131072              # 128 rows * 1024 byte stride
        add     a3, a3, a6              # C10 = C[128,0]
        la      a5, mat_dist
        li      a6, 131584              # 131072 + 512
        add     a5, a5, a6              # C11 = C[128,128]

        # ---- Initialize accumulators from C tiles (all zero) ----
        mlc     acc0, (t4), t5
        mlc     acc1, (a4), t5
        mlc     acc2, (a3), t5
        mlc     acc3, (a5), t5

        # ===== K-tile 0 =====
        mla     tr0, (t0), t1          # A0
        mla     tr1, (a1), t1          # A1
        mlb     tr2, (t2), t3          # B0
        mlb     tr3, (a2), t3          # B1
        mmacc   acc0, tr2, tr0         # acc0 += A0 * B0
        mmacc   acc1, tr3, tr0         # acc1 += A0 * B1
        mmacc   acc2, tr2, tr1         # acc2 += A1 * B0
        mmacc   acc3, tr3, tr1         # acc3 += A1 * B1

        # Advance all four operand pointers to the next K-tile.
        add     t0, t0, a7
        add     a1, a1, a7
        add     t2, t2, a7
        add     a2, a2, a7

        # ===== K-tile 1 =====
        mla     tr0, (t0), t1
        mla     tr1, (a1), t1
        mlb     tr2, (t2), t3
        mlb     tr3, (a2), t3
        mmacc   acc0, tr2, tr0
        mmacc   acc1, tr3, tr0
        mmacc   acc2, tr2, tr1
        mmacc   acc3, tr3, tr1

        # ---- Store accumulators back to C ----
        msc     acc0, (t4), t5
        msc     acc1, (a4), t5
        msc     acc2, (a3), t5
        msc     acc3, (a5), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        j       gemm_check


gemm_check:
        fence   rw, rw
        la      a1, mat_dist
        li      a2, 128                 # expected: K_total * 1 * 1 = 128
        li      a3, 65536               # 256 * 256 elements
        li      a0, 0
gemm_loop:
        lw      a4, 0(a1)
        bne     a4, a2, gemm_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, gemm_loop
        j       gemm_done

gemm_fail:
        li      a0, 1

gemm_done:
        ret


# Top-left C[64x64] = A[64x64] * B[64x64]^T using the existing max-sized
# backing buffers. The row strides remain those of the current storage layout:
# A/B = 128 bytes per row, C = 1024 bytes per row.
matrix_gemm_64:
        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32

        li      t1, 128                 # A row stride
        li      t3, 128                 # B row stride
        li      t5, 1024                # C row stride

        la      t0, mat_na              # A = A[0,0]
        la      t2, mat_nb              # B = B[0,0]
        la      t4, mat_dist            # C = C[0,0]

        mlc     acc0, (t4), t5
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        j       gemm_check_64


gemm_check_64:
        fence   rw, rw
        la      a1, mat_dist
        li      a2, 64                  # expected: K_total * 1 * 1 = 64
        li      a3, 64                  # rows
        li      a0, 0
gemm_row_loop_64:
        mv      a4, a1
        li      a5, 64                  # cols
gemm_col_loop_64:
        lw      a6, 0(a4)
        bne     a6, a2, gemm_fail_64
        addi    a4, a4, 4
        addi    a5, a5, -1
        bnez    a5, gemm_col_loop_64
        addi    a1, a1, 1024            # next row in 256-col backing store
        addi    a3, a3, -1
        bnez    a3, gemm_row_loop_64
        j       gemm_done_64

gemm_fail_64:
        li      a0, 1

gemm_done_64:
        ret
