.text
.global matrix_init
.global matrix_gemm
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

# A: 64 rows x 64 cols int8, row-major (K-inner). All elements = 1.
mat_na:
        .rept   4096            # 64 * 64 = 4096
        .byte   1
        .endr

# B: 64 rows x 64 cols int8, N-major / pre-transposed (K-inner).
# Same layout convention as ame-mmacc/mat_nb. All elements = 1.
mat_nb:
        .rept   4096            # 64 * 64 = 4096
        .byte   1
        .endr

# C: 64 rows x 64 cols int32, row-major. Initialized to 0 so each
# output tile's accumulator (loaded via mlc) starts from zero.
mat_dist:
        .rept   4096            # 64 * 64 = 4096
        .word   0
        .endr


.section .text

matrix_init:
        # set mstatus, enable matrix extension (XS|VS bits)
        lui a0, 8194
        addiw a0, a0, 512
        csrs mstatus, a0

        li a0, 64
        msettilem       a0
        msettilen       a0
        li a0, 64
        msettilek       a0

        msettilemi      64
        msettileni      64
        msettileki      64
        ret


# C[64x64] = A[64x64] * B[64x64]^T  (B stored pre-transposed, K-inner)
# Tile shape: M_t = N_t = K_t = 64. A/B are int8, C is int32.
matrix_gemm:
        msyncregreset   sync1

        # Program mcfg for all tile/acc registers used.
        # tr0-tr1 = int8 (mcfg0-1), acc0 = int32 (mcfg4)
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32

        li      t1, 64                  # A row stride (K_total bytes)
        li      t3, 64                  # B row stride (K_total bytes)
        li      t5, 256                 # C row stride (N_total * 4 bytes)

        # ---- Single output tile -> acc0 -> C[0:64, 0:64] ----
        la      t0, mat_na              # A[0,0]
        la      t2, mat_nb              # B[0,0]
        la      t4, mat_dist            # C[0,0]
        mlc     acc0, (t4), t5          # acc0 = 0 (mat_dist initialized to 0)
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mmacc   acc0, tr1, tr0          # acc0 += tr0 * tr1
        msc     acc0, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        j       gemm_check


gemm_check:
        fence   rw, rw
        la      a1, mat_dist
        li      a2, 64                  # expected: K_total * 1 * 1 = 64
        li      a3, 4096                # 64 * 64 elements
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
