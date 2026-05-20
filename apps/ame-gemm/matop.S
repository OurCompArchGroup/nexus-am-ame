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
matrix_gemm:
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

        # ---- Output tile (mi=0, ni=0) -> acc0 -> C[0:128, 0:128] ----
        la      t0, mat_na              # A[0,0]
        la      t2, mat_nb              # B[0,0]
        la      t4, mat_dist            # C[0,0]
        mlc     acc0, (t4), t5          # acc0 = 0 (mat_dist initialized to 0)
        mla     tr0, (t0), t1           # K-tile 0
        mlb     tr1, (t2), t3
        mmacc   acc0, tr1, tr0          # acc0 += tr0 * tr1
        add     t0, t0, a7
        add     t2, t2, a7
        mla     tr0, (t0), t1           # K-tile 1
        mlb     tr1, (t2), t3
        mmacc   acc0, tr1, tr0          # acc0 += tr0 * tr1
        msc     acc0, (t4), t5

        # ---- Output tile (mi=0, ni=1) -> acc1 -> C[0:128, 128:256] ----
        la      t0, mat_na              # A[0,0]
        la      t2, mat_nb
        li      a6, 16384               # 128 * 128 = N_tile rows of B^T
        add     t2, t2, a6              # B[128,0] (ni=1)
        la      t4, mat_dist
        addi    t4, t4, 512             # ni=1: 128 elements * 4 bytes
        mlc     acc1, (t4), t5
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mmacc   acc1, tr3, tr2          # acc1 += tr2 * tr3
        add     t0, t0, a7
        add     t2, t2, a7
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mmacc   acc1, tr3, tr2          # acc1 += tr2 * tr3
        msc     acc1, (t4), t5

        # ---- Output tile (mi=1, ni=0) -> acc2 -> C[128:256, 0:128] ----
        la      t0, mat_na
        li      a6, 16384               # 128 rows * K_total(128) = 16384
        add     t0, t0, a6              # A[128,0]
        la      t2, mat_nb              # B[0,0]
        la      t4, mat_dist
        li      a6, 131072              # 128 rows * 1024 stride
        add     t4, t4, a6              # C[128,0]
        mlc     acc2, (t4), t5
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mmacc   acc2, tr1, tr0          # acc2 += tr0 * tr1
        add     t0, t0, a7
        add     t2, t2, a7
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mmacc   acc2, tr1, tr0          # acc2 += tr0 * tr1
        msc     acc2, (t4), t5

        # ---- Output tile (mi=1, ni=1) -> acc3 -> C[128:256, 128:256] ----
        la      t0, mat_na
        li      a6, 16384
        add     t0, t0, a6              # A[128,0]
        la      t2, mat_nb
        li      a6, 16384
        add     t2, t2, a6              # B[128,0]
        la      t4, mat_dist
        li      a6, 131584              # 131072 + 512 = C[128,128]
        add     t4, t4, a6
        mlc     acc3, (t4), t5
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mmacc   acc3, tr3, tr2          # acc3 += tr2 * tr3
        add     t0, t0, a7
        add     t2, t2, a7
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mmacc   acc3, tr3, tr2          # acc3 += tr2 * tr3
        msc     acc3, (t4), t5

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
