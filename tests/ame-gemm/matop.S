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

# A: 64 rows x 64 cols int8, row-major (K-inner). All elements = 1.
mat_na:
        .rept   32768             # 64 * 64 = 4096
        .byte   1
        .endr

# B: 64 rows x 64 cols int8, N-major / pre-transposed (K-inner).
# Same layout convention as ame-mmacc/mat_nb. All elements = 1.
mat_nb:
        .rept   32768            # 64 * 64 = 4096
        .byte   1
        .endr

# C: 64 rows x 64 cols int32, row-major. Initialized to 0 so each
# output tile's accumulator (loaded via mlc) starts from zero.
mat_dist:
        .rept   65536            # 64 * 64 = 4096
        .word   0
        .endr


.section .text

#if defined(MXFP_RVV_PACK) && MXFP_RVV_PACK
# Reuse this AM assembly owner. Integer register names below are encoding
# placeholders only: the custom operation reads/writes VECTOR v1/v8/v16.
.global vector_mxfp4_pack32
vector_mxfp4_pack32:
        mv      t5, a1
        li      t0, 32
        vsetvli zero, t0, e32, m8, tu, mu
        vle32.v v8, (a0)
        vmv.v.x v16, a2
        j       .Lfp4_mx_unmasked

#if defined(MXFP_RVV_PACK_BATCH) && MXFP_RVV_PACK_BATCH
#ifndef MXFP_RVV_PACK_LMUL
#define MXFP_RVV_PACK_LMUL 8
#endif
#if MXFP_RVV_PACK_LMUL != 8 && MXFP_RVV_PACK_LMUL != 4 && MXFP_RVV_PACK_LMUL != 2
#error "MXFP_RVV_PACK_LMUL must be 8, 4, or 2"
#endif
#if MXFP_RVV_PACK_LMUL != 8 && defined(MXFP_RVV_PACK_WHOLE) && MXFP_RVV_PACK_WHOLE
#error "Shorter-LMUL diagnostics require the existing bounded stores"
#endif
# LMUL 4/2 split PACK_MX only when vlenb == 16; otherwise reuse m8.
# Keep the original m8 load/broadcast, K32 scale and adjacent-nibble output.
# Same K32 instruction and adjacent-nibble stores as pack32, with one real
# final sticky-fcsr read. Zero blocks perform no memory accesses.
.global vector_mxfp4_pack_blocks
vector_mxfp4_pack_blocks:
        mv      t3, a0
        mv      t5, a1
        mv      t2, a2
        mv      t4, a3
#if MXFP_RVV_PACK_LMUL == 8
        beqz    t4, .Lfp4_batch_done
#else
        beqz    t4, .Lfp4_batch_fcsr
        # Check once per nonzero call; t6 is unchanged by either loop path.
        csrr    t6, vlenb
        addi    t6, t6, -16
#endif
#if defined(MXFP_RVV_PACK_WHOLE) && MXFP_RVV_PACK_WHOLE
        # A K32 packed block is exactly one physical register only at
        # VLEN128. Preserve bounded stores elsewhere, including unaligned
        # outputs. t6 is a per-call choice; zero blocks never touch VL.
        csrr    t6, vlenb
        addi    t6, t6, -16
        andi    t0, t5, 15
        or      t6, t6, t0
#endif
.Lfp4_batch_next:
        li      t0, 32
        vsetvli zero, t0, e32, m8, tu, mu
        vle32.v v8, (t3)
        lbu     t1, 0(t2)
        vmv.v.x v16, t1
        # Same PACK_MX as .Lfp4_mx_unmasked, no new conversion arithmetic.
#if MXFP_RVV_PACK_LMUL != 8
        beqz    t6, .Lfp4_batch_short1
#endif
        .insn r 0x7b, 1, 0x07, x1, x16, x8
#if defined(MXFP_RVV_PACK_WHOLE) && MXFP_RVV_PACK_WHOLE
        bnez    t6, .Lfp4_batch_store1_bounded
        vs1r.v  v1, (t5)
        j       .Lfp4_batch_store1_done
.Lfp4_batch_store1_bounded:
#endif
        vsetivli zero, 16, e8, m1, tu, mu
        vse8.v  v1, (t5)
#if MXFP_RVV_PACK_LMUL != 8
        j       .Lfp4_batch_store1_done
.Lfp4_batch_short1:
#if MXFP_RVV_PACK_LMUL == 4
        vsetivli zero, 16, e32, m4, tu, mu
        .insn r 0x7b, 1, 0x07, x1, x16, x8
        .insn r 0x7b, 1, 0x07, x3, x20, x12
        vsetivli zero, 8, e8, m1, tu, mu
        vse8.v  v1, (t5)
        addi    t0, t5, 8
        vse8.v  v3, (t0)
#else
        vsetivli zero, 8, e32, m2, tu, mu
        .insn r 0x7b, 1, 0x07, x1, x16, x8
        .insn r 0x7b, 1, 0x07, x3, x18, x10
        .insn r 0x7b, 1, 0x07, x5, x20, x12
        .insn r 0x7b, 1, 0x07, x7, x22, x14
        vsetivli zero, 4, e8, m1, tu, mu
        vse8.v  v1, (t5)
        addi    t0, t5, 4
        vse8.v  v3, (t0)
        addi    t0, t0, 4
        vse8.v  v5, (t0)
        addi    t0, t0, 4
        vse8.v  v7, (t0)
#endif
#endif
.Lfp4_batch_store1_done:
        addi    t3, t3, 128
        addi    t5, t5, 16
        addi    t2, t2, 1
        addi    t4, t4, -1
#if defined(MXFP_RVV_PACK_PAIR) && MXFP_RVV_PACK_PAIR
        # The original m8 pair is unchanged. Shorter-LMUL mode 2 uses
        # v2/v4 (m4) or v2/v4/v6/v24 (m2), disjoint from the first pool.
        # Matched two-block schedule. Mode 1 uses v1 twice; mode 2 changes
        # only the second PACK_MX/store to v2. Both destinations are disjoint
        # from the unchanged m8 sources v8-v15 and scales v16-v23.
        beqz    t4, .Lfp4_batch_done
        li      t0, 32
        vsetvli zero, t0, e32, m8, tu, mu
        vle32.v v8, (t3)
        lbu     t1, 0(t2)
        vmv.v.x v16, t1
#if MXFP_RVV_PACK_LMUL != 8
        beqz    t6, .Lfp4_batch_short2
#endif
#if MXFP_RVV_PACK_PAIR == 2
        .insn r 0x7b, 1, 0x07, x2, x16, x8
#else
        .insn r 0x7b, 1, 0x07, x1, x16, x8
#endif
#if defined(MXFP_RVV_PACK_WHOLE) && MXFP_RVV_PACK_WHOLE
        bnez    t6, .Lfp4_batch_store2_bounded
#if MXFP_RVV_PACK_PAIR == 2
        vs1r.v  v2, (t5)
#else
        vs1r.v  v1, (t5)
#endif
        j       .Lfp4_batch_store2_done
.Lfp4_batch_store2_bounded:
#endif
        vsetivli zero, 16, e8, m1, tu, mu
#if MXFP_RVV_PACK_PAIR == 2
        vse8.v  v2, (t5)
#else
        vse8.v  v1, (t5)
#endif
#if MXFP_RVV_PACK_LMUL != 8
        j       .Lfp4_batch_store2_done
.Lfp4_batch_short2:
#if MXFP_RVV_PACK_LMUL == 4
        vsetivli zero, 16, e32, m4, tu, mu
#if MXFP_RVV_PACK_PAIR == 2
        .insn r 0x7b, 1, 0x07, x2, x16, x8
        .insn r 0x7b, 1, 0x07, x4, x20, x12
        vsetivli zero, 8, e8, m1, tu, mu
        vse8.v  v2, (t5)
        addi    t0, t5, 8
        vse8.v  v4, (t0)
#else
        .insn r 0x7b, 1, 0x07, x1, x16, x8
        .insn r 0x7b, 1, 0x07, x3, x20, x12
        vsetivli zero, 8, e8, m1, tu, mu
        vse8.v  v1, (t5)
        addi    t0, t5, 8
        vse8.v  v3, (t0)
#endif
#else
        vsetivli zero, 8, e32, m2, tu, mu
#if MXFP_RVV_PACK_PAIR == 2
        .insn r 0x7b, 1, 0x07, x2, x16, x8
        .insn r 0x7b, 1, 0x07, x4, x18, x10
        .insn r 0x7b, 1, 0x07, x6, x20, x12
        .insn r 0x7b, 1, 0x07, x24, x22, x14
        vsetivli zero, 4, e8, m1, tu, mu
        vse8.v  v2, (t5)
        addi    t0, t5, 4
        vse8.v  v4, (t0)
        addi    t0, t0, 4
        vse8.v  v6, (t0)
        addi    t0, t0, 4
        vse8.v  v24, (t0)
#else
        .insn r 0x7b, 1, 0x07, x1, x16, x8
        .insn r 0x7b, 1, 0x07, x3, x18, x10
        .insn r 0x7b, 1, 0x07, x5, x20, x12
        .insn r 0x7b, 1, 0x07, x7, x22, x14
        vsetivli zero, 4, e8, m1, tu, mu
        vse8.v  v1, (t5)
        addi    t0, t5, 4
        vse8.v  v3, (t0)
        addi    t0, t0, 4
        vse8.v  v5, (t0)
        addi    t0, t0, 4
        vse8.v  v7, (t0)
#endif
#endif
#endif
.Lfp4_batch_store2_done:
        addi    t3, t3, 128
        addi    t5, t5, 16
        addi    t2, t2, 1
        addi    t4, t4, -1
#endif
        bnez    t4, .Lfp4_batch_next
.Lfp4_batch_done:
#if MXFP_RVV_PACK_LMUL != 8 || (defined(MXFP_RVV_PACK_WHOLE) && MXFP_RVV_PACK_WHOLE)
        # Nonzero calls retain the old wrapper's final VL/vtype. Zero
        # calls must not change the caller's incoming vector configuration.
#if MXFP_RVV_PACK_LMUL == 8
        beqz    a3, .Lfp4_batch_fcsr
#endif
        vsetivli zero, 16, e8, m1, tu, mu
.Lfp4_batch_fcsr:
#endif
        frcsr   a0
        ret
#endif

.global vector_test_fp4_pack
vector_test_fp4_pack:
        # values/scales/out/old/config; config = vl/vtype/mask/mode.
        mv      t5, a2
        vsetivli zero, 16, e8, m1, tu, mu
        vle8.v  v1, (a3)
        ld      t0, 0(a4)
        ld      t1, 8(a4)
        ld      t4, 24(a4)
        vsetvl  zero, t0, t1
        vle32.v v8, (a0)
        vle32.v v16, (a1)
        addi    t1, a4, 16
        vlm.v   v0, (t1)
        andi    t0, t4, 1
        bnez    t0, .Lfp4_test_mx
        andi    t0, t4, 2
        bnez    t0, .Lfp4_rne_masked
        # funct6=000010, vm=1, vs2=v8, vs1=0, vd=v1.
        .insn r 0x7b, 1, 0x05, x1, x0, x8
        j       .Lfp4_store
.Lfp4_rne_masked:
        .insn r 0x7b, 1, 0x04, x1, x0, x8
        j       .Lfp4_store
.Lfp4_test_mx:
        andi    t0, t4, 2
        bnez    t0, .Lfp4_mx_masked
.Lfp4_mx_unmasked:
        # funct6=000011, vm=1, vs2=v8, vs1=v16, vd=v1.
        .insn r 0x7b, 1, 0x07, x1, x16, x8
        j       .Lfp4_store
.Lfp4_mx_masked:
        .insn r 0x7b, 1, 0x06, x1, x16, x8
.Lfp4_store:
        # Packed vd is EMUL=LMUL/8. Store the physical register so the test
        # also observes preservation above the logical destination extent.
        vsetivli zero, 16, e8, m1, tu, mu
        vse8.v  v1, (t5)
        frcsr   a0
        ret
#endif

matrix_init_128:
        # set mstatus, enable matrix extension (XS|VS bits)
        lui a0, 8194
        addiw a0, a0, 512
        csrs mstatus, a0

        li a0, 64
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
        # tr0-tr1 = int8 (mcfg0-1), acc0 = int32 (mcfg4)
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        SETCFG  mcfg5, MCFG_INT32
        SETCFG  mcfg6, MCFG_INT32
        SETCFG  mcfg7, MCFG_INT32

        li      t1, 64                  # A row stride (K_total bytes)
        li      t3, 64                  # B row stride (K_total bytes)
        li      t5, 256                 # C row stride (N_total * 4 bytes)

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
        li a7, 64
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
        li      a2, 128                  # expected: K_total * 1 * 1 = 64
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
