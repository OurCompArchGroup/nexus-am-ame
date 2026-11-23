.text
.global matrix_init_128
.global matrix_gemm_128
.global matrix_init_64
.global matrix_gemm_64
.global matrix_gemm_mxfp8_e4m3
.global matrix_gemm_mxfp8_e5m2
.global matrix_gemm_mxfp4
.global matrix_gemm_mxcase
.global matrix_test_xmfrm
.global matrix_test_mxlife
.global matrix_test_mxpanel
.global mat_dist

// mcfg type_code constants (table_sel = 0)
.equ MCFG_INT8,  0x02
.equ MCFG_INT32, 0x04

.macro SETCFG mcfg_sel, type_code
    li      t6, \type_code
    msetcfg \mcfg_sel, t6
.endm

# Experimental block-scale load encoding from the MX32 proposal.
# func3=001, func7=0011000 for mlascale / 1011000 for mlbscale.
# Use x0/x1 as the encoded scale-register index placeholders.
.macro MLASCALE_RAW scale_index, base, stride
    .insn r 0x2b, 1, 0x18, \scale_index, \base, \stride
.endm

.macro MLBSCALE_RAW scale_index, base, stride
    .insn r 0x2b, 1, 0x58, \scale_index, \base, \stride
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

.align 8
mat_mxfp8_e4m3:
        .rept   32
        .byte   0x38
        .endr
mat_mxfp8_e5m2:
        .rept   32
        .byte   0x3c
        .endr
mat_mxfp4_a:
        .rept   16
        .byte   0x21
        .endr
mat_mxfp4_b:
        .rept   16
        .byte   0x43
        .endr
mat_mx_scale:
        .byte   0x7f
mat_mx_result:
        .word   0
mat_mx_case_c:
        .word   0
mat_mx_case_result:
        .word   0
mat_xmfrm_trap_count:
        .dword  0
mat_xmfrm_trap_cause:
        .dword  0


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

.macro MX_INIT mcfg_value
        lui     a0, 8194
        addiw   a0, a0, 512
        csrs    mstatus, a0
        msettilemi      1
        msettileni      1
        msettileki      32
        li      t6, \mcfg_value
        msetcfg mcfg0, t6
        msetcfg mcfg1, t6
        li      t6, 0x0c
        msetcfg mcfg4, t6
.endm

.macro MX_RUN a_data, a_stride, b_data, b_stride, expected
        msyncregreset   sync1
        la      t0, \a_data
        li      t1, \a_stride
        mla     tr0, (t0), t1
        la      a0, \b_data
        li      a1, \b_stride
        mlb     tr1, (a0), a1
        la      t2, mat_mx_scale
        li      t3, 1
        MLASCALE_RAW x0, t2, t3
        la      a2, mat_mx_scale
        li      a3, 1
        MLBSCALE_RAW x1, a2, a3
        la      t4, mat_mx_result
        li      t5, 4
        sw      x0, 0(t4)
        mfence
        mlc     acc0, (t4), t5
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5
        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        lw      a4, 0(t4)
        li      a5, \expected
        li      a0, 0
        beq     a4, a5, 1f
        li      a0, 1
1:
        ret
.endm

matrix_gemm_mxfp8_e4m3:
        MX_INIT 0x18
        MX_RUN mat_mxfp8_e4m3, 32, mat_mxfp8_e4m3, 32, 0x42000000

matrix_gemm_mxfp8_e5m2:
        MX_INIT 0x17
        MX_RUN mat_mxfp8_e5m2, 32, mat_mxfp8_e5m2, 32, 0x42000000

matrix_gemm_mxfp4:
        MX_INIT 0x16
        # A bytes 0x21, B bytes 0x43.
        # low/high values per pair: 0.5*1.5 + 1*2 = 2.75.
        # 16 pairs make 44.0. Swapping A's two nibbles makes 72.0.
        MX_RUN mat_mxfp4_a, 16, mat_mxfp4_b, 16, 0x42300000

# Execute one raw NEMU exact-block vector with the real instruction path.
# a0 = mcfg, a1 = K, a2 = scale_a array, a3 = scale_b array,
# a4 = packed A, a5 = packed B, a6 = C bits, a7 = A/B row stride.
# Returns actual result bits in a0 and accrued mfflags in a1.
matrix_gemm_mxcase:
        addi    sp, sp, -32
        sd      ra, 24(sp)
        sd      s0, 16(sp)
        sd      s1, 8(sp)
        mv      s0, a4
        mv      s1, a5
        msyncregreset   sync1

        lui     t0, 8194
        addiw   t0, t0, 512
        csrs    mstatus, t0
        li      t0, 1
        msettilem       t0
        msettilen       t0
        msettilek       a1
        li      t0, 0
        csrw    0x809, t0
        li      t0, 8
        csrw    0x808, t0
        msetcfg mcfg0, a0
        msetcfg mcfg1, a0
        li      t0, 0x0c
        msetcfg mcfg4, t0
        # Publish caller-produced A/B/scale bytes before asynchronous loads.
        mfence
        mv      t0, s0
        mla     tr0, (t0), a7
        mv      t0, s1
        mla     tr1, (t0), a7
        mv      t0, a2
        addi    t1, a1, 31
        srli    t1, t1, 5
        MLASCALE_RAW x0, t0, t1
        mv      t0, a3
        MLBSCALE_RAW x1, t0, t1

        la      t2, mat_mx_case_c
        sw      a6, 0(t2)
        mfence
        li      t3, 4
        mlc     acc0, (t2), t3
        mmacc   acc0, tr1, tr0
        la      t2, mat_mx_case_result
        msc     acc0, (t2), t3
        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        lwu     a0, 0(t2)
        csrr    a1, 0x808
        andi    a1, a1, 0x1f
        slli    t0, a1, 32
        slli    a0, a0, 32
        srli    a0, a0, 32
        or      a0, a0, t0

        ld      ra, 24(sp)
        ld      s0, 16(sp)
        ld      s1, 8(sp)
        addi    sp, sp, 32
        ret

# Existing native-panel probe, parameterized for a bounded K loop.
# a0=A, a1=B, a2=scale A, a3=scale B, a4=C, a5=MxPanelConfig.
# Config u64 offsets: rows0/tile_k8/k_tiles16/payload_stride24/
# scale_stride32/out_stride40. No alternate GEMM kernel or ISA is added.
# Scale panels are compact [rows][tile_k/32] bytes, consecutive in K.
# Keep address/stride GPRs live through acquire; return matrix flags.
matrix_test_mxpanel:
#if defined(MXFP_RMS_DYNAMIC) && MXFP_RMS_DYNAMIC
        li      t5, 0           # original entry never reads appended controls
        j       .Lmxpanel_entry
.global matrix_test_mxrms_panel
matrix_test_mxrms_panel:
        ld      t5, 56(a5)      # MxRmsPanelConfig stage1=load,2=consume
.Lmxpanel_entry:
#endif
        addi    sp, sp, -64
        sd      s0, 0(sp)
        sd      s1, 8(sp)
        sd      s2, 16(sp)
        sd      s3, 24(sp)
        sd      s4, 32(sp)
        sd      s5, 40(sp)
        sd      s6, 48(sp)
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
        sd      s7, 56(sp)
        ld      s7, 48(a5)       # opt-in source F32; ordinary callers use zero
#endif
        ld      s0, 0(a5)
        ld      s1, 8(a5)
        ld      s2, 16(a5)
        ld      s3, 24(a5)
        ld      s4, 32(a5)
        ld      s5, 40(a5)
        lui     t0, 8194
        addiw   t0, t0, 512
        csrs    mstatus, t0
#if defined(MXFP_RMS_DYNAMIC) && MXFP_RMS_DYNAMIC
        bnez    t5, .Lmxpanel_rms_setup
#endif
        msettilem       s0
        msettilen       s0
        msettilek       s1
        li      t0, 0x16
        msetcfg mcfg0, t0
        msetcfg mcfg1, t0
        li      t0, 0x0c
        msetcfg mcfg4, t0
        csrw    0x808, zero
        csrrw   s6, 0x809, zero
        msyncregreset   sync1
        mzero   acc0
        mfence
        li      t4, 0
.Lmxpanel_k:
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
        beqz    s7, .Lmxpanel_ordinary
        # Publish real SF first. The conversion's owned generation must not
        # depend on an uncompleted scale load or on a later live SF register.
        MLASCALE_RAW x0, a2, s4
        MLBSCALE_RAW x1, a3, s4
        mrelease        sync1
        addi    t4, t4, 1
        macquire        sync1, t4
        mfence
        # Explicit experimental R27 encodings; no compiler dialect alias.
        .insn r 0x2b, 1, 0x1a, x0, a0, s3
        .insn r 0x2b, 1, 0x5a, x1, a1, s3
        j       .Lmxpanel_load_wait
.Lmxpanel_ordinary:
#endif
        mla     tr0, (a0), s3
        mlb     tr1, (a1), s3
        MLASCALE_RAW x0, a2, s4
        MLBSCALE_RAW x1, a3, s4
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
.Lmxpanel_load_wait:
#endif
        mrelease        sync1
        addi    t4, t4, 1
        macquire        sync1, t4
        mfence
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
        # Reject before MMACC, including on a later K tile. The caller will
        # restart this complete operator through CPU quant + ordinary loads.
        beqz    s7, .Lmxpanel_compute
        csrr    t0, 0x808
        andi    t0, t0, 16
        bnez    t0, .Lmxpanel_return
#endif
#if (defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD) || \
    (defined(MXFP_RMS_DYNAMIC) && MXFP_RMS_DYNAMIC)
.Lmxpanel_compute:
#endif
        mmacc   acc0, tr1, tr0
        mrelease        sync1
        addi    t4, t4, 1
        macquire        sync1, t4
        mfence
        srli    t0, s1, 1
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
        beqz    s7, .Lmxpanel_advance
        slli    t0, s1, 2        # F32 source K64 = 256B, not packed 32B
.Lmxpanel_advance:
#endif
        add     a0, a0, t0
        add     a1, a1, t0
        mul     t0, s0, s4
        add     a2, a2, t0
        add     a3, a3, t0
        addi    s2, s2, -1
        bnez    s2, .Lmxpanel_k
        msc     acc0, (a4), s5
        mrelease        sync1
        addi    t4, t4, 1
        macquire        sync1, t4
        mfence
#if (defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD) || \
    (defined(MXFP_RMS_DYNAMIC) && MXFP_RMS_DYNAMIC)
.Lmxpanel_return:
#endif
        csrr    a0, 0x808
        andi    a0, a0, 31
#if defined(MXFP_RMS_DYNAMIC) && MXFP_RMS_DYNAMIC
.Lmxpanel_restore:
#endif
        csrw    0x809, s6
        ld      s0, 0(sp)
        ld      s1, 8(sp)
        ld      s2, 16(sp)
        ld      s3, 24(sp)
        ld      s4, 32(sp)
        ld      s5, 40(sp)
        ld      s6, 48(sp)
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
        ld      s7, 56(sp)
#endif
        addi    sp, sp, 64
        ret

#if defined(MXFP_RMS_DYNAMIC) && MXFP_RMS_DYNAMIC
# New default-off entry only. No dynamic request is issued without the C
# caller's explicit external config/model attestation and true prefix NX.
.Lmxpanel_rms_setup:
        # Drain before clearing flags: old sticky/late NV is not owned status.
        msyncregreset   sync1
        mrelease        sync1
        li      t4, 1
        macquire        sync1, t4
        mfence
        csrr    t2, 0x808
        andi    t2, t2, 31
        csrrw   s6, 0x809, zero
        li      t0, 1
        beq     t5, t0, .Lmxpanel_rms_load_setup
        # Consume only a previously successful, acquired generation.
        beqz    t2, .Lmxpanel_rms_load_setup
        mv      a0, t2
        j       .Lmxpanel_restore
.Lmxpanel_rms_load_setup:
        csrw    0x808, zero
        msettilem       s0
        msettilen       s0
        msettilek       s1
        li      t0, 0x16
        msetcfg mcfg0, t0
        msetcfg mcfg1, t0
        li      t0, 0x0c
        msetcfg mcfg4, t0
        li      t0, 1
        bne     t5, t0, .Lmxpanel_rms_consume
        ld      t0, 64(a5)      # tested side A=0/B=1
        ld      t1, 72(a5)      # NOT the original supplied convert_f32 selector
        mfence
        bnez    t0, .Lmxpanel_rms_load_b
        bnez    t1, .Lmxpanel_rms_dynamic_a
        mla     tr0, (a0), s3
        MLASCALE_RAW x0, a2, s4
        j       .Lmxpanel_rms_ready
.Lmxpanel_rms_dynamic_a:
        # Frozen experimental candidate, not an official Zames allocation.
        .insn r 0x2b, 1, 0x1c, x0, a0, s3
        j       .Lmxpanel_rms_ready
.Lmxpanel_rms_load_b:
        bnez    t1, .Lmxpanel_rms_dynamic_b
        mlb     tr1, (a0), s3
        MLBSCALE_RAW x1, a2, s4
        j       .Lmxpanel_rms_ready
.Lmxpanel_rms_dynamic_b:
        .insn r 0x2b, 1, 0x5c, x1, a0, s3
.Lmxpanel_rms_ready:
        mrelease        sync1
        addi    t4, t4, 1
        macquire        sync1, t4
        mfence
        csrr    a0, 0x808
        andi    a0, a0, 31
        slli    t2, t2, 32      # separately report drained OLD flags
        or      a0, a0, t2
        j       .Lmxpanel_restore
.Lmxpanel_rms_consume:
        # Identity companion only. Do NOT reload the tested TR or SF bank.
        ld      t0, 64(a5)
        li      t1, 64
        mfence
        bnez    t0, .Lmxpanel_rms_companion_a
        msettilen       t1      # tested A8x64 times identity B64x64
        mzero   acc0
        mlb     tr1, (a1), s3
        MLBSCALE_RAW x1, a3, s4
        j       .Lmxpanel_rms_companion_wait
.Lmxpanel_rms_companion_a:
        msettilem       t1      # identity A64x64 times tested B8x64
        mzero   acc0
        mla     tr0, (a0), s3
        MLASCALE_RAW x0, a2, s4
.Lmxpanel_rms_companion_wait:
        mrelease        sync1
        addi    t4, t4, 1
        macquire        sync1, t4
        mfence
        j       .Lmxpanel_compute
#endif

# Separate lifecycle probe: the frozen mxcase helper/table stay unchanged.
# Stack: A[64], B[64], scale A/B at128/129, C at132, result at136.
# s0 addresses fifteen caller-owned u64 observations; s1 saves scalar flags.
.macro MXLIFE_CONFIG
        li      t0, 1
        msettilem       t0
        msettilen       t0
        li      t0, 32
        msettilek       t0
        li      t0, 0
        csrw    0x809, t0
        li      t0, 0x17
        msetcfg mcfg0, t0
        msetcfg mcfg1, t0
        li      t0, 0x0c
        msetcfg mcfg4, t0
.endm

.macro MXLIFE_LOAD
        mfence
        addi    t0, sp, 0
        li      t1, 32
        mla     tr0, (t0), t1
        addi    t0, sp, 64
        mla     tr1, (t0), t1
        addi    t0, sp, 128
        li      t1, 1
        MLASCALE_RAW x0, t0, t1
        addi    t0, sp, 129
        MLBSCALE_RAW x1, t0, t1
.endm

.macro MXLIFE_CALC result_offset, flags_offset
        msyncregreset   sync1
        sw      zero, 132(sp)
        mfence
        addi    t2, sp, 132
        li      t3, 4
        mlc     acc0, (t2), t3
        mmacc   acc0, tr1, tr0
        addi    t2, sp, 136
        msc     acc0, (t2), t3
        mrelease        sync1
        li      t0, 1
        macquire        sync1, t0
        lwu     t1, 136(sp)
        sd      t1, \result_offset(s0)
        csrr    t1, 0x808
        andi    t1, t1, 31
        sd      t1, \flags_offset(s0)
.endm

matrix_test_mxlife:
        addi    sp, sp, -192
        sd      ra, 184(sp)
        sd      s0, 160(sp)
        sd      s1, 168(sp)
        mv      s0, a0
        lui     t0, 8194
        addiw   t0, t0, 512
        csrs    mstatus, t0
        csrr    s1, fflags
        li      t0, 2
        csrw    fflags, t0
        li      t0, 1
        csrw    0x808, t0
        # One nonzero product in K32; do not depend on discarded register data.
        addi    t0, sp, 0
        li      t1, 16
.Lmxlife_zero:
        sd      zero, 0(t0)
        addi    t0, t0, 8
        addi    t1, t1, -1
        bnez    t1, .Lmxlife_zero
        li      t0, 0x7d
        sb      t0, 0(sp)
        li      t0, 0x3c
        sb      t0, 64(sp)
        li      t0, 127
        sb      t0, 128(sp)
        sb      t0, 129(sp)
        MXLIFE_CONFIG
        MXLIFE_LOAD
        MXLIFE_CALC 0, 8
        csrr    t0, fflags
        sd      t0, 16(s0)
        # The preceding acquire makes flags observations legal across MINIT.
        minit
        csrr    t0, mstatus
        srli    t0, t0, 25
        andi    t0, t0, 3
        sd      t0, 24(s0)
        csrr    t0, 0x808
        andi    t0, t0, 31
        sd      t0, 32(s0)
        csrr    t0, fflags
        sd      t0, 40(s0)
        # Reload every operand and scale after MINIT; changed scale means2.0.
        li      t0, 0x3c
        sb      t0, 0(sp)
        li      t0, 128
        sb      t0, 128(sp)
        MXLIFE_CONFIG
        MXLIFE_LOAD
        MXLIFE_CALC 48, 56
        # Same epoch, same scale generation: reinitialize C, no scale reload.
        MXLIFE_CALC 64, 72
        # Discard only new loads, never an unacquired arithmetic observation.
        # A trace must establish that old work was still pending at MINIT.
        MXLIFE_LOAD
        minit
        csrr    t0, mstatus
        srli    t0, t0, 25
        andi    t0, t0, 3
        sd      t0, 80(s0)
        csrr    t0, 0x808
        andi    t0, t0, 31
        sd      t0, 88(s0)
        li      t0, 129
        sb      t0, 128(sp)
        MXLIFE_CONFIG
        MXLIFE_LOAD
        MXLIFE_CALC 96, 104
        csrr    t0, fflags
        sd      t0, 112(s0)
        csrw    fflags, s1
        ld      ra, 184(sp)
        ld      s0, 160(sp)
        ld      s1, 168(sp)
        addi    sp, sp, 192
        ret

# Test every non-RNE xmfrm value against a valid MXFP4 mmacc. The M-mode
# exception handler must report illegal instruction and resume after mmacc.
# Return: C[31:0], xmfrm[39:32], mcause[47:40], trap count[55:48].
matrix_test_xmfrm:
        addi    sp, sp, -64
        sd      ra, 56(sp)
        sd      s0, 48(sp)
        sd      s1, 40(sp)
        sd      s2, 32(sp)
        sd      s3, 24(sp)
        sd      s4, 16(sp)
        mv      s0, a0
        msyncregreset   sync1

        la      t0, mat_xmfrm_trap_count
        sd      zero, 0(t0)
        la      t0, mat_xmfrm_trap_cause
        li      t1, -1
        sd      t1, 0(t0)
        csrr    s1, mtvec
        la      t0, mxfrm_trap_handler
        csrw    mtvec, t0

        lui     t0, 8194
        addiw   t0, t0, 512
        csrs    mstatus, t0
        li      t0, 1
        msettilem       t0
        msettilen       t0
        li      t0, 32
        msettilek       t0
        li      t0, 0x16
        msetcfg mcfg0, t0
        msetcfg mcfg1, t0
        li      t0, 0x0c
        msetcfg mcfg4, t0

        la      t0, mat_mxfp4_a
        li      t1, 16
        mla     tr0, (t0), t1
        la      t0, mat_mxfp4_b
        mlb     tr1, (t0), t1
        la      t0, mat_mx_scale
        li      t1, 1
        MLASCALE_RAW x0, t0, t1
        la      t0, mat_mx_scale
        MLBSCALE_RAW x1, t0, t1

        la      t2, mat_mx_case_c
        li      t3, 0x12345678
        sw      t3, 0(t2)
        mfence
        li      t3, 4
        mlc     acc0, (t2), t3
        csrw    xmfrm, s0
        mmacc   acc0, tr1, tr0
        la      t2, mat_mx_case_result
        msc     acc0, (t2), t3
        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0

        la      t2, mat_mx_case_c
        lw      s2, 0(t2)
        csrr    s3, xmfrm
        la      t2, mat_xmfrm_trap_cause
        ld      t0, 0(t2)
        slli    t0, t0, 40
        la      t2, mat_xmfrm_trap_count
        ld      t1, 0(t2)
        slli    t1, t1, 48
        slli    s3, s3, 32
        or      s2, s2, s3
        or      s2, s2, t0
        or      s2, s2, t1

        csrw    mtvec, s1
        li      t0, 0
        csrw    xmfrm, t0
        mv      a0, s2
        ld      ra, 56(sp)
        ld      s0, 48(sp)
        ld      s1, 40(sp)
        ld      s2, 32(sp)
        ld      s3, 24(sp)
        ld      s4, 16(sp)
        addi    sp, sp, 64
        ret

.align 4
mxfrm_trap_handler:
        addi    sp, sp, -32
        sd      t0, 0(sp)
        sd      t1, 8(sp)
        sd      t2, 16(sp)
        csrr    t0, mcause
        la      t1, mat_xmfrm_trap_cause
        sd      t0, 0(t1)
        csrr    t0, mepc
        addi    t0, t0, 4
        csrw    mepc, t0
        la      t0, mat_xmfrm_trap_count
        ld      t1, 0(t0)
        addi    t1, t1, 1
        sd      t1, 0(t0)
        ld      t0, 0(sp)
        ld      t1, 8(sp)
        ld      t2, 16(sp)
        addi    sp, sp, 32
        mret


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
        SETCFG  mcfg4, MCFG_INT32

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
