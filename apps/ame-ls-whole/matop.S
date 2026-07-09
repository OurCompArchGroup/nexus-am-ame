// Zames v0.6.0 whole-register load/store test (ref: ame-ls-word).
//
// The whole-register variants are the same A/B/C load/store instructions with
// the whole mode bit (func4[3]) set to 1; the assembly uses a .whole suffix:
//
//   mla.whole md,  (rs1)   # load whole A register  (md  = tile reg tr0-tr3)
//   mlb.whole md,  (rs1)   # load whole B register  (md  = tile reg tr0-tr3)
//   mlc.whole md,  (rs1)   # load whole C register  (md  = acc  reg acc0-acc3)
//   msa.whole ms3, (rs1)   # store whole A register (ms3 = tile reg tr0-tr3)
//   msb.whole ms3, (rs1)   # store whole B register (ms3 = tile reg tr0-tr3)
//   msc.whole ms3, (rs1)   # store whole C register (ms3 = acc  reg acc0-acc3)
//
// This is the PROPOSAL-12 version. In proposal-12 the whole form keeps the rs2
// operand ("mnem $md, (rs1), rs2") and does not state that it is unused, so rs2
// is the inter-row byte stride, exactly as for the non-whole load/store. The
// current toolchain matches this: LLVM (llvm-project-ame-new) encodes rs2 in all
// six whole forms, and NEMU-matrix walks ROWNUM rows advancing each by rs2. To
// transfer the full register as one contiguous image we therefore pass the
// natural per-row byte width in rs2 (passing x0 would collapse every row onto
// the base address, storing only one row).
//
// The per-row width is derived from the size CSRs so the test is config-agnostic:
//   * tile row bytes = trlenb                         (CSR 0xcc2)
//   * acc  row bytes = alenb / (tlenb / trlenb)       = alenb / ROWNUM
//
// (Proposal-14 later redefines whole accesses as stride-less contiguous images
// with rs2 forced to x0; that variant is left for future work.)
//
// Whole-register accesses otherwise:
//   * copy the COMPLETE architectural register contents (raw state transfer,
//     no element-type interpretation) - intended for context save/restore;
//   * do NOT depend on the programmed mtilem / mtilek / mtilen values;
//   * whole + transpose is reserved.
//
// The size of a whole transfer is fixed by the implementation:
//   * tile register  = tlenb bytes   (CSR 0xcc1)
//   * acc  register  = alenb bytes   (CSR 0xcc3)
//
// Test strategy per matrix class: load the whole register from a patterned
// source buffer, store it back to a sentinel-filled destination buffer, then
// compare byte-for-byte over the architectural register size. The independence
// test additionally shrinks mtilem/k/n between the whole load and whole store
// to prove the store still transfers the entire register.

.text
.global matrix_init_128
.global matrix_init_64
.global matrix_whole_c
.global matrix_whole_a
.global matrix_whole_b
.global matrix_whole_indep_128
.global matrix_whole_indep_64

.global mat_src
.global mat_dist

.section .rodata

.section .data
.align 8

// Source image: filled with a byte pattern by the C driver before the tests.
mat_src:
        .rept   65536
        .byte   0
        .endr

// Destination image: filled with a sentinel by the C driver before each test.
mat_dist:
        .rept   65536
        .byte   0
        .endr

.section .text

// ---------------------------------------------------------------------------
// mcfg type_code constants (table_sel = 0)
//   int8  = 0x02
//   int32 = 0x04
// ---------------------------------------------------------------------------
.equ MCFG_INT8,  0x02
.equ MCFG_INT32, 0x04

// Whole-register byte-size CSRs.
.equ CSR_TLENB,  0xcc1         // tile register size in bytes  (TLEN/8)
.equ CSR_TRLENB, 0xcc2         // tile register row size       (TRLEN/8)
.equ CSR_ALENB,  0xcc3         // acc  register size in bytes  (ALEN/8)

// Helper macro: program type_code into mcfgN via scratch GPR t6.
.macro SETCFG mcfg_sel, type_code
        li      t6, \type_code
        msetcfg \mcfg_sel, t6
.endm

// Compute the whole-register row stride (bytes) into t2.
//   ACC_STRIDE -> alenb / (tlenb / trlenb)
//   TILE_STRIDE -> trlenb
.macro ACC_STRIDE
        csrr    t3, CSR_TLENB
        csrr    t4, CSR_TRLENB
        divu    t5, t3, t4             // ROWNUM = tlenb / trlenb
        csrr    t3, CSR_ALENB
        divu    t2, t3, t5             // acc row bytes = alenb / ROWNUM
.endm

.macro TILE_STRIDE
        csrr    t2, CSR_TRLENB         // tile row bytes = trlenb
.endm

matrix_init_128:
        // Enable matrix context (mstatus.MS) plus FS/VS for safety.
        lui a0, 8194
        addiw a0, a0, 512
        csrs mstatus, a0

        li a0, 128
        msettilem       a0
        msettilen       a0
        li a0, 64
        msettilek       a0

        msettilemi     128
        msettileni     128
        msettileki     64

        fence iorw,iorw

        // Program the registers touched by the whole tests. Whole transfers do
        // not interpret element type, but the register class must still be
        // valid and mcfg is programmed for a well-defined architectural state.
        SETCFG  mcfg0, MCFG_INT8       // tr0  (A / B whole)
        SETCFG  mcfg1, MCFG_INT8       // tr1  (B whole)
        SETCFG  mcfg4, MCFG_INT32      // acc0 (C whole)

        ret

matrix_init_64:
        // Enable matrix context (mstatus.MS) plus FS/VS for safety.
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

        fence iorw,iorw

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32

        ret

// ---------------------------------------------------------------------------
// C matrix whole register: mlc.whole / msc.whole on acc0. Size = alenb.
// ---------------------------------------------------------------------------
matrix_whole_c:
        la      t0, mat_src
        la      t1, mat_dist
        ACC_STRIDE                     // t2 = acc row stride
        fence iorw,iorw
        msyncregreset   sync1
        mlc.whole       acc0, (t0), t2 // load whole acc0  <- src
        msc.whole       acc0, (t1), t2 // store whole acc0 -> dist
        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence iorw,iorw

        li      a0, 0
        csrr    a3, CSR_ALENB          // compare alenb bytes ...
        srli    a3, a3, 2              // ... as words
        la      a1, mat_dist
        la      a2, mat_src
        j       cmp_loop

// ---------------------------------------------------------------------------
// A matrix whole register: mla.whole / msa.whole on tile reg tr0. Size = tlenb.
// ---------------------------------------------------------------------------
matrix_whole_a:
        la      t0, mat_src
        la      t1, mat_dist
        TILE_STRIDE                    // t2 = tile row stride
        fence iorw,iorw
        msyncregreset   sync1
        mla.whole       tr0, (t0), t2  // load whole tr0  <- src
        msa.whole       tr0, (t1), t2  // store whole tr0 -> dist
        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence iorw,iorw

        li      a0, 0
        csrr    a3, CSR_TLENB          // compare tlenb bytes
        srli    a3, a3, 2
        la      a1, mat_dist
        la      a2, mat_src
        j       cmp_loop

// ---------------------------------------------------------------------------
// B matrix whole register: mlb.whole / msb.whole on tile reg tr1. Size = tlenb.
// ---------------------------------------------------------------------------
matrix_whole_b:
        la      t0, mat_src
        la      t1, mat_dist
        TILE_STRIDE                    // t2 = tile row stride
        fence iorw,iorw
        msyncregreset   sync1
        mlb.whole       tr1, (t0), t2  // load whole tr1  <- src
        msb.whole       tr1, (t1), t2  // store whole tr1 -> dist
        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence iorw,iorw

        li      a0, 0
        csrr    a3, CSR_TLENB
        srli    a3, a3, 2
        la      a1, mat_dist
        la      a2, mat_src
        j       cmp_loop

// ---------------------------------------------------------------------------
// Independence from tile config: load the whole acc0, then shrink
// mtilem/mtilek/mtilen before the whole store. A correct whole store ignores
// the tile settings and still writes the entire alenb-byte register image, so
// the full compare must pass.
// ---------------------------------------------------------------------------
matrix_whole_indep_128:
        la      t0, mat_src
        la      t1, mat_dist
        ACC_STRIDE                     // t2 = acc row stride (from fixed size CSRs)
        fence iorw,iorw
        msyncregreset   sync1
        mlc.whole       acc0, (t0), t2 // load whole acc0 with full tiles

        // Shrink the active tile shape; whole store must not be affected. The
        // stride in t2 comes from the fixed tlenb/trlenb/alenb CSRs, not mtile*,
        // so it is unchanged by the shrink.
        li      a0, 8
        msettilem       a0
        msettilen       a0
        li      a0, 4
        msettilek       a0

        msc.whole       acc0, (t1), t2 // whole store ignores tile shape

        // Restore the full tile shape for any later use.
        li      a0, 128
        msettilem       a0
        msettilen       a0
        li      a0, 64
        msettilek       a0

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence iorw,iorw

        li      a0, 0
        csrr    a3, CSR_ALENB          // entire acc image must have been written
        srli    a3, a3, 2
        la      a1, mat_dist
        la      a2, mat_src
        j       cmp_loop

matrix_whole_indep_64:
        la      t0, mat_src
        la      t1, mat_dist
        ACC_STRIDE
        fence iorw,iorw
        msyncregreset   sync1
        mlc.whole       acc0, (t0), t2

        li      a0, 8
        msettilem       a0
        msettilen       a0
        li      a0, 4
        msettilek       a0

        msc.whole       acc0, (t1), t2

        li      a0, 64
        msettilem       a0
        msettilen       a0
        msettilek       a0

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence iorw,iorw

        li      a0, 0
        csrr    a3, CSR_ALENB
        srli    a3, a3, 2
        la      a1, mat_dist
        la      a2, mat_src
        j       cmp_loop

// ---------------------------------------------------------------------------
// Shared word-wise comparison: a1=dist, a2=src, a3=word count, a0=return code.
// ---------------------------------------------------------------------------
cmp_loop:
        lw      a5, 0(a1)
        lw      a6, 0(a2)
        bne     a5, a6, check_fail
        addi    a1, a1, 4
        addi    a2, a2, 4
        addi    a3, a3, -1
        bnez    a3, cmp_loop
        j       check_done

check_fail:
        li      a0, 1                  // error: set return value to 1

check_done:
        ret
