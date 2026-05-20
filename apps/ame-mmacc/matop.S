// Zames v0.6.0 port of ame-mmacc. The original used the Xuantie 0.6 dialect
// where element width and signedness were encoded in the mnemonic
// (e.g. mlae8 / mmacc.w.b / mmaccu.w.b). Under Zames the mnemonics carry no
// width or signedness and every tile/accumulator must be programmed through
// msetcfg before use.

.text
.global matrix_init
.global matrix_mmacc0
.global matrix_mmacc1
.global matrix_mmacc2
.global matrix_mmacc3
.global matrix_mmacc4
.global matrix_restore
.global mat_dist

.section .data
.align 8
mat_na:
        .rept   8192            // 128 * 64 bytes, all -1 (0xff)
        .byte   -1
        .endr
mat_nb:
        .rept   8192            // 64 * 128 bytes, all -2 (0xfe)
        .byte   -2
        .endr
mat_dist:
        .rept   65536           // 4 * 128 * 128 words of 3
        .word   3
        .endr

// ---------------------------------------------------------------------------
// mcfg type_code constants (table_sel = 0)
//   int8   = 0x02
//   uint8  = 0x03
//   int32  = 0x04
// ---------------------------------------------------------------------------
.equ MCFG_INT8,  0x02
.equ MCFG_UINT8, 0x03
.equ MCFG_INT32, 0x04

.section .text

matrix_restore:
        li      a0, 0
        la      a1, mat_dist
        li      a2, 3
        li      a3, 16384
restore_flag:
        sw      a2, 0(a1)
        sw      a2, 4(a1)
        sw      a2, 8(a1)
        sw      a2, 12(a1)
        addi    a1, a1, 16
        addi    a3, a3, -1
        bnez    a3, restore_flag
        fence   rw, rw
        ret

matrix_init:
        // Enable matrix context (mstatus.MS=01) plus FS and VS for safety.
        // bit 25 = MS[0], bit 13 = FS[0], bit 9 = VS[0]
        lui     a0, 8194
        addiw   a0, a0, 512
        csrs    mstatus, a0

        // Tile shape: 128 x 64 x 128.
        li      a0, 128
        msettilem       a0
        msettilen       a0
        li      a0, 64
        msettilek       a0

        msettilemi      128
        msettileni      128
        msettileki      64
        ret

// ---------------------------------------------------------------------------
// Helper macro: program type_code into mcfgN via a scratch GPR.
//   cfg_imm uses msetcfg <mcfgN>, rs1 — the destination is the 3-bit selector
//   (0..7) reusing tr0..acc3 numbering.
// ---------------------------------------------------------------------------
.macro SETCFG mcfg_sel, type_code
        li      t6, \type_code
        msetcfg \mcfg_sel, t6
.endm

// ---------------------------------------------------------------------------
// matrix_mmacc0 .. matrix_mmacc3 exercise the four sign combinations of the
// 8-bit -> 32-bit multiply-accumulate, each writing into a different acc tile
// region of mat_dist. The expected results match the original Xuantie test:
//   signed   signed   : (-1) * (-2) * 64 + 3 = 131
//   unsigned unsigned : 0xff * 0xfe * 64 + 3 = 4145283
//   signed   unsigned : (-1) * 0xfe * 64 + 3 = -16253
//   unsigned signed   : 0xff * (-2) * 64 + 3 = -32637
// ---------------------------------------------------------------------------

// Common epilogue: release/acquire then jump to matrix_check.
.macro FINISH_AND_CHECK
        mrelease        sync1
        li              a0, 1
        macquire        sync1, a0
        j               matrix_check
.endm

matrix_mmacc0:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 512

        msyncregreset   sync1

        // Op 1: acc0 = tr0(s8) * tr1(s8)
        SETCFG  mcfg0, MCFG_INT8       // tr0 = int8
        SETCFG  mcfg1, MCFG_INT8       // tr1 = int8
        SETCFG  mcfg4, MCFG_INT32      // acc0 = int32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc0, (t4), t5
        // mmacc md, ms2, ms1   semantics: md += ms1 * ms2
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        // Op 2: acc1 = tr1(u8) * tr2(u8)
        li      a7, 65536
        add     t4, t4, a7
        SETCFG  mcfg1, MCFG_UINT8      // override tr1 sign
        SETCFG  mcfg2, MCFG_UINT8
        SETCFG  mcfg5, MCFG_INT32
        mla     tr1, (t0), t1
        mlb     tr2, (t2), t3
        mlc     acc1, (t4), t5
        mmacc   acc1, tr2, tr1
        msc     acc1, (t4), t5

        // Op 3: acc2 = tr2(s8) * tr3(u8)
        add     t4, t4, a7
        SETCFG  mcfg2, MCFG_INT8       // tr2 signed
        SETCFG  mcfg3, MCFG_UINT8      // tr3 unsigned
        SETCFG  mcfg6, MCFG_INT32
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mlc     acc2, (t4), t5
        mmacc   acc2, tr3, tr2
        msc     acc2, (t4), t5

        // Op 4: acc3 = tr3(u8) * tr0(s8)
        add     t4, t4, a7
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr3, (t0), t1
        mlb     tr0, (t2), t3
        mlc     acc3, (t4), t5
        mmacc   acc3, tr0, tr3
        msc     acc3, (t4), t5

        FINISH_AND_CHECK

matrix_mmacc1:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 512

        msyncregreset   sync1

        // Op 1: acc1 = tr0(s8) * tr1(s8)
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg5, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc1, (t4), t5
        mmacc   acc1, tr1, tr0
        msc     acc1, (t4), t5

        // Op 2: acc2 = tr1(u8) * tr2(u8)
        li      a7, 65536
        add     t4, t4, a7
        SETCFG  mcfg1, MCFG_UINT8
        SETCFG  mcfg2, MCFG_UINT8
        SETCFG  mcfg6, MCFG_INT32
        mla     tr1, (t0), t1
        mlb     tr2, (t2), t3
        mlc     acc2, (t4), t5
        mmacc   acc2, tr2, tr1
        msc     acc2, (t4), t5

        // Op 3: acc3 = tr2(s8) * tr3(u8)
        add     t4, t4, a7
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mlc     acc3, (t4), t5
        mmacc   acc3, tr3, tr2
        msc     acc3, (t4), t5

        // Op 4: acc0 = tr3(u8) * tr0(s8)
        add     t4, t4, a7
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr3, (t0), t1
        mlb     tr0, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr0, tr3
        msc     acc0, (t4), t5

        FINISH_AND_CHECK

matrix_mmacc2:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 512

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg6, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc2, (t4), t5
        mmacc   acc2, tr1, tr0
        msc     acc2, (t4), t5

        li      a7, 65536
        add     t4, t4, a7
        SETCFG  mcfg1, MCFG_UINT8
        SETCFG  mcfg2, MCFG_UINT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr1, (t0), t1
        mlb     tr2, (t2), t3
        mlc     acc3, (t4), t5
        mmacc   acc3, tr2, tr1
        msc     acc3, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr3, tr2
        msc     acc0, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg5, MCFG_INT32
        mla     tr3, (t0), t1
        mlb     tr0, (t2), t3
        mlc     acc1, (t4), t5
        mmacc   acc1, tr0, tr3
        msc     acc1, (t4), t5

        FINISH_AND_CHECK

matrix_mmacc3:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 512

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc3, (t4), t5
        mmacc   acc3, tr1, tr0
        msc     acc3, (t4), t5

        li      a7, 65536
        add     t4, t4, a7
        SETCFG  mcfg1, MCFG_UINT8
        SETCFG  mcfg2, MCFG_UINT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr1, (t0), t1
        mlb     tr2, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr2, tr1
        msc     acc0, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg5, MCFG_INT32
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mlc     acc1, (t4), t5
        mmacc   acc1, tr3, tr2
        msc     acc1, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg6, MCFG_INT32
        mla     tr3, (t0), t1
        mlb     tr0, (t2), t3
        mlc     acc2, (t4), t5
        mmacc   acc2, tr0, tr3
        msc     acc2, (t4), t5

        FINISH_AND_CHECK

// matrix_mmacc4: doubled accumulate in a single tile region.
// Expected: (-1)*(-2)*64*2 + 3 = 259, mirroring the original test.
matrix_mmacc4:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 512

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr1, tr0
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence   rw, rw

        la      a1, mat_dist
        li      a2, 259
        li      a3, 16384
        li      a0, 0
check_loop4:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop4
        j       check_done

matrix_check:
        fence   rw, rw
        la      a1, mat_dist
        li      a2, 131
        li      a3, 16384
        li      a0, 0
check_loop0:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop0

        li      a2, 4145283
        li      a3, 16384
check_loop1:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop1

        li      a2, -16253
        li      a3, 16384
check_loop2:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop2

        li      a2, -32637
        li      a3, 16384
check_loop3:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop3

        j       check_done

check_fail:
        li      a0, 1
check_done:
        ret
