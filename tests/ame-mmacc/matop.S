// Zames v0.6.0 port of ame-mmacc. The original used the Xuantie 0.6 dialect
// where element width and signedness were encoded in the mnemonic
// (e.g. mlae8 / mmacc.w.b / mmaccu.w.b). Under Zames the mnemonics carry no
// width or signedness and every tile/accumulator must be programmed through
// msetcfg before use.

.text
.global matrix_init_128
.global matrix_init_64
.global matrix_mmacc0_128
.global matrix_mmacc0_64
.global matrix_mmacc1_128
.global matrix_mmacc1_64
.global matrix_mmacc2_128
.global matrix_mmacc2_64
.global matrix_mmacc3_128
.global matrix_mmacc3_64
.global matrix_mmacc4_128
.global matrix_mmacc4_64
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

matrix_init_128:
        // Enable matrix context (mstatus.MS=01) plus FS and VS for safety.
        // bit 25 = MS[0], bit 13 = FS[0], bit 9 = VS[0]
        lui     a0, 8194
        addiw   a0, a0, 512
        csrs    mstatus, a0

        // Tile shape: 128 x 64 -> 128.
        li      a0, 128
        msettilem       a0
        msettilen       a0
        li      a0, 64
        msettilek       a0

        msettilemi      128
        msettileni      128
        msettileki      64
        ret

matrix_init_64:
        lui     a0, 8194
        addiw   a0, a0, 512
        csrs    mstatus, a0

        li      a0, 64
        msettilem       a0
        msettilen       a0
        msettilek       a0

        msettilemi      64
        msettileni      64
        msettileki      64
        ret

// ---------------------------------------------------------------------------
// Helper macro: program type_code into mcfgN via a scratch GPR.
//   cfg_imm uses msetcfg <mcfgN>, rs1 - the destination is the 3-bit selector
//   (0..7) reusing tr0..acc3 numbering.
// ---------------------------------------------------------------------------
.macro SETCFG mcfg_sel, type_code
        li      t6, \type_code
        msetcfg \mcfg_sel, t6
.endm

// ---------------------------------------------------------------------------
// matrix_mmacc0 .. matrix_mmacc3 exercise the four sign combinations of the
// 8-bit -> 32-bit multiply-accumulate. The arithmetic expectation is unchanged
// in 64-mode because K remains 64; only the logical C-tile footprint shrinks.
//   signed   signed   : (-1) * (-2) * 64 + 3 = 131
//   unsigned unsigned : 0xff * 0xfe * 64 + 3 = 4145283
//   signed   unsigned : (-1) * 0xfe * 64 + 3 = -16253
//   unsigned signed   : 0xff * (-2) * 64 + 3 = -32637
// ---------------------------------------------------------------------------

.macro FINISH_AND_CHECK target
        mrelease        sync1
        li              a0, 1
        macquire        sync1, a0
        j               \target
.endm

matrix_mmacc0_128:
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
        msc     acc0, (t4), t5

        li      a7, 65536
        add     t4, t4, a7
        SETCFG  mcfg1, MCFG_UINT8
        SETCFG  mcfg2, MCFG_UINT8
        SETCFG  mcfg5, MCFG_INT32
        mla     tr1, (t0), t1
        mlb     tr2, (t2), t3
        mlc     acc1, (t4), t5
        mmacc   acc1, tr2, tr1
        msc     acc1, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg6, MCFG_INT32
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mlc     acc2, (t4), t5
        mmacc   acc2, tr3, tr2
        msc     acc2, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr3, (t0), t1
        mlb     tr0, (t2), t3
        mlc     acc3, (t4), t5
        mmacc   acc3, tr0, tr3
        msc     acc3, (t4), t5

        FINISH_AND_CHECK matrix_check_128

matrix_mmacc1_128:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 512

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg5, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc1, (t4), t5
        mmacc   acc1, tr1, tr0
        msc     acc1, (t4), t5

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

        add     t4, t4, a7
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mlc     acc3, (t4), t5
        mmacc   acc3, tr3, tr2
        msc     acc3, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr3, (t0), t1
        mlb     tr0, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr0, tr3
        msc     acc0, (t4), t5

        FINISH_AND_CHECK matrix_check_128

matrix_mmacc2_128:
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

        FINISH_AND_CHECK matrix_check_128

matrix_mmacc3_128:
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

        FINISH_AND_CHECK matrix_check_128

matrix_mmacc4_128:
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
check_loop4_128:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop4_128
        j       check_done

matrix_mmacc0_64:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 256

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        li      a7, 16384
        add     t4, t4, a7
        SETCFG  mcfg1, MCFG_UINT8
        SETCFG  mcfg2, MCFG_UINT8
        SETCFG  mcfg5, MCFG_INT32
        mla     tr1, (t0), t1
        mlb     tr2, (t2), t3
        mlc     acc1, (t4), t5
        mmacc   acc1, tr2, tr1
        msc     acc1, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg6, MCFG_INT32
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mlc     acc2, (t4), t5
        mmacc   acc2, tr3, tr2
        msc     acc2, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr3, (t0), t1
        mlb     tr0, (t2), t3
        mlc     acc3, (t4), t5
        mmacc   acc3, tr0, tr3
        msc     acc3, (t4), t5

        FINISH_AND_CHECK matrix_check_64

matrix_mmacc1_64:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 256

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg5, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc1, (t4), t5
        mmacc   acc1, tr1, tr0
        msc     acc1, (t4), t5

        li      a7, 16384
        add     t4, t4, a7
        SETCFG  mcfg1, MCFG_UINT8
        SETCFG  mcfg2, MCFG_UINT8
        SETCFG  mcfg6, MCFG_INT32
        mla     tr1, (t0), t1
        mlb     tr2, (t2), t3
        mlc     acc2, (t4), t5
        mmacc   acc2, tr2, tr1
        msc     acc2, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr2, (t0), t1
        mlb     tr3, (t2), t3
        mlc     acc3, (t4), t5
        mmacc   acc3, tr3, tr2
        msc     acc3, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg3, MCFG_UINT8
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr3, (t0), t1
        mlb     tr0, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr0, tr3
        msc     acc0, (t4), t5

        FINISH_AND_CHECK matrix_check_64

matrix_mmacc2_64:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 256

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg6, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc2, (t4), t5
        mmacc   acc2, tr1, tr0
        msc     acc2, (t4), t5

        li      a7, 16384
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

        FINISH_AND_CHECK matrix_check_64

matrix_mmacc3_64:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 256

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc3, (t4), t5
        mmacc   acc3, tr1, tr0
        msc     acc3, (t4), t5

        li      a7, 16384
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

        FINISH_AND_CHECK matrix_check_64

matrix_mmacc4_64:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 256

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
        li      a3, 4096
        li      a0, 0
check_loop4_64:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop4_64
        j       check_done

matrix_check_128:
        fence   rw, rw
        la      a1, mat_dist
        li      a2, 131
        li      a3, 16384
        li      a0, 0
check_loop0_128:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop0_128

        li      a2, 4145283
        li      a3, 16384
check_loop1_128:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop1_128

        li      a2, -16253
        li      a3, 16384
check_loop2_128:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop2_128

        li      a2, -32637
        li      a3, 16384
check_loop3_128:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop3_128

        j       check_done

matrix_check_64:
        fence   rw, rw
        la      a1, mat_dist
        li      a2, 131
        li      a3, 4096
        li      a0, 0
check_loop0_64:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop0_64

        li      a2, 4145283
        li      a3, 4096
check_loop1_64:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop1_64

        li      a2, -16253
        li      a3, 4096
check_loop2_64:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop2_64

        li      a2, -32637
        li      a3, 4096
check_loop3_64:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop3_64

        j       check_done

check_fail:
        li      a0, 1
check_done:
        ret
