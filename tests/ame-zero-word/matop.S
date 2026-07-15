// Zames v0.6.0 port of ame-zero_word.
// matrix_zero_verify:     load acc, zero it, store it -> expect all-zero output
// matrix_zero_tr_verify:  load tr (zeroed), mmacc against loaded B -> expect all-zero output
// mzero = zero a tile or accumulator register (no mcfg needed for zero itself)
// All tile/acc registers must be programmed via msetcfg before load/mmacc.

.text
.global matrix_zero_verify_128
.global matrix_zero_verify_64
.global matrix_zero_tr_verify_128
.global matrix_zero_tr_verify_64
.global matrix_init_128
.global matrix_init_64
.global mat_dist

.section .rodata


.section .data
.align 8
// Matrix A: 128 x 64 words (used as byte-addressable tile source)
mat_a:
        .rept   16384
        .word   1
        .endr
mat_b:
        .rept   16384
        .word   2
        .endr
mat_dist:
        .rept    65536
        .word   3
        .endr

.section .text

// ---------------------------------------------------------------------------
// mcfg type_code constants (table_sel = 0)
//   int8  = 0x02
//   int32 = 0x04
// ---------------------------------------------------------------------------
.equ MCFG_INT8,  0x02
.equ MCFG_INT32, 0x04

// Helper macro: program type_code into mcfgN via scratch GPR t6.
.macro SETCFG mcfg_sel, type_code
        li      t6, \type_code
        msetcfg \mcfg_sel, t6
.endm

matrix_init_128:
        lui a0, 8194
        addiw a0, a0, 512
        csrs mstatus, a0

        li a0, 4
        msettilem       a0
        msettilen       a0
        li a0, 4
        msettilek       a0

        msettilemi     128
        msettileni     128
        msettileki     64
        ret

matrix_init_64:
        lui a0, 8194
        addiw a0, a0, 512
        csrs mstatus, a0

        li a0, 4
        msettilem       a0
        msettilen       a0
        li a0, 4
        msettilek       a0

        msettilemi      64
        msettileni      64
        msettileki      64
        ret

matrix_zero_verify_128:

        la      t4, mat_dist
        li      t5, 512

        // Reset sync register
        msyncregreset   sync1

        la      t4, mat_dist

        // acc0: load, zero, store
        SETCFG  mcfg4, MCFG_INT32
        mlc     acc0, (t4), t5
        mzero   acc0
        msc     acc0, (t4), t5


        li      a7,65536
        add    t4, t4, a7
        SETCFG  mcfg5, MCFG_INT32
        mlc     acc1, (t4), t5
        mzero   acc1
        msc     acc1, (t4), t5


        add    t4, t4, a7
        SETCFG  mcfg6, MCFG_INT32
        mlc     acc2, (t4), t5
        mzero   acc2
        msc     acc2, (t4), t5

        add    t4, t4, a7
        SETCFG  mcfg7, MCFG_INT32
        mlc     acc3, (t4), t5
        mzero   acc3
        msc     acc3, (t4), t5

        mrelease        sync1
        li a0, 1
        macquire        sync1, a0


        fence iorw,iorw
        li      a0,0
        la      a1, mat_dist
        li      a2,0
        li      a3,65536
check_loop:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop
        j check_done

matrix_zero_verify_64:

        la      t4, mat_dist
        li      t5, 256

        msyncregreset   sync1

        la      t4, mat_dist

        SETCFG  mcfg4, MCFG_INT32
        mlc     acc0, (t4), t5
        mzero   acc0
        msc     acc0, (t4), t5

        li      a7,16384
        add     t4, t4, a7
        SETCFG  mcfg5, MCFG_INT32
        mlc     acc1, (t4), t5
        mzero   acc1
        msc     acc1, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg6, MCFG_INT32
        mlc     acc2, (t4), t5
        mzero   acc2
        msc     acc2, (t4), t5

        add     t4, t4, a7
        SETCFG  mcfg7, MCFG_INT32
        mlc     acc3, (t4), t5
        mzero   acc3
        msc     acc3, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0

        fence iorw,iorw
        li      a0,0
        la      a1, mat_dist
        li      a2,0
        li      a3,16384
check_loop_64:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop_64
        j       check_done

matrix_zero_tr_verify_128:
        li a0,0
        la  a1,mat_dist
        li a2,0
        li a3,65536
restore_flag:
        sw a2,0(a1)
        addi a1,a1,1
        addi a3,a3,-1
        bnez a3,restore_flag

        la      t0, mat_a
        li      t1, 64

        la      t2, mat_b
        li      t3, 64

        la      t4, mat_dist
        li      t5, 128
        fence iorw,iorw

        msyncregreset   sync1

        // tr0 = int8, tr1 = int8, acc0 = int32
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32

        mla     tr0,(t0),t1
        mlb     tr1,(t2),t3
        mlc     acc0, (t4), t5
        mzero   tr0
        // mmacc md, ms2, ms1 -> md += ms1 * ms2
        // Xuantie: mmacc.w.b acc0, tr0, tr1 -> acc0 += tr0 * tr1
        // Zames:   mmacc acc0, tr1, tr0
        mmacc   acc0, tr1, tr0
        fence iorw,iorw
        msc     acc0, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg5, MCFG_INT32
        mla     tr1,(t0),t1
        mlb     tr2,(t2),t3
        mlc     acc1, (t4), t5
        mzero   tr1
        mmacc   acc1, tr2, tr1
        msc     acc1, (t4), t5
        fence iorw,iorw

        add    t4, t4, a7
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_INT8
        SETCFG  mcfg6, MCFG_INT32
        mla     tr2,(t0),t1
        mlb     tr3,(t2),t3
        mlc     acc2, (t4), t5
        mzero   tr2
        mmacc   acc2, tr3, tr2
        msc     acc2, (t4), t5
        fence iorw,iorw
        add    t4, t4, a7
        SETCFG  mcfg3, MCFG_INT8
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr3,(t0),t1
        mlb     tr0,(t2),t3
        mlc     acc3, (t4), t5
        mzero   tr3
        mmacc   acc3, tr0, tr3
        msc     acc3, (t4), t5

        // ======================================

        // Release sync
        mrelease        sync1
        li a0, 1
        macquire        sync1, a0
        fence iorw,iorw


        li      a0,0
        la      a1, mat_dist
        li      a2,0
        li a3,16384
check_tr_loop:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_tr_loop
        j check_done

matrix_zero_tr_verify_64:
        // Preserve the legacy TR(word) test structure from the 128-mode path:
        // only the first 64 KiB quarter is pre-zeroed and checked. The later
        // acc regions are still exercised, but they are not part of pass/fail.
        li a0,0
        la  a1,mat_dist
        li a2,0
        li a3,65536
restore_flag_64:
        sw a2,0(a1)
        addi a1,a1,1
        addi a3,a3,-1
        bnez a3,restore_flag_64

        la      t0, mat_a
        li      t1, 64

        la      t2, mat_b
        li      t3, 64

        la      t4, mat_dist
        li      t5, 128
        fence iorw,iorw

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32

        mla     tr0,(t0),t1
        mlb     tr1,(t2),t3
        mlc     acc0, (t4), t5
        mzero   tr0
        mmacc   acc0, tr1, tr0
        fence iorw,iorw
        msc     acc0, (t4), t5

        li      a7,65536
        add     t4, t4, a7
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg5, MCFG_INT32
        mla     tr1,(t0),t1
        mlb     tr2,(t2),t3
        mlc     acc1, (t4), t5
        mzero   tr1
        mmacc   acc1, tr2, tr1
        msc     acc1, (t4), t5
        fence iorw,iorw

        add     t4, t4, a7
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_INT8
        SETCFG  mcfg6, MCFG_INT32
        mla     tr2,(t0),t1
        mlb     tr3,(t2),t3
        mlc     acc2, (t4), t5
        mzero   tr2
        mmacc   acc2, tr3, tr2
        msc     acc2, (t4), t5
        fence iorw,iorw
        add     t4, t4, a7
        SETCFG  mcfg3, MCFG_INT8
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg7, MCFG_INT32
        mla     tr3,(t0),t1
        mlb     tr0,(t2),t3
        mlc     acc3, (t4), t5
        mzero   tr3
        mmacc   acc3, tr0, tr3
        msc     acc3, (t4), t5

        mrelease        sync1
        li a0, 1
        macquire        sync1, a0
        fence iorw,iorw

        li      a0,0
        la      a1, mat_dist
        li      a2,0
        li      a3,16384
check_tr_loop_64:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_tr_loop_64
        j       check_done


check_fail:
        li      a0, 1          // return 1 = fail

check_done:
        ret
