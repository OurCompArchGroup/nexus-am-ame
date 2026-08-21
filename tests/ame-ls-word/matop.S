// Zames v0.6.0 port of ame-ls_word.
// mlc/msc   = load/store accumulator (non-transposed)
// mlct/msct = load/store accumulator (transposed)
// All acc registers must be programmed via msetcfg before first use.
// Only acc registers (int32) are used here; no tile registers accessed.

.text
.global matrix_init_128
.global matrix_init_64
.global matrix_ls_128
.global matrix_ls_64
.global matrix_ltst_128
.global matrix_ltst_64
.global matrix_lts_128
.global matrix_lts_64
.global matrix_lst_128
.global matrix_lst_64

.global mat_dist
.global mat_a_t

.section .rodata


.section .data
.align 8

mat_a:
    .rept   16384
    .word   0
    .endr

mat_a_t:
    .rept 16384
    .word 0
    .endr

mat_dist:
    .rept    65536
    .word   5
    .endr

.section .text

// ---------------------------------------------------------------------------
// mcfg type_code constants (table_sel = 0)
//   int32 = 0x04
// ---------------------------------------------------------------------------
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

        li a0, 128
        msettilem       a0
        msettilen       a0
        li a0, 64
        msettilek       a0

        msettilemi     128
        msettileni     128
        msettileki     64

        fence iorw,iorw
        la   a4,mat_a
        li   a5,2
        sw   a5,4(a4)

        la  a4,mat_a_t
        sw a5, 512(a4)

        fence iorw,iorw

        // Program acc0..acc3 as int32 (mcfg4..mcfg7)
        SETCFG  mcfg4, MCFG_INT32
        SETCFG  mcfg5, MCFG_INT32
        SETCFG  mcfg6, MCFG_INT32
        SETCFG  mcfg7, MCFG_INT32

        ret

matrix_init_64:
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
        la   a4,mat_a
        li   a5,2
        sw   a5,4(a4)

        la   a4,mat_a_t
        sw   a5,256(a4)

        fence iorw,iorw

        SETCFG  mcfg4, MCFG_INT32
        SETCFG  mcfg5, MCFG_INT32
        SETCFG  mcfg6, MCFG_INT32
        SETCFG  mcfg7, MCFG_INT32

        ret

matrix_ls_128:
        la      t0, mat_a
        li  t1,512
        la      t4,mat_dist
        li      t5,512
        li      a7,65536
        fence iorw,iorw
        msyncregreset   sync1
        mlc     acc0, (t0), t1
        msc     acc0, (t4), t5

        add    t4, t4, a7
        mlc     acc1, (t0), t1
        msc     acc1, (t4), t5

        add    t4, t4, a7
        mlc     acc2, (t0), t1
        msc     acc2, (t4), t5

        add    t4, t4, a7
        mlc     acc3, (t0), t1
        msc     acc3, (t4), t5

        mrelease        sync1
        li a0, 1
        macquire        sync1, a0
        fence iorw,iorw
        li      a0,0
        la      a1, mat_dist
        la      a2, mat_a
        li      a3, 16384
        li      a4,4
check_ls:
        lw      a5, 0(a1)
        lw      a6, 0(a2)
        bne     a5, a6, check_fail
        addi    a1, a1, 4
        addi    a2,a2,4
        addi    a3, a3, -1
        bnez    a3, check_ls
        addi    a4,a4,-1

        li      a3,16384
        la      a2,mat_a
        bnez   a4,check_ls
        j check_done

matrix_ltst_128:
        la      t0, mat_a
        li  t1,512
        la      t4,mat_dist
        li      t5,512
        li      a7,65536

        msyncregreset   sync1
        mlct    acc0, (t0), t1
        msct    acc0, (t4), t5

        add    t4, t4, a7
        mlct    acc1, (t0), t1
        msct    acc1, (t4), t5

        add    t4, t4, a7
        mlct    acc2, (t0), t1
        msct    acc2, (t4), t5

        add    t4, t4, a7
        mlct    acc3, (t0), t1
        msct    acc3, (t4), t5

        mrelease        sync1
        li a0, 1
        macquire        sync1, a0
        fence iorw,iorw
        li      a0,0
        la      a1, mat_dist
        la      a2, mat_a
        li      a3, 16384
        li      a4,4
check_ltst:
        lw      a5, 0(a1)
        lw      a6,0(a2)
        bne     a5, a6, check_fail
        addi    a1, a1, 4
        addi    a2,a2,4
        addi    a3, a3, -1
        bnez    a3, check_ltst
        addi    a4,a4,-1

        li      a3,16384
        la      a2,mat_a
        bnez   a4,check_ltst
        j check_done

matrix_lts_128:
        la      t0, mat_a
        li      t1,512
        la      t4,mat_dist
        li      t5,512
        li      a7,65536

        msyncregreset   sync1
        mlct    acc0, (t0), t1
        msc     acc0, (t4), t5

        add    t4, t4, a7
        mlct    acc1, (t0), t1
        msc     acc1, (t4), t5

        add    t4, t4, a7
        mlct    acc2, (t0), t1
        msc     acc2, (t4), t5

        add    t4, t4, a7
        mlct    acc3, (t0), t1
        msc     acc3, (t4), t5

        mrelease        sync1
        li a0, 1
        macquire        sync1, a0
        fence iorw,iorw
        li      a0,0
        la      a1, mat_dist
        la      a2, mat_a_t
        li      a3, 16384
        li      a4, 4
check_lts:
        lw      a5, 0(a1)
        lw      a6, 0(a2)
        bne     a5, a6, check_fail
        addi    a1, a1, 4
        addi    a2, a2,4
        addi    a3, a3, -1
        bnez    a3, check_lts

        addi    a4,a4,-1
        li      a3,16384
        la      a2,mat_a_t
        bnez   a4,check_lts
        j      check_done


matrix_lst_128:
        la      t0, mat_a
        li  t1,512
        la      t4,mat_dist
        li      t5,512
        li      a7,65536

        msyncregreset   sync1
        mlc     acc0, (t0), t1
        msct    acc0, (t4), t5

        add    t4, t4, a7
        mlc     acc1, (t0), t1
        msct    acc1, (t4), t5

        add    t4, t4, a7
        mlc     acc2, (t0), t1
        msct    acc2, (t4), t5

        add    t4, t4, a7
        mlc     acc3, (t0), t1
        msct    acc3, (t4), t5

        mrelease        sync1
        li a0, 1
        macquire        sync1, a0
        fence iorw,iorw
        li      a0,0
        la      a1, mat_dist
        la      a2, mat_a_t
        li      a3, 16384
        li      a4,4
check_lst:
        lw      a5, 0(a1)
        lw      a6, 0(a2)
        bne     a5, a6, check_fail
        addi    a1, a1, 4
        addi    a2,a2,4
        addi    a3, a3, -1
        bnez    a3, check_lst
        addi    a4,a4,-1

        li      a3,16384
        la      a2,mat_a_t
        bnez   a4,check_lst
        j      check_done

matrix_ls_64:
        la      t0, mat_a
        li      t1,256
        la      t4,mat_dist
        li      t5,256
        li      a7,16384
        fence iorw,iorw
        msyncregreset   sync1
        mlc     acc0, (t0), t1
        msc     acc0, (t4), t5

        add     t4, t4, a7
        mlc     acc1, (t0), t1
        msc     acc1, (t4), t5

        add     t4, t4, a7
        mlc     acc2, (t0), t1
        msc     acc2, (t4), t5

        add     t4, t4, a7
        mlc     acc3, (t0), t1
        msc     acc3, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence iorw,iorw
        li      a0,0
        la      a1, mat_dist
        la      a2, mat_a
        li      a3, 4096
        li      a4,4
check_ls_64:
        lw      a5, 0(a1)
        lw      a6, 0(a2)
        bne     a5, a6, check_fail
        addi    a1, a1, 4
        addi    a2, a2, 4
        addi    a3, a3, -1
        bnez    a3, check_ls_64
        addi    a4, a4, -1

        li      a3,4096
        la      a2,mat_a
        bnez    a4,check_ls_64
        j       check_done

matrix_ltst_64:
        la      t0, mat_a
        li      t1,256
        la      t4,mat_dist
        li      t5,256
        li      a7,16384

        msyncregreset   sync1
        mlct    acc0, (t0), t1
        msct    acc0, (t4), t5

        add     t4, t4, a7
        mlct    acc1, (t0), t1
        msct    acc1, (t4), t5

        add     t4, t4, a7
        mlct    acc2, (t0), t1
        msct    acc2, (t4), t5

        add     t4, t4, a7
        mlct    acc3, (t0), t1
        msct    acc3, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence iorw,iorw
        li      a0,0
        la      a1, mat_dist
        la      a2, mat_a
        li      a3, 4096
        li      a4,4
check_ltst_64:
        lw      a5, 0(a1)
        lw      a6,0(a2)
        bne     a5, a6, check_fail
        addi    a1, a1, 4
        addi    a2, a2, 4
        addi    a3, a3, -1
        bnez    a3, check_ltst_64
        addi    a4, a4, -1

        li      a3,4096
        la      a2,mat_a
        bnez    a4,check_ltst_64
        j       check_done

matrix_lts_64:
        la      t0, mat_a
        li      t1,256
        la      t4,mat_dist
        li      t5,256
        li      a7,16384

        msyncregreset   sync1
        mlct    acc0, (t0), t1
        msc     acc0, (t4), t5

        add     t4, t4, a7
        mlct    acc1, (t0), t1
        msc     acc1, (t4), t5

        add     t4, t4, a7
        mlct    acc2, (t0), t1
        msc     acc2, (t4), t5

        add     t4, t4, a7
        mlct    acc3, (t0), t1
        msc     acc3, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence iorw,iorw
        li      a0,0
        la      a1, mat_dist
        la      a2, mat_a_t
        li      a3, 4096
        li      a4, 4
check_lts_64:
        lw      a5, 0(a1)
        lw      a6, 0(a2)
        bne     a5, a6, check_fail
        addi    a1, a1, 4
        addi    a2, a2, 4
        addi    a3, a3, -1
        bnez    a3, check_lts_64

        addi    a4, a4, -1
        li      a3,4096
        la      a2,mat_a_t
        bnez    a4,check_lts_64
        j       check_done

matrix_lst_64:
        la      t0, mat_a
        li      t1,256
        la      t4,mat_dist
        li      t5,256
        li      a7,16384

        msyncregreset   sync1
        mlc     acc0, (t0), t1
        msct    acc0, (t4), t5

        add     t4, t4, a7
        mlc     acc1, (t0), t1
        msct    acc1, (t4), t5

        add     t4, t4, a7
        mlc     acc2, (t0), t1
        msct    acc2, (t4), t5

        add     t4, t4, a7
        mlc     acc3, (t0), t1
        msct    acc3, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence iorw,iorw
        li      a0,0
        la      a1, mat_dist
        la      a2, mat_a_t
        li      a3, 4096
        li      a4,4
check_lst_64:
        lw      a5, 0(a1)
        lw      a6, 0(a2)
        bne     a5, a6, check_fail
        addi    a1, a1, 4
        addi    a2, a2, 4
        addi    a3, a3, -1
        bnez    a3, check_lst_64
        addi    a4, a4, -1

        li      a3,4096
        la      a2,mat_a_t
        bnez    a4,check_lst_64
        j       check_done



check_fail:
        li      a0, 1         // error: set return value to 1

check_done:
        ret
// void* get_matrix_a(void)
get_matrix_a:
        la a0, mat_a
        ret
