// Zames v0.6.0 port of ame-ls-ab.
// mla/mlat = load A tile (normal / transposed)
// mlb/mlbt = load B tile (normal / transposed)
// mlc/msc  = load/store accumulator
// mzero    = zero a tile or accumulator register
// mmacc md, ms2, ms1  -->  md += ms1 * ms2
// All tile/acc registers must be programmed via msetcfg before first use.

.text
.global matrix_init
.global restore_mat
.global matrix_ab
.global matrix_atb
.global matrix_abt
.global matrix_atbt

.global mat_dist
.global mat_a
.global mat_b

.section .rodata

.section .data
.align 8
mat_a:
        .rept   8192           // 128 * 64 = 8192 bytes, all 1
        .byte   1
        .endr


mat_b:
        .rept   8192           // 64 * 128 = 8192 bytes, all 2
        .byte   2
        .endr

mat_dist:
        .rept   65536
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

restore_mat:
        la a4,mat_a
        li  a5,1
        li  a6,8192
restore_mat_a_flag:
        sb  a5,0(a4)
        addi a4,a4,1
        addi a6,a6,-1
        bnez a6, restore_mat_a_flag

        la a4,mat_b
        li  a5,2
        li  a6,8192
restore_mat_b_flag:
        sb  a5,0(a4)
        addi a4,a4,1
        addi a6,a6,-1
        bnez a6, restore_mat_b_flag
        ret

matrix_init:
        // Enable matrix context (mstatus.MS=01) plus FS and VS for safety.
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

        // Program types for all tile/acc registers used across all functions.
        // tr0..tr3 = int8 (mcfg0..mcfg3), acc0..acc3 = int32 (mcfg4..mcfg7)
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg3, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        SETCFG  mcfg5, MCFG_INT32
        SETCFG  mcfg6, MCFG_INT32
        SETCFG  mcfg7, MCFG_INT32

        ret


matrix_ab:
        la a4,mat_a
        li  a5,2
        sb  a5,1(a4)

        la      t0, mat_a
        li      t1, 64

        la      t2, mat_b
        li      t3, 64

        la      t4, mat_dist
        li      t5, 512       // C column stride: 128 cols * 4 bytes = 512

        fence rw,rw
        msyncregreset   sync1

        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mzero   acc0
        // mmacc md, ms2, ms1  -> md += ms1 * ms2
        // Xuantie: mmacc.w.b acc0, tr0, tr1  -> acc0 += tr0 * tr1
        // Zames:   mmacc acc0, tr1, tr0
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mla     tr1,(t0),t1
        mlb     tr2,(t2),t3
        mzero   acc1
        mmacc   acc1, tr2, tr1
        msc     acc1, (t4), t5

        add    t4, t4, a7
        mla     tr2,(t0),t1
        mlb     tr3,(t2),t3
        mzero   acc2
        mmacc   acc2, tr3, tr2
        msc     acc2, (t4), t5

        add    t4, t4, a7
        mla     tr3,(t0),t1
        mlb     tr0,(t2),t3
        mzero   acc3
        mmacc   acc3, tr0, tr3
        msc     acc3, (t4), t5

        mrelease        sync1
        li a0, 1
        macquire        sync1, a0
        fence iorw,iorw

        li      a0,0
        la      a1, mat_dist

        li a5,4
check_ab_big:
        li a2 ,130
        li a3  , 128
check_ab_1:
        lw    a4,0(a1)

        bne a4,a2,check_fail
        addi a1,a1,4
        addi a3,a3,-1
        bnez a3,check_ab_1
        li a2,128
        li a3,16256
check_ab_2:
        lw    a4,0(a1)
        bne a4,a2,check_fail
        addi a1,a1,4
        addi a3,a3,-1
        bnez a3,check_ab_2

        addi a5,a5,-1
        bnez a5,check_ab_big

        j check_done

matrix_atb:
        fence iorw,iorw
        la a4,mat_a
        li  a5,2
        sb  a5,256(a4)

        la      t0, mat_a
        li      t1, 128

        la      t2, mat_b
        li      t3, 64

        la      t4, mat_dist
        li      t5, 512

        fence iorw,iorw
        msyncregreset   sync1

        mlat    tr0, (t0), t1
        mlb     tr1, (t2), t3
        mzero   acc0
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mlat    tr1,(t0),t1
        mlb     tr2,(t2),t3
        mzero   acc1
        mmacc   acc1, tr2, tr1
        msc     acc1, (t4), t5

        add    t4, t4, a7
        mlat    tr2,(t0),t1
        mlb     tr3,(t2),t3
        mzero   acc2
        mmacc   acc2, tr3, tr2
        msc     acc2, (t4), t5

        add    t4, t4, a7
        mlat    tr3,(t0),t1
        mlb     tr0,(t2),t3
        mzero   acc3
        mmacc   acc3, tr0, tr3
        msc     acc3, (t4), t5

        mrelease        sync1
        li a0, 1
        macquire        sync1, a0
        fence iorw,iorw

        li      a0,0
        la      a1, mat_dist

        li a5,4
check_atb_big:
        li a2 ,130
        li a3  , 128
check_atb_1:
        lw    a4,0(a1)

        bne a4,a2,check_fail
        addi a1,a1,4
        addi a3,a3,-1
        bnez a3,check_atb_1

        li a2,128
        li a3,16256
check_atb_2:
        lw    a4,0(a1)
        bne a4,a2,check_fail
        addi a1,a1,4
        addi a3,a3,-1
        bnez a3,check_atb_2

        addi a5,a5,-1
        bnez a5,check_atb_big

        j check_done


matrix_abt:
        fence iorw,iorw
        la a4,mat_b
        li  a5, 1
        sb  a5, 128(a4)

        la      t0, mat_a
        li      t1, 64

        la      t2, mat_b
        li      t3, 128

        la      t4, mat_dist
        li      t5, 512

        fence iorw,iorw
        msyncregreset   sync1

        mla     tr0, (t0), t1
        mlbt    tr1, (t2), t3
        mzero   acc0
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mla     tr1,(t0),t1
        mlbt    tr2,(t2),t3
        mzero   acc1
        mmacc   acc1, tr2, tr1
        msc     acc1, (t4), t5

        add    t4, t4, a7
        mla     tr2,(t0),t1
        mlbt    tr3,(t2),t3
        mzero   acc2
        mmacc   acc2, tr3, tr2
        msc     acc2, (t4), t5

        add    t4, t4, a7
        mla     tr3,(t0),t1
        mlbt    tr0,(t2),t3
        mzero   acc3
        mmacc   acc3, tr0, tr3
        msc     acc3, (t4), t5

        mrelease        sync1
        li a0, 1
        macquire        sync1, a0
        fence iorw,iorw

        li      a0,0
        la      a1, mat_dist

        li a7,4
check_abt_big:
        li a2 , 127
        li a3  , 128
        li a6,128

check_abt_1:
        lw    a4,0(a1)
        bne a4,a2,check_fail

        li a5, 127
        addi a1,a1,4
check_abt_2:  // rest 127
        lw a4,0(a1)
        addi a1,a1,4
        addi a5,a5,-1
        bne a4,a3,check_fail
        bnez a5,check_abt_2

        addi a6,a6,-1
        bnez a6,check_abt_1

        addi a7,a7,-1
        bnez a7,check_abt_big


        j check_done



matrix_atbt:
        fence iorw,iorw
        la a4,mat_b
        li  a5, 1
        sb  a5, 128(a4)
        la a4,mat_a
        li a5,2
        sb a5,128(a4)

        la      t0, mat_a
        li      t1, 128

        la      t2, mat_b
        li      t3, 128

        la      t4, mat_dist
        li      t5, 512

        fence iorw,iorw
        msyncregreset   sync1

        mlat    tr0, (t0), t1
        mlbt    tr1, (t2), t3
        mzero   acc0
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mlat    tr1,(t0),t1
        mlbt    tr2,(t2),t3
        mzero   acc1
        mmacc   acc1, tr2, tr1
        msc     acc1, (t4), t5

        add    t4, t4, a7
        mlat    tr2,(t0),t1
        mlbt    tr3,(t2),t3
        mzero   acc2
        mmacc   acc2, tr3, tr2
        msc     acc2, (t4), t5

        add    t4, t4, a7
        mlat    tr3,(t0),t1
        mlbt    tr0,(t2),t3
        mzero   acc3
        mmacc   acc3, tr0, tr3
        msc     acc3, (t4), t5

        mrelease        sync1
        li a0, 1
        macquire        sync1, a0
        fence iorw,iorw

        li      a0,0
        la      a1, mat_dist

        li a7,4
check_atbt_big: // first row
        li a2  ,128
        li a3  ,130

        lw    a4,0(a1)
        bne  a4,a2,check_fail
        addi a1,a1,4
        li a5,127
check_atbt_1:
        lw    a4,0(a1)
        bne a4,a3,check_fail
        addi a1,a1,4
        addi a5,a5,-1
        bnez a5,check_atbt_1
// end of first row

        li a2,127
        li a3,128
        li a6,127 // loop idx
check_atbt_2:// first elem of next row
        lw   a4,0(a1)
        bne a4,a2,check_fail // end of first elem of next row
// rest of next row
        li a5, 127
        addi a1,a1,4
check_atbt_3:
        lw a4,0(a1)
        addi a1,a1,4
        addi a5,a5,-1
        bne a4,a3,check_fail
        bnez a5,check_atbt_3

        addi a6,a6,-1
        bnez a6,check_atbt_2
// end of next row
        addi a7,a7,-1
        bnez a7,check_atbt_big // big loop for next matrix

        j check_done


check_fail:
        li      a0, 1         // error: set return value to 1

check_done:
        la a4,mat_a
        li  a5,1
        sb  a5,1(a4)
        sb  a5,64(a4)
        ret
