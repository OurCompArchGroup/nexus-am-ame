# AME proposal-14 FP8 E5M2 x FP8 E5M2 -> FP32 matrix multiply tests.

.text
.global matrix_init
.global matrix_mmacc0
.global matrix_mmacc1
.global matrix_mmacc2
.global matrix_mmacc3
.global matrix_mmacc4
.global mat_dist
.global mat_target
.global mat_target_double
.global matrix_restore

# mcfg type_code constants for table_sel = 0.
.equ MCFG_FP8E5M2, 0x07
.equ MCFG_FP32,    0x0c

.macro SETCFG mcfg_sel, type_code
    li      t6, \type_code
    msetcfg \mcfg_sel, t6
.endm

.macro FINISH
    mrelease        sync1
    li              a0, 1
    macquire        sync1, a0
    mfence
    fence           rw, rw
.endm

.section .data
.align 8
mat_a:
    .rept 8192
    .byte 0x3c                    # FP8 E5M2 value 1.0
    .endr

mat_b:
    .rept 8192
    .byte 0x3c                    # FP8 E5M2 value 1.0
    .endr

mat_dist:
    .rept 16384
    .word 0x00000000
    .endr

mat_target:
    .rept 16384
    .word 0x42800000              # 64.0f
    .endr

mat_target_double:
    .rept 16384
    .word 0x43000000              # 128.0f
    .endr

.section .text

matrix_init:
    lui     a0, 8194
    addiw   a0, a0, 512
    csrs    mstatus, a0

    li      a0, 128
    msettilem a0
    msettilen a0
    li      a0, 64
    msettilek a0
    msettilemi 128
    msettileni 128
    msettileki 64
    ret

matrix_restore:
    la      a1, mat_dist
    li      a2, 16384
    li      t0, 0
1:
    sw      t0, 0(a1)
    addi    a1, a1, 4
    addi    a2, a2, -1
    bnez    a2, 1b
    ret

matrix_mmacc0:
    la      t0, mat_a
    li      t1, 64
    la      t2, mat_b
    li      t3, 64
    la      t4, mat_dist
    li      t5, 512

    msyncregreset sync1
    mfence
    SETCFG  mcfg0, MCFG_FP8E5M2
    SETCFG  mcfg1, MCFG_FP8E5M2
    SETCFG  mcfg4, MCFG_FP32
    mla     tr0, (t0), t1
    mlb     tr1, (t2), t3
    mlc     acc0, (t4), t5
    mmacc   acc0, tr1, tr0
    msc     acc0, (t4), t5
    FINISH

    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16384
1:
    lw      t0, 0(a1)
    lw      t1, 0(a2)
    bne     t0, t1, 2f
    addi    a1, a1, 4
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, 1b
    li      a0, 0
    ret
2:
    li      a0, 1
    ret

matrix_mmacc1:
    la      t0, mat_a
    li      t1, 64
    la      t2, mat_b
    li      t3, 64
    la      t4, mat_dist
    li      t5, 512

    msyncregreset sync1
    mfence
    SETCFG  mcfg1, MCFG_FP8E5M2
    SETCFG  mcfg2, MCFG_FP8E5M2
    SETCFG  mcfg5, MCFG_FP32
    mla     tr1, (t0), t1
    mlb     tr2, (t2), t3
    mlc     acc1, (t4), t5
    mmacc   acc1, tr2, tr1
    msc     acc1, (t4), t5
    FINISH

    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16384
3:
    lw      t0, 0(a1)
    lw      t1, 0(a2)
    bne     t0, t1, 4f
    addi    a1, a1, 4
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, 3b
    li      a0, 0
    ret
4:
    li      a0, 1
    ret

matrix_mmacc2:
    la      t0, mat_a
    li      t1, 64
    la      t2, mat_b
    li      t3, 64
    la      t4, mat_dist
    li      t5, 512

    msyncregreset sync1
    mfence
    SETCFG  mcfg2, MCFG_FP8E5M2
    SETCFG  mcfg3, MCFG_FP8E5M2
    SETCFG  mcfg6, MCFG_FP32
    mla     tr2, (t0), t1
    mlb     tr3, (t2), t3
    mlc     acc2, (t4), t5
    mmacc   acc2, tr3, tr2
    msc     acc2, (t4), t5
    FINISH

    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16384
5:
    lw      t0, 0(a1)
    lw      t1, 0(a2)
    bne     t0, t1, 6f
    addi    a1, a1, 4
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, 5b
    li      a0, 0
    ret
6:
    li      a0, 1
    ret

matrix_mmacc3:
    la      t0, mat_a
    li      t1, 64
    la      t2, mat_b
    li      t3, 64
    la      t4, mat_dist
    li      t5, 512

    msyncregreset sync1
    mfence
    SETCFG  mcfg3, MCFG_FP8E5M2
    SETCFG  mcfg0, MCFG_FP8E5M2
    SETCFG  mcfg7, MCFG_FP32
    mla     tr3, (t0), t1
    mlb     tr0, (t2), t3
    mlc     acc3, (t4), t5
    mmacc   acc3, tr0, tr3
    msc     acc3, (t4), t5
    FINISH

    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16384
7:
    lw      t0, 0(a1)
    lw      t1, 0(a2)
    bne     t0, t1, 8f
    addi    a1, a1, 4
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, 7b
    li      a0, 0
    ret
8:
    li      a0, 1
    ret

matrix_mmacc4:
    la      t0, mat_a
    li      t1, 64
    la      t2, mat_b
    li      t3, 64
    la      t4, mat_dist
    li      t5, 512

    msyncregreset sync1
    mfence
    SETCFG  mcfg0, MCFG_FP8E5M2
    SETCFG  mcfg1, MCFG_FP8E5M2
    SETCFG  mcfg4, MCFG_FP32
    mla     tr0, (t0), t1
    mlb     tr1, (t2), t3
    mlc     acc0, (t4), t5
    mmacc   acc0, tr1, tr0
    mmacc   acc0, tr1, tr0
    msc     acc0, (t4), t5
    FINISH

    la      a1, mat_dist
    la      a2, mat_target_double
    li      a3, 16384
9:
    lw      t0, 0(a1)
    lw      t1, 0(a2)
    bne     t0, t1, 10f
    addi    a1, a1, 4
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, 9b
    li      a0, 0
    ret
10:
    li      a0, 1
    ret
