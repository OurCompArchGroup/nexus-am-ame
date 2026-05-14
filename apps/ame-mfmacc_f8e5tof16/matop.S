.text
.global matrix_init
.global matrix_mfmacc0
.global matrix_mfmacc1
.global matrix_mfmacc2
.global matrix_mfmacc3
.global matrix_mfmacc4
.global mat_dist
.global mat_target
.global mat_target_double
.global matrix_restore

.section .rodata

.section .data
.align 8
mat_a:
    .rept 8192
    .byte 0x3C
    .endr

mat_b:
    .rept 8192
    .byte 0x3C
    .endr

mat_dist:
    .rept 16384
    .hword 0x0000
    .endr

mat_target:
    .rept 16384
    .hword 0x5400
    .endr

mat_target_double:
    .rept 16384
    .hword 0x5800
    .endr


.section .text


matrix_init:
    lui a0, 8194
    addiw a0, a0, 512
    csrs mstatus, a0

    li a0, 128
    msettilem a0
    msettilen a0
    li a0, 64
    msettilek a0

    msettilemi 128
    msettileni 128
    msettileki 64
    ret

matrix_restore:
    la  a1, mat_dist
    li  a2, 16384
    li  t0, 0
1:
    sh  t0, 0(a1)
    addi a1, a1, 2
    addi a2, a2, -1
    bnez a2, 1b
    ret


matrix_mfmacc0:
    la  t0, mat_a
    li  t1, 64
    la  t2, mat_b
    li  t3, 64
    la  t4, mat_dist
    li  t5, 256

    msyncreset tok1

    mlae8  tr0, (t0), t1
    mlbe8  tr1, (t2), t3
    mlce16 acc0, (t4), t5
    mfmacc.h.e5 acc0, tr0, tr1
    msce16 acc0, (t4), t5
    mrelease tok1
    li a0, 1
    macquire a0, tok1

    fence io,io
    li  a0, 0
    la  a1, mat_dist
    la  a2, mat_target
    li  a3, 16384
check_loop0:
    lh  t0, 0(a1)
    lh  t1, 0(a2)
    bne t0, t1, check_fail
    addi a1, a1, 2
    addi a2, a2, 2
    addi a3, a3, -1
    bnez a3, check_loop0
    j check_done


matrix_mfmacc1:
    la  t0, mat_a
    li  t1, 64
    la  t2, mat_b
    li  t3, 64
    la  t4, mat_dist
    li  t5, 256

    msyncreset tok1

    mlae8  tr1, (t0), t1
    mlbe8  tr2, (t2), t3
    mlce16 acc1, (t4), t5
    mfmacc.h.e5 acc1, tr1, tr2
    msce16 acc1, (t4), t5
    mrelease tok1
    li a0, 1
    macquire a0, tok1

    li  a0, 0
    la  a1, mat_dist
    la  a2, mat_target
    li  a3, 16384
check_loop1:
    lh  t0, 0(a1)
    lh  t1, 0(a2)
    bne t0, t1, check_fail
    addi a1, a1, 2
    addi a2, a2, 2
    addi a3, a3, -1
    bnez a3, check_loop1
    j check_done


matrix_mfmacc2:
    la  t0, mat_a
    li  t1, 64
    la  t2, mat_b
    li  t3, 64
    la  t4, mat_dist
    li  t5, 256

    msyncreset tok1

    mlae8  tr2, (t0), t1
    mlbe8  tr3, (t2), t3
    mlce16 acc2, (t4), t5
    mfmacc.h.e5 acc2, tr2, tr3
    msce16 acc2, (t4), t5
    mrelease tok1
    li a0, 1
    macquire a0, tok1

    li  a0, 0
    la  a1, mat_dist
    la  a2, mat_target
    li  a3, 16384
check_loop2:
    lh  t0, 0(a1)
    lh  t1, 0(a2)
    bne t0, t1, check_fail
    addi a1, a1, 2
    addi a2, a2, 2
    addi a3, a3, -1
    bnez a3, check_loop2
    j check_done


matrix_mfmacc3:
    la  t0, mat_a
    li  t1, 64
    la  t2, mat_b
    li  t3, 64
    la  t4, mat_dist
    li  t5, 256

    msyncreset tok1

    mlae8  tr3, (t0), t1
    mlbe8  tr0, (t2), t3
    mlce16 acc3, (t4), t5
    mfmacc.h.e5 acc3, tr3, tr0
    msce16 acc3, (t4), t5
    mrelease tok1
    li a0, 1
    macquire a0, tok1

    li  a0, 0
    la  a1, mat_dist
    la  a2, mat_target
    li  a3, 16384
check_loop3:
    lh  t0, 0(a1)
    lh  t1, 0(a2)
    bne t0, t1, check_fail
    addi a1, a1, 2
    addi a2, a2, 2
    addi a3, a3, -1
    bnez a3, check_loop3
    j check_done


matrix_mfmacc4:
    la  t0, mat_a
    li  t1, 64
    la  t2, mat_b
    li  t3, 64
    la  t4, mat_dist
    li  t5, 256

    msyncreset tok1

    mlae8  tr0, (t0), t1
    mlbe8  tr1, (t2), t3
    mlce16 acc0, (t4), t5
    mfmacc.h.e5 acc0, tr0, tr1
    mfmacc.h.e5 acc0, tr0, tr1
    msce16 acc0, (t4), t5
    mrelease tok1
    li a0, 1
    macquire a0, tok1

    fence io,io
    li  a0, 0
    la  a1, mat_dist
    la  a2, mat_target_double
    li  a3, 16384
check_loop4:
    lh  t0, 0(a1)
    lh  t1, 0(a2)
    bne t0, t1, check_fail
    addi a1, a1, 2
    addi a2, a2, 2
    addi a3, a3, -1
    bnez a3, check_loop4
    j check_done

check_fail:
    li  a0, 1

check_done:
    ret
