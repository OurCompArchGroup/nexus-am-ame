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
# .global mat_target_t
.global matrix_restore

.section .rodata


.section .data
.align 8
mat_a:
    .byte 0x3C, 0x40, 0x42, 0x44   # 1.0, 2.0, 3.0, 4.0
    .byte 0x45, 0x46, 0x47, 0x48   # 5.0, 6.0, 7.0, 8.0
    .byte 0x48, 0x49, 0x49, 0x4A   # 8.0, 10.0, 10.0, 12.0
    .byte 0x4A, 0x4B, 0x4B, 0x4C   # 12.0, 14.0, 14.0, 16.0

mat_id:
    .byte 0x3C, 0x00, 0x00, 0x00
    .byte 0x00, 0x3C, 0x00, 0x00
    .byte 0x00, 0x00, 0x3C, 0x00
    .byte 0x00, 0x00, 0x00, 0x3C

# FP16 result storage for E4 test
mat_dist:
    .rept 16
    .hword 0x0000
    .endr
mat_target:
    .hword 0x3C00, 0x4000, 0x4200, 0x4400
    .hword 0x4500, 0x4600, 0x4700, 0x4800
    .hword 0x4800, 0x4900, 0x4900, 0x4A00
    .hword 0x4A00, 0x4B00, 0x4B00, 0x4C00
mat_target_double:
    .hword 0x4000, 0x4400, 0x4600, 0x4800   # 2.0, 4.0, 6.0, 8.0
    .hword 0x4900, 0x4A00, 0x4B00, 0x4C00   # 10.0, 12.0, 14.0, 16.0
    .hword 0x4C00, 0x4D00, 0x4D00, 0x4E00   # 16.0, 20.0, 20.0, 24.0
    .hword 0x4E00, 0x4F00, 0x4F00, 0x5000   # 24.0, 28.0, 28.0, 32.0
# matrix_c:
#         .rept    1024        # 128 * 128 = 16384 个元素
#         .byte   3
#         .endr

.section .text

# matrix_restore:
#         la  a1,mat_dist
#         li a2,0
#         li a3,64
# restore_flag:
#         sw a2,0(a1)
#         addi a1,a1,4
#         addi a3,a3,-1
#         bnez a3,restore_flag
#         ret

matrix_init:
        # 启用矩阵扩展
        lui a0, 8194
        addiw a0, a0, 512
        csrs mstatus, a0

        # 设置矩阵维度
        li a0, 4
        msettilem       a0      # M = 32
        msettilen       a0      # N = 32
        msettilek       a0      # K = 32
        
        msettilemi     4       # Mi = 32
        msettileni     4       # Ni = 32
        msettileki     4       # Ki = 32
        ret
        
matrix_restore:
    la      a1, mat_dist   # 目标FP16矩阵（示例）
    li      a2, 16                # 16个元素
    li      t0, 0
1:
    sh      t0, 0(a1)
    addi    a1, a1, 2
    addi    a2, a2, -1
    bnez    a2, 1b
    ret

matrix_mfmacc0:
    la      t0, mat_a            # A matrix (8-bit E4)
    li      t1, 4                   # column stride (4 bytes for 4 elements)
    la      t2, mat_id           # B matrix (8-bit E4 identity)
    li      t3, 4                   # column stride
    la      t4, mat_dist     # C matrix (FP16 result)
    li      t5, 8                   # column stride (8 bytes for 4 halfwords)

    msyncreset      tok1

    mlae8   tr0, (t0), t1            # load 8-bit E4 matrix A into tensor reg tr0
    mlbe8   tr1, (t2), t3            # load 8-bit E4 matrix B into tensor reg tr1
    mlce16  acc0, (t4), t5           # load initial FP16 values from C (all zeros)
    mfmacc.h.e5 acc0, tr0, tr1        # acc0 += tr0 * tr1 (8-bit mul, FP32 accumulate)
    msce16  acc0, (t4), t5           # store FP32 result as FP16 to C
    mrelease        tok1
    li      a0, 1
    macquire        a0, tok1

    # Verify results
    fence io,io
    li      a0, 0
    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16
check_loop0:
    lh      t0, 0(a1)                # load halfword result
    lh      t1, 0(a2)                # load halfword expected
    bne     t0, t1, check_fail
    addi    a1, a1, 2
    addi    a2, a2, 2
    addi    a3, a3, -1
    bnez    a3, check_loop0
    j check_done



matrix_mfmacc1:
    la      t0, mat_a            # A matrix (8-bit E4)
    li      t1, 4                   # column stride (4 bytes for 4 elements)
    la      t2, mat_id           # B matrix (8-bit E4 identity)
    li      t3, 4                   # column stride
    la      t4, mat_dist     # C matrix (FP16 result)
    li      t5, 8                   # column stride (8 bytes for 4 halfwords)

    msyncreset      tok1

    mlae8   tr1, (t0), t1            # load 8-bit E4 matrix A into tensor reg tr0
    mlbe8   tr2, (t2), t3            # load 8-bit E4 matrix B into tensor reg tr1
    mlce16  acc1, (t4), t5           # load initial FP16 values from C (all zeros)
    mfmacc.h.e5 acc1, tr1, tr2        # acc0 += tr0 * tr1 (8-bit mul, FP32 accumulate)
    msce16  acc1, (t4), t5           # store FP32 result as FP16 to C
    mrelease        tok1
    li      a0, 1
    macquire        a0, tok1

    # Verify results
    li      a0, 0
    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16
check_loop1:
    lh      t0, 0(a1)                # load halfword result
    lh      t1, 0(a2)                # load halfword expected
    bne     t0, t1, check_fail
    addi    a1, a1, 2
    addi    a2, a2, 2
    addi    a3, a3, -1
    bnez    a3, check_loop1
    j check_done



matrix_mfmacc2:
    la      t0, mat_a            # A matrix (8-bit E4)
    li      t1, 4                   # column stride (4 bytes for 4 elements)
    la      t2, mat_id           # B matrix (8-bit E4 identity)
    li      t3, 4                   # column stride
    la      t4, mat_dist     # C matrix (FP16 result)
    li      t5, 8                   # column stride (8 bytes for 4 halfwords)

    msyncreset      tok1

    mlae8   tr2, (t0), t1            # load 8-bit E4 matrix A into tensor reg tr0
    mlbe8   tr3, (t2), t3            # load 8-bit E4 matrix B into tensor reg tr1
    mlce16  acc2, (t4), t5           # load initial FP16 values from C (all zeros)
    mfmacc.h.e5 acc2, tr2, tr3        # acc0 += tr0 * tr1 (8-bit mul, FP32 accumulate)
    msce16  acc2, (t4), t5           # store FP32 result as FP16 to C
    mrelease        tok1
    li      a0, 1
    macquire        a0, tok1

    # Verify results
    li      a0, 0
    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16
check_loop2:
    lh      t0, 0(a1)                # load halfword result
    lh      t1, 0(a2)                # load halfword expected
    bne     t0, t1, check_fail
    addi    a1, a1, 2
    addi    a2, a2, 2
    addi    a3, a3, -1
    bnez    a3, check_loop2
    j check_done



matrix_mfmacc3:
    la      t0, mat_a            # A matrix (8-bit E4)
    li      t1, 4                   # column stride (4 bytes for 4 elements)
    la      t2, mat_id           # B matrix (8-bit E4 identity)
    li      t3, 4                   # column stride
    la      t4, mat_dist     # C matrix (FP16 result)
    li      t5, 8                   # column stride (8 bytes for 4 halfwords)

    msyncreset      tok1

    mlae8   tr3, (t0), t1            # load 8-bit E4 matrix A into tensor reg tr0
    mlbe8   tr0, (t2), t3            # load 8-bit E4 matrix B into tensor reg tr1
    mlce16  acc3, (t4), t5           # load initial FP16 values from C (all zeros)
    mfmacc.h.e5 acc3, tr3, tr0        # acc0 += tr0 * tr1 (8-bit mul, FP32 accumulate)
    msce16  acc3, (t4), t5           # store FP32 result as FP16 to C
    mrelease        tok1
    li      a0, 1
    macquire        a0, tok1

    # Verify results
    li      a0, 0
    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16
check_loop3:
    lh      t0, 0(a1)                # load halfword result
    lh      t1, 0(a2)                # load halfword expected
    bne     t0, t1, check_fail
    addi    a1, a1, 2
    addi    a2, a2, 2
    addi    a3, a3, -1
    bnez    a3, check_loop3
    j check_done


matrix_mfmacc4:
    la      t0, mat_a            # A matrix (8-bit E4)
    li      t1, 4                   # column stride (4 bytes for 4 elements)
    la      t2, mat_id           # B matrix (8-bit E4 identity)
    li      t3, 4                   # column stride
    la      t4, mat_dist     # C matrix (FP16 result)
    li      t5, 8                   # column stride (8 bytes for 4 halfwords)

    msyncreset      tok1

    mlae8   tr0, (t0), t1            # load 8-bit E4 matrix A into tensor reg tr0
    mlbe8   tr1, (t2), t3            # load 8-bit E4 matrix B into tensor reg tr1
    mlce16  acc0, (t4), t5           # load initial FP16 values from C (all zeros)
    mfmacc.h.e5 acc0, tr0, tr1        # acc0 += tr0 * tr1 (8-bit mul, FP32 accumulate)
    mfmacc.h.e5 acc0, tr0, tr1
    msce16  acc0, (t4), t5           # store FP32 result as FP16 to C
    mrelease        tok1
    li      a0, 1
    macquire        a0, tok1

    # Verify results
    fence io,io
    li      a0, 0
    la      a1, mat_dist
    la      a2, mat_target_double
    li      a3, 16
check_loop4:
    lh      t0, 0(a1)                # load halfword result
    lh      t1, 0(a2)                # load halfword expected
    bne     t0, t1, check_fail
    addi    a1, a1, 2
    addi    a2, a2, 2
    addi    a3, a3, -1
    bnez    a3, check_loop4
    j check_done


check_fail:
    li      a0, 1          # 返回值：1表示失败



check_done:
        ret
