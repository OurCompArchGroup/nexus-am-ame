.text
.global matrix_init
.global matrix_mmacc0
.global matrix_mmacc1
.global matrix_mmacc2
.global matrix_mmacc3
.global matrix_mmacc4
.global matrix_restore

.global mat_dist


.section .rodata

.section .data
.align 8
# mat_a:
#         .rept   8192           # 128 * 64 = 8192 个元素
#         .byte   1
#         .endr

mat_na:
        .rept   8192           # 128 * 64 = 8192 个元素
        .byte   -1
        .endr
# mat_b:
#         .rept   8192           # 64 * 128 = 8192 个元素
#         .byte   2
#         .endr
mat_nb:
        .rept   8192           # 64 * 128 = 8192 个元素
        .byte   -2
        .endr
mat_dist:
        .rept   65536          # 128 * 128 = 16384 个元素
        .word   3
        .endr


.section .text

matrix_restore:
        li a0,0
        la  a1,mat_dist
        li a2,3
        # li a3,65536
        li a3,16384
restore_flag:
        sw a2,0(a1)
        sw a2,4(a1)
        sw a2,8(a1)
        sw a2,12(a1)
        addi a1,a1,16
        addi a3,a3,-1
        bnez a3,restore_flag
        fence rw,rw
        ret


matrix_init:
        # set mstatus, enable matrix extension
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
        ret


matrix_mmacc0:

        la      t0, mat_na
        li      t1, 64        
        
        la      t2, mat_nb
        li      t3, 64       
        
        la      t4, mat_dist
        li      t5, 512       # 矩阵 C 的列步长 (N = 128, 32位数据所以是 128 * 4 = 512 字节)

        msyncreset      tok1
        
        mlae8   tr0, (t0), t1
        mlbe8   tr1, (t2), t3
        mlce32  acc0, (t4), t5
        mmacc.w.b acc0, tr0, tr1
        msce32 acc0, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mlae8   tr1,(t0),t1
        mlbe8   tr2,(t2),t3
        mlce32   acc1, (t4), t5
        mmaccu.w.b acc1,tr1,tr2
        msce32  acc1, (t4), t5

        add    t4, t4, a7
        mlae8   tr2,(t0),t1
        mlbe8   tr3,(t2),t3
        mlce32   acc2, (t4), t5
        mmaccsu.w.b acc2,tr2,tr3
        msce32  acc2, (t4), t5
        
        add    t4, t4, a7
        mlae8   tr3,(t0),t1
        mlbe8   tr0,(t2),t3
        mlce32   acc3, (t4), t5
        mmaccus.w.b acc3,tr3,tr0
        msce32  acc3, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
        j matrix_check

        
matrix_mmacc1:

        la      t0, mat_na
        li      t1, 64        
        
        la      t2, mat_nb
        li      t3, 64       
        
        la      t4, mat_dist
        li      t5, 512       # 矩阵 C 的列步长 (N = 128, 32位数据所以是 128 * 4 = 512 字节)

        msyncreset      tok1
        
        mlae8   tr0, (t0), t1
        mlbe8   tr1, (t2), t3
        mlce32  acc1, (t4), t5
        mmacc.w.b acc1, tr0, tr1
        msce32 acc1, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mlae8   tr1,(t0),t1
        mlbe8   tr2,(t2),t3
        mlce32   acc2, (t4), t5
        mmaccu.w.b acc2,tr1,tr2
        msce32  acc2, (t4), t5

        add    t4, t4, a7
        mlae8   tr2,(t0),t1
        mlbe8   tr3,(t2),t3
        mlce32   acc3, (t4), t5
        mmaccsu.w.b acc3,tr2,tr3
        msce32  acc3, (t4), t5
        
        add    t4, t4, a7
        mlae8   tr3,(t0),t1
        mlbe8   tr0,(t2),t3
        mlce32   acc0, (t4), t5
        mmaccus.w.b acc0,tr3,tr0
        msce32  acc0, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
        j matrix_check

matrix_mmacc2:

        la      t0, mat_na
        li      t1, 64        
        
        la      t2, mat_nb
        li      t3, 64       
        
        la      t4, mat_dist
        li      t5, 512       # 矩阵 C 的列步长 (N = 128, 32位数据所以是 128 * 4 = 512 字节)

        msyncreset      tok1
        
        mlae8   tr0, (t0), t1
        mlbe8   tr1, (t2), t3
        mlce32  acc2, (t4), t5
        mmacc.w.b  acc2, tr0, tr1
        msce32 acc2, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mlae8   tr1,(t0),t1
        mlbe8   tr2,(t2),t3
        mlce32   acc3, (t4), t5
        mmaccu.w.b  acc3,tr1,tr2
        msce32  acc3, (t4), t5

        add    t4, t4, a7
        mlae8   tr2,(t0),t1
        mlbe8   tr3,(t2),t3
        mlce32   acc0, (t4), t5
        mmaccsu.w.b acc0,tr2,tr3
        msce32  acc0, (t4), t5
        
        add    t4, t4, a7
        mlae8   tr3,(t0),t1
        mlbe8   tr0,(t2),t3
        mlce32   acc1, (t4), t5
        mmaccus.w.b acc1,tr3,tr0
        msce32  acc1, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
        j matrix_check
matrix_mmacc3:

        la      t0, mat_na
        li      t1, 64        
        
        la      t2, mat_nb
        li      t3, 64       
        
        la      t4, mat_dist
        li      t5, 512       # 矩阵 C 的列步长 (N = 128, 32位数据所以是 128 * 4 = 512 字节)

        msyncreset      tok1
        
        mlae8   tr0, (t0), t1
        mlbe8   tr1, (t2), t3
        mlce32  acc3, (t4), t5
        mmacc.w.b  acc3, tr0, tr1
        msce32 acc3, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mlae8   tr1,(t0),t1
        mlbe8   tr2,(t2),t3
        mlce32   acc0, (t4), t5
        mmaccu.w.b  acc0,tr1,tr2
        msce32  acc0, (t4), t5

        add    t4, t4, a7
        mlae8   tr2,(t0),t1
        mlbe8   tr3,(t2),t3
        mlce32   acc1, (t4), t5
        mmaccsu.w.b acc1,tr2,tr3
        msce32  acc1, (t4), t5
        
        add    t4, t4, a7
        mlae8   tr3,(t0),t1
        mlbe8   tr0,(t2),t3
        mlce32   acc2, (t4), t5
        mmaccus.w.b acc2,tr3,tr0
        msce32  acc2, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
        j matrix_check

matrix_mmacc4:
        la      t0, mat_na
        li      t1, 64        
        
        la      t2, mat_nb
        li      t3, 64       
        
        la      t4, mat_dist
        li      t5, 512      
        msyncreset      tok1
        
        mlae8   tr0, (t0), t1
        mlbe8   tr1, (t2), t3
        mlce32  acc0, (t4), t5
        mmacc.w.b acc0, tr0, tr1
        mmacc.w.b acc0, tr0, tr1
        msce32 acc0, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
        fence rw,rw
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
        j check_done


matrix_check:
        fence rw,rw
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

        li a2, 4145283
        li      a3, 16384   
check_loop1:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop1

        li a2, -16253
        li      a3, 16384   
 check_loop2:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop2

        li a2, -32637
        li      a3, 16384   
check_loop3:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_loop3

        j check_done
    
check_fail:
        li      a0, 1         # 发现错误，设置错误标志为 1
        
check_done:
        ret

