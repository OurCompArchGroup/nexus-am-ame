.text
.global matrix_zero_verify
.global matrix_zero_tr_verify
.global matrix_init
.global mat_dist

.section .rodata


.section .data
.align 8
# 矩阵 A: 128 x 64, 每个元素 1 字节 (8位), 所有元素初始化为 1


# 矩阵 C: 4x4, 每个元素 4 字节 (32位), 所有元素初始化为 3
mat_a:
        .rept   16384           # 128 * 64 = 8192 个元素
        .word   1
        .endr
mat_b:
        .rept   16384           # 64 * 128 = 8192 个元素
        .word   2
        .endr
mat_dist:
        .rept    65536        
        .word   3
        .endr
# mat_id:
#  .word 1,0,0,0
#  .word 0,1,0,0
#  .word 0,0,1,0
#  .word 0,0,0,1
.section .text

matrix_init:
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

matrix_zero_verify:
        
        la      t4, mat_dist    # 矩阵C的地址
        li      t5, 512        # 列步长 (N=32, 每个元素1字节)
        
        # 重置token
        msyncreset      tok1
        
        la      t4, mat_dist
        mlce32   acc0, (t4), t5
        mzero1r acc0
        msce32  acc0, (t4), t5
        
        
        li      a7,65536
        add    t4, t4, a7
        mlce32   acc1, (t4), t5
        mzero1r acc1
        msce32  acc1, (t4), t5
        

        add    t4, t4, a7
        mlce32   acc2, (t4), t5
        mzero1r acc2
        msce32  acc2, (t4), t5
        
        add    t4, t4, a7
        mlce32   acc3, (t4), t5
        mzero1r acc3
        msce32  acc3, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
        

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

matrix_zero_tr_verify:
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

        msyncreset      tok1
        

        
        mlae8   tr0,(t0),t1
        mlbe8   tr1,(t2),t3
        mlce32  acc0, (t4), t5
        mzero1r tr0
        mmacc.w.b acc0,tr0,tr1
        fence iorw,iorw
        msce32  acc0, (t4), t5
        
        li      a7,65536
        add    t4, t4, a7
        mlae8   tr1,(t0),t1
        mlbe8   tr2,(t2),t3
        mlce32   acc1, (t4), t5
        mzero1r tr1
        mmacc.w.b acc1,tr1,tr2
        msce32  acc1, (t4), t5
        fence iorw,iorw

        add    t4, t4, a7
        mlae8   tr2,(t0),t1
        mlbe8   tr3,(t2),t3
        mlce32   acc2, (t4), t5
        mzero1r tr2
        mmacc.w.b acc2,tr2,tr3
        msce32  acc2, (t4), t5
        fence iorw,iorw
        add    t4, t4, a7
        mlae8   tr3,(t0),t1
        mlbe8   tr0,(t2),t3
        mlce32   acc3, (t4), t5
        mzero1r tr3
        mmacc.w.b acc3,tr3,tr0
        msce32  acc3, (t4), t5
        
        # ======================================
        
        # 释放token
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
        fence iorw,iorw
        

        li      a0,0
        la      a1, mat_dist
        li      a2,0
        # li      a3,65536
        li a3,16384
check_tr_loop:
        lw      a4, 0(a1)
        bne     a4, a2, check_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, check_tr_loop
        j check_done


check_fail:
        li      a0, 1          # 返回值：1表示失败

check_done:
        ret

