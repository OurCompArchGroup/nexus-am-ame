.text
.global matrix_init
.global restore_mat
.global matrix_ab
.global matrix_atb
.global matrix_abt
.global matrix_atbt
# .global matrix_restore

.global mat_dist
.global mat_a
.global mat_b

.section .rodata

.section .data
.align 8
mat_a:
        .rept   8192           # 128 * 64 = 8192 个元素
        .byte   1
        .endr


mat_b:
        .rept   8192           # 64 * 128 = 8192 个元素
        .byte   2
        .endr

mat_dist:
        .rept   65536          
        .word   3
        .endr


.section .text

# matrix_restore:
#         li a0,0
#         la  a1,mat_dist
#         li a2,3
#         li a3,65536
# restore_flag:
#         sw a2,0(a1)
#         addi a1,a1,4
#         addi a3,a3,-1
#         bnez a3,restore_flag
#         fence iorw,iorw
#         ret

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


matrix_ab:
        
        la a4,mat_a
        li  a5,2
        sb  a5,1(a4)

        la      t0, mat_a
        li      t1, 64        
        
        la      t2, mat_b
        li      t3, 64       
        
        la      t4, mat_dist
        li      t5, 512       # 矩阵 C 的列步长 (N = 128, 32位数据所以是 128 * 4 = 512 字节)
        
        fence rw,rw
        msyncreset      tok1
        
        mlae8   tr0, (t0), t1
        mlbe8   tr1, (t2), t3
        # mlce32  acc0, (t4), t5
        mzero1r acc0
        mmacc.w.b acc0, tr0, tr1
        msce32 acc0, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mlae8   tr1,(t0),t1
        mlbe8   tr2,(t2),t3
        mzero1r acc1
        mmacc.w.b acc1,tr1,tr2
        msce32  acc1, (t4), t5

        add    t4, t4, a7
        mlae8   tr2,(t0),t1
        mlbe8   tr3,(t2),t3
        mzero1r acc2
        mmacc.w.b acc2,tr2,tr3
        msce32  acc2, (t4), t5
        
        add    t4, t4, a7
        mlae8   tr3,(t0),t1
        mlbe8   tr0,(t2),t3
        mzero1r acc3
        mmacc.w.b acc3,tr3,tr0
        msce32  acc3, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
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
        li      t5, 512       # 矩阵 C 的列步长 (N = 128, 32位数据所以是 128 * 4 = 512 字节)
        
        fence iorw,iorw
        msyncreset      tok1
        
        mlate8   tr0, (t0), t1
        mlbe8   tr1, (t2), t3
        mzero1r acc0
        mmacc.w.b acc0, tr0, tr1
        msce32 acc0, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mlate8   tr1,(t0),t1
        mlbe8   tr2,(t2),t3
        mzero1r acc1
        mmacc.w.b acc1,tr1,tr2
        msce32  acc1, (t4), t5

        add    t4, t4, a7
        mlate8   tr2,(t0),t1
        mlbe8   tr3,(t2),t3
        mzero1r acc2
        mmacc.w.b acc2,tr2,tr3
        msce32  acc2, (t4), t5
        
        add    t4, t4, a7
        mlate8   tr3,(t0),t1
        mlbe8   tr0,(t2),t3
        mzero1r acc3
        mmacc.w.b acc3,tr3,tr0
        msce32  acc3, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
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
        li      t5, 512       # 矩阵 C 的列步长 (N = 128, 32位数据所以是 128 * 4 = 512 字节)
        
        fence iorw,iorw
        msyncreset      tok1
        
        mlae8   tr0, (t0), t1
        mlbte8   tr1, (t2), t3
        mzero1r acc0
        mmacc.w.b acc0, tr0, tr1
        msce32 acc0, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mlae8   tr1,(t0),t1
        mlbte8   tr2,(t2),t3
        mzero1r acc1
        mmacc.w.b acc1,tr1,tr2
        msce32  acc1, (t4), t5

        add    t4, t4, a7
        mlae8   tr2,(t0),t1
        mlbte8   tr3,(t2),t3
        mzero1r acc2
        mmacc.w.b acc2,tr2,tr3
        msce32  acc2, (t4), t5
        
        add    t4, t4, a7
        mlae8   tr3,(t0),t1
        mlbte8   tr0,(t2),t3
        mzero1r acc3
        mmacc.w.b acc3,tr3,tr0
        msce32  acc3, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
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
check_abt_2:  # rest 127
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
        li      t5, 512       # 矩阵 C 的列步长 (N = 128, 32位数据所以是 128 * 4 = 512 字节)
        
        fence iorw,iorw
        msyncreset      tok1
        
        mlate8   tr0, (t0), t1
        mlbte8   tr1, (t2), t3
        mzero1r acc0
        mmacc.w.b acc0, tr0, tr1
        msce32 acc0, (t4), t5

        li      a7,65536
        add    t4, t4, a7
        mlate8   tr1,(t0),t1
        mlbte8   tr2,(t2),t3
        mzero1r acc1
        mmacc.w.b acc1,tr1,tr2
        msce32  acc1, (t4), t5

        add    t4, t4, a7
        mlate8   tr2,(t0),t1
        mlbte8   tr3,(t2),t3
        mzero1r acc2
        mmacc.w.b acc2,tr2,tr3
        msce32  acc2, (t4), t5
        
        add    t4, t4, a7
        mlate8   tr3,(t0),t1
        mlbte8   tr0,(t2),t3
        mzero1r acc3
        mmacc.w.b acc3,tr3,tr0
        msce32  acc3, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
        fence iorw,iorw

        li      a0,0
        la      a1, mat_dist
        
        li a7,4
check_atbt_big: ## first row
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
### end of first row

        li a2,127
        li a3,128
        li a6,127 # loop idx
check_atbt_2:# first elem of next row
        lw   a4,0(a1)
        bne a4,a2,check_fail #end of first elem of next row
# rest of next row
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
## end of next row
        addi a7,a7,-1
        bnez a7,check_atbt_big #big loop for next matrix

        j check_done


check_fail:
        li      a0, 1         # 发现错误，设置错误标志为 1
        
check_done:
        la a4,mat_a
        li  a5,1
        sb  a5,1(a4)
        sb  a5,64(a4)
        ret



