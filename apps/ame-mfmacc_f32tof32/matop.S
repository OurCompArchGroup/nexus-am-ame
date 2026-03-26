.text
.global matrix_init
.global matrix_mfmacc0
.global matrix_mfmacc1
.global matrix_mfmacc2
.global matrix_mfmacc3
.global matrix_mfmacc4
.global mat_dist
.global mat_target
.global mat_target_n
.global matrix_restore

.section .rodata


.section .data
.align 8
mat_id_fp32:
    .float 1.0, 0.0, 0.0, 0.0
    .float 0.0, 1.0, 0.0, 0.0
    .float 0.0, 0.0, 1.0, 0.0
    .float 0.0, 0.0, 0.0, 1.0

mat_nid_fp32:
    .float -1.0, 0.0, 0.0, 0.0
    .float 0.0, -1.0, 0.0, 0.0
    .float 0.0, 0.0, -1.0, 0.0
    .float 0.0, 0.0, 0.0, -1.0

# Expected results (FP32) - 由于累加器是FP32
mat_target:
    .float 1.0, 2.0, 3.0, 4.0
    .float 5.0, 6.0, 7.0, 8.0
    .float 9.0, 10.0, 11.0, 12.0
    .float 13.0, 14.0, 15.0, 16.0
mat_target_n:
    .float -1.0, -2.0, -3.0, -4.0
    .float -5.0, -6.0, -7.0, -8.0
    .float -9.0, -10.0, -11.0, -12.0
    .float -13.0, -14.0, -15.0, -16.0

mat_a_fp32:
    .float 1.0, 2.0, 3.0, 4.0
    .float 5.0, 6.0, 7.0, 8.0
    .float 9.0, 10.0, 11.0, 12.0
    .float 13.0, 14.0, 15.0, 16.0

mat_na_fp32:
    .float -1.0, -2.0, -3.0, -4.0
    .float -5.0, -6.0, -7.0, -8.0
    .float -9.0, -10.0, -11.0, -12.0
    .float -13.0, -14.0, -15.0, -16.0

mat_dist:
    .rept 16
    .float 0.0
    .endr


.section .text


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
    la      a1, mat_dist
    li      a2, 0
    li      a3, 16
restore_flag_fp:
    fmv.w.x ft0, a2        # 将整数0转换为浮点数0.0
    fsw     ft0, 0(a1)     # 存储单精度浮点数
    addi    a1, a1, 4
    addi    a3, a3, -1
    bnez    a3, restore_flag_fp
    ret


matrix_mfmacc0:
    la      t0, mat_a_fp32    
    li      t1, 16             
    la      t2, mat_id_fp32   
    li      t3, 16           
    
    la      t4, mat_dist 
    li      t5, 16          
    

    msyncreset      tok1
  
    mlae32  tr0, (t0), t1     
    mlbe32  tr1, (t2), t3   
    mlce32  acc0, (t4), t5
    mfmacc.s acc0, tr0, tr1  
    msce32  acc0, (t4), t5   

    mrelease        tok1
    li      a0, 1
    macquire        a0, tok1
    

    li      a0, 0
    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16           # 检查前16个元素（4x4矩阵）
    
check_loop0:
    flw     ft0, 0(a1)       # 加载计算结果
    flw     ft1, 0(a2)       # 加载预期结果

    feq.s   t6, ft0, ft1
    beqz    t6, check_fail
    
    addi    a1, a1, 4
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, check_loop0
    j       check_done

matrix_mfmacc1:
    la      t0, mat_na_fp32    
    li      t1, 16             
    la      t2, mat_id_fp32   
    li      t3, 16           
    
    la      t4, mat_dist 
    li      t5, 16          
    

    msyncreset      tok1
  
    mlae32  tr1, (t0), t1     
    mlbe32  tr2, (t2), t3   
    mlce32  acc1, (t4), t5
    mfmacc.s acc1, tr1, tr2  
    msce32  acc1, (t4), t5   

    mrelease        tok1
    li      a0, 1
    macquire        a0, tok1
    

    li      a0, 0
    la      a1, mat_dist
    la      a2, mat_target_n
    li      a3, 16           # 检查前16个元素（4x4矩阵）
    
check_loop1:
    flw     ft0, 0(a1)       # 加载计算结果
    flw     ft1, 0(a2)       # 加载预期结果

    feq.s   t6, ft0, ft1
    beqz    t6, check_fail
    
    addi    a1, a1, 4
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, check_loop1
    j       check_done


matrix_mfmacc2:
    la      t0, mat_a_fp32    
    li      t1, 16             
    la      t2, mat_nid_fp32   
    li      t3, 16           
    
    la      t4, mat_dist 
    li      t5, 16          
    

    msyncreset      tok1
  
    mlae32  tr2, (t0), t1     
    mlbe32  tr3, (t2), t3   
    mlce32  acc2, (t4), t5
    mfmacc.s acc2, tr2, tr3  
    msce32  acc2, (t4), t5   

    mrelease        tok1
    li      a0, 1
    macquire        a0, tok1
    

    li      a0, 0
    la      a1, mat_dist
    la      a2, mat_target_n
    li      a3, 16           # 检查前16个元素（4x4矩阵）
    
check_loop2:
    flw     ft0, 0(a1)       # 加载计算结果
    flw     ft1, 0(a2)       # 加载预期结果

    feq.s   t6, ft0, ft1
    beqz    t6, check_fail
    
    addi    a1, a1, 4
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, check_loop2
    j       check_done

matrix_mfmacc3:
    la      t0, mat_na_fp32    
    li      t1, 16             
    la      t2, mat_nid_fp32   
    li      t3, 16           
    
    la      t4, mat_dist 
    li      t5, 16          
    

    msyncreset      tok1
  
    mlae32  tr3, (t0), t1     
    mlbe32  tr0, (t2), t3   
    mlce32  acc3, (t4), t5
  
    mfmacc.s acc3, tr3, tr0  
    msce32  acc3, (t4), t5   

    mrelease        tok1
    li      a0, 1
    macquire        a0, tok1
    

    li      a0, 0
    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16           # 检查前16个元素（4x4矩阵）
    
check_loop3:
    flw     ft0, 0(a1)       # 加载计算结果
    flw     ft1, 0(a2)       # 加载预期结果

    feq.s   t6, ft0, ft1
    beqz    t6, check_fail
    
    addi    a1, a1, 4
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, check_loop3
    j       check_done

matrix_mfmacc4:
    la      t0, mat_a_fp32    
    li      t1, 16             
    la      t2, mat_id_fp32   
    li      t3, 16           
    
    la      t4, mat_dist 
    li      t5, 16          
    

    msyncreset      tok1
  
    mlae32  tr0, (t0), t1     
    mlbe32  tr1, (t2), t3   

    mlce32  acc0, (t4), t5
    mfmacc.s acc0, tr0, tr1  
    mfmacc.s acc0, tr0, tr1  
    msce32  acc0, (t4), t5   

    mrelease        tok1
    li      a0, 1
    macquire        a0, tok1
    

    li      a0, 0
    la      a1, mat_dist
    la      a2, mat_target
    li      a3, 16           # 检查前16个元素（4x4矩阵）
    
check_loop4:
    flw     ft0, 0(a1)       # 加载计算结果
    flw     ft1, 0(a2)       # 加载预期结果
    fadd.s  ft2,ft1,ft1

    feq.s   t6, ft0, ft2
    beqz    t6, check_fail
    
    addi    a1, a1, 4
    addi    a2, a2, 4
    addi    a3, a3, -1
    bnez    a3, check_loop4 
    j       check_done

check_fail:
        li      a0, 1          # 返回值：1表示失败

check_done:
        ret

