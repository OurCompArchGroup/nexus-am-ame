.text
.global matrix_init
.global matrix_ls
.global matrix_ltst
.global matrix_lts
.global matrix_lst

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

matrix_init:
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

        ret
matrix_ls:
        la      t0, mat_a
        li  t1,512
        la      t4,mat_dist
        li      t5,512
        li      a7,65536
        fence iorw,iorw
        msyncreset      tok1
        mlce32  acc0, (t0), t1
        msce32 acc0, (t4), t5

        add    t4, t4, a7
        mlce32  acc1, (t0), t1
        msce32 acc1, (t4), t5

        add    t4, t4, a7
        mlce32  acc2, (t0), t1
        msce32 acc2, (t4), t5
        
        add    t4, t4, a7
        mlce32  acc3, (t0), t1
        msce32 acc3, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
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

matrix_ltst:
        la      t0, mat_a
        li  t1,512
        la      t4,mat_dist
        li      t5,512
        li      a7,65536

        msyncreset      tok1
        mlcte32  acc0, (t0), t1
        mscte32 acc0, (t4), t5

        add    t4, t4, a7
        mlcte32  acc1, (t0), t1
        mscte32 acc1, (t4), t5

        add    t4, t4, a7
        mlcte32  acc2, (t0), t1
        mscte32 acc2, (t4), t5
        
        add    t4, t4, a7
        mlcte32  acc3, (t0), t1
        mscte32 acc3, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
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


matrix_lts:
        la      t0, mat_a
        li      t1,512
        la      t4,mat_dist
        li      t5,512
        li      a7,65536

        msyncreset      tok1
        mlcte32  acc0, (t0), t1
        msce32 acc0, (t4), t5

        add    t4, t4, a7
        mlcte32  acc1, (t0), t1
        msce32 acc1, (t4), t5

        add    t4, t4, a7
        mlcte32  acc2, (t0), t1
        msce32 acc2, (t4), t5
        
        add    t4, t4, a7
        mlcte32  acc3, (t0), t1
        msce32 acc3, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
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



matrix_lst:
        la      t0, mat_a
        li  t1,512
        la      t4,mat_dist
        li      t5,512
        li      a7,65536  

        msyncreset      tok1
        mlce32  acc0, (t0), t1
        mscte32 acc0, (t4), t5

        add    t4, t4, a7
        mlce32  acc1, (t0), t1
        mscte32 acc1, (t4), t5

        add    t4, t4, a7
        mlce32  acc2, (t0), t1
        mscte32 acc2, (t4), t5
        
        add    t4, t4, a7
        mlce32  acc3, (t0), t1
        mscte32 acc3, (t4), t5
        
        mrelease        tok1
        li a0, 1
        macquire        a0, tok1
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



check_fail:
        li      a0, 1         # 发现错误，设置错误标志为 1
        
check_done:
        ret
# void* get_matrix_c(void)
get_matrix_a:
        la a0, mat_a
        ret

