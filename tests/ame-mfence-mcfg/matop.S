// ame-mfence-mcfg: exercises Zames mcfg read/write and the Zmasync mfence.
//
// Three subtests:
//   matrix_mcfg_roundtrip - msetcfg followed by mgetcfg for every selector
//                           mcfg0..mcfg7, comparing the read value to the
//                           value that was written.
//   matrix_mcfg_behavior  - same int8 tile data interpreted first as int8 then
//                           as uint8 by reprogramming mcfg, expecting two
//                           different mmacc results from identical bytes.
//   matrix_mfence         - a known-good signed mmacc bracketed by mfence on
//                           both sides.

.text
.global matrix_init_128
.global matrix_init_64
.global matrix_restore
.global matrix_mcfg_roundtrip
.global matrix_mcfg_behavior_128
.global matrix_mcfg_behavior_64
.global matrix_mfence_128
.global matrix_mfence_64
.global matrix_nofence_128
.global matrix_nofence_64
.global mat_dist
.global mcfg_observed
.global mcfg_expected
.global mat_na

.section .data
.align 8
mat_na:
        .rept   8192            // 128 * 64 bytes, all -1 (0xff)
        .byte   -1
        .endr
mat_nb:
        .rept   8192            // 64 * 128 bytes, all -2 (0xfe)
        .byte   -2
        .endr
mat_dist:
        .rept   65536           // 4 * 128 * 128 words of 3
        .word   3
        .endr

// 8 type codes we round-trip.
mcfg_expected:
        .word   0x02            // int8
        .word   0x03            // uint8
        .word   0x04            // int32
        .word   0x09            // fp16
        .word   0x0A            // bf16
        .word   0x0C            // fp32
        .word   0x02            // int8
        .word   0x04            // int32
mcfg_observed:
        .rept   8
        .word   0
        .endr

.equ MCFG_INT8,  0x02
.equ MCFG_UINT8, 0x03
.equ MCFG_INT32, 0x04

.section .text

// Reset the per-test scratch in mat_dist to 3 (matches ame-mmacc).
matrix_restore:
        la      a1, mat_dist
        li      a2, 3
        li      a3, 16384
restore_flag:
        sw      a2, 0(a1)
        sw      a2, 4(a1)
        sw      a2, 8(a1)
        sw      a2, 12(a1)
        addi    a1, a1, 16
        addi    a3, a3, -1
        bnez    a3, restore_flag
        // Also clear the mcfg observation buffer so a stale pass doesn't mask
        // a regression on the next subtest.
        la      a1, mcfg_observed
        li      a3, 8
clear_obs:
        sw      zero, 0(a1)
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, clear_obs
        fence   rw, rw
        ret

matrix_init_128:
        // Enable matrix context (mstatus.MS=01) plus FS and VS for safety.
        lui     a0, 8194
        addiw   a0, a0, 512
        csrs    mstatus, a0

        li      a0, 128
        msettilem       a0
        msettilen       a0
        li      a0, 64
        msettilek       a0

        msettilemi      128
        msettileni      128
        msettileki      64
        ret

matrix_init_64:
        lui     a0, 8194
        addiw   a0, a0, 512
        csrs    mstatus, a0

        li      a0, 64
        msettilem       a0
        msettilen       a0
        msettilek       a0

        msettilemi      64
        msettileni      64
        msettileki      64
        ret

.macro SETCFG mcfg_sel, type_code
        li      t6, \type_code
        msetcfg \mcfg_sel, t6
.endm

// ---------------------------------------------------------------------------
// matrix_mcfg_roundtrip
//   For each selector mcfg0..mcfg7, write the corresponding entry from
//   mcfg_expected[] using msetcfg, read it back with mgetcfg, store into
//   mcfg_observed[], then compare in scalar code.
// ---------------------------------------------------------------------------
matrix_mcfg_roundtrip:
        la      t0, mcfg_expected
        la      t1, mcfg_observed

        lw      t2, 0(t0)
        msetcfg mcfg0, t2
        mgetcfg t3, mcfg0
        sw      t3, 0(t1)

        lw      t2, 4(t0)
        msetcfg mcfg1, t2
        mgetcfg t3, mcfg1
        sw      t3, 4(t1)

        lw      t2, 8(t0)
        msetcfg mcfg2, t2
        mgetcfg t3, mcfg2
        sw      t3, 8(t1)

        lw      t2, 12(t0)
        msetcfg mcfg3, t2
        mgetcfg t3, mcfg3
        sw      t3, 12(t1)

        lw      t2, 16(t0)
        msetcfg mcfg4, t2
        mgetcfg t3, mcfg4
        sw      t3, 16(t1)

        lw      t2, 20(t0)
        msetcfg mcfg5, t2
        mgetcfg t3, mcfg5
        sw      t3, 20(t1)

        lw      t2, 24(t0)
        msetcfg mcfg6, t2
        mgetcfg t3, mcfg6
        sw      t3, 24(t1)

        lw      t2, 28(t0)
        msetcfg mcfg7, t2
        mgetcfg t3, mcfg7
        sw      t3, 28(t1)

        la      t0, mcfg_expected
        la      t1, mcfg_observed
        li      t2, 8
        li      a0, 0
rt_check:
        lw      t3, 0(t0)
        lw      t4, 0(t1)
        bne     t3, t4, rt_fail
        addi    t0, t0, 4
        addi    t1, t1, 4
        addi    t2, t2, -1
        bnez    t2, rt_check
        ret
rt_fail:
        li      a0, 1
        ret

// ---------------------------------------------------------------------------
// matrix_mcfg_behavior
//   Verify that re-issuing msetcfg between operations changes the datapath
//   interpretation of the same source bytes. mat_na = -1 (= 0xff), mat_nb =
//   -2 (= 0xfe), with k=64.
// ---------------------------------------------------------------------------
matrix_mcfg_behavior_128:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 512

        msyncregreset   sync1
        mfence

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        li      a7, 65536
        add     t4, t4, a7
        SETCFG  mcfg1, MCFG_UINT8
        SETCFG  mcfg5, MCFG_INT32
        mlb     tr1, (t2), t3
        mlc     acc1, (t4), t5
        mmacc   acc1, tr1, tr0
        msc     acc1, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence   rw, rw

        la      a1, mat_dist
        li      a2, 131
        li      a3, 16384
beh_loop_a_128:
        lw      a4, 0(a1)
        bne     a4, a2, beh_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, beh_loop_a_128

        li      a2, -16253
        li      a3, 16384
beh_loop_b_128:
        lw      a4, 0(a1)
        bne     a4, a2, beh_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, beh_loop_b_128

        li      a0, 0
        ret

matrix_mcfg_behavior_64:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 256

        msyncregreset   sync1
        mfence

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        li      a7, 16384
        add     t4, t4, a7
        SETCFG  mcfg1, MCFG_UINT8
        SETCFG  mcfg5, MCFG_INT32
        mlb     tr1, (t2), t3
        mlc     acc1, (t4), t5
        mmacc   acc1, tr1, tr0
        msc     acc1, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        fence   rw, rw

        la      a1, mat_dist
        li      a2, 131
        li      a3, 4096
beh_loop_a_64:
        lw      a4, 0(a1)
        bne     a4, a2, beh_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, beh_loop_a_64

        li      a2, -16253
        li      a3, 4096
beh_loop_b_64:
        lw      a4, 0(a1)
        bne     a4, a2, beh_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, beh_loop_b_64

        li      a0, 0
        ret

beh_fail:
        li      a0, 1
        ret

// ---------------------------------------------------------------------------
// matrix_mfence
//   Bracket a known-good signed mmacc with mfence. Pre-mfence acts as the
//   scalar->matrix barrier; post-mfence acts before the scalar code reads the
//   result back.
// ---------------------------------------------------------------------------
matrix_mfence_128:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 512

        msyncregreset   sync1
        mfence

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        mfence
        fence   rw, rw

        la      a1, mat_dist
        li      a2, 131
        li      a3, 16384
fence_loop_128:
        lw      a4, 0(a1)
        bne     a4, a2, fence_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, fence_loop_128

        li      a0, 0
        ret

matrix_mfence_64:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 256

        msyncregreset   sync1
        mfence

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0
        mfence
        fence   rw, rw

        la      a1, mat_dist
        li      a2, 131
        li      a3, 4096
fence_loop_64:
        lw      a4, 0(a1)
        bne     a4, a2, fence_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, fence_loop_64

        li      a0, 0
        ret

fence_fail:
        li      a0, 1
        ret

matrix_nofence_128:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 512

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0

        // Expect 1 * (-2) * 64 + 3 = -125 if the scalar fill of mat_na to 1
        // was visible to the matrix engine. A stale view of the static-init
        // -1 would give 131, mismatching here.
        la      a1, mat_dist
        li      a2, -125
        li      a3, 16384
nofence_loop_128:
        lw      a4, 0(a1)
        bne     a4, a2, fence_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, nofence_loop_128

        li      a0, 0
        ret

matrix_nofence_64:
        la      t0, mat_na
        li      t1, 64
        la      t2, mat_nb
        li      t3, 64
        la      t4, mat_dist
        li      t5, 256

        msyncregreset   sync1

        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        mlc     acc0, (t4), t5
        mmacc   acc0, tr1, tr0
        msc     acc0, (t4), t5

        mrelease        sync1
        li      a0, 1
        macquire        sync1, a0

        la      a1, mat_dist
        li      a2, -125
        li      a3, 4096
nofence_loop_64:
        lw      a4, 0(a1)
        bne     a4, a2, fence_fail
        addi    a1, a1, 4
        addi    a3, a3, -1
        bnez    a3, nofence_loop_64

        li      a0, 0
        ret
