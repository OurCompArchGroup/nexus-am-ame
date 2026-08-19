// Compute-based A/B transpose-load probe.
//
// The first pair of loads uses the requested element width.  Once the loads
// have drained, the same raw MatrixReg bits are interpreted as int8.  An
// identity int8 multiply exposes those bytes through the supported CReg store
// instruction (msc), avoiding msa/msb.

.text
.global matrix_init
.global transpose_probe
.extern a_source
.extern b_source
.extern identity
.extern a_result
.extern b_result

.equ MCFG_INT4,  0x00
.equ MCFG_INT8,  0x02
.equ MCFG_INT32, 0x04
.equ MCFG_FP16, 0x09

.macro SETCFG mcfg_sel, type_code
        li      t6, \type_code
        msetcfg \mcfg_sel, t6
.endm

.macro WAIT_MATRIX threshold
        mrelease        sync1
        li      a0, \threshold
        macquire        sync1, a0
        fence   iorw, iorw
.endm

#if TRANSPOSE_PROBE_MASK == 1
.equ ORIGINAL_MCFG, MCFG_INT8
.equ ORIGINAL_K, 64
.equ SOURCE_STRIDE, 128
#elif TRANSPOSE_PROBE_MASK == 2
.equ ORIGINAL_MCFG, MCFG_FP16
.equ ORIGINAL_K, 32
.equ SOURCE_STRIDE, 256
#elif TRANSPOSE_PROBE_MASK == 4
.equ ORIGINAL_MCFG, MCFG_INT32
.equ ORIGINAL_K, 16
.equ SOURCE_STRIDE, 512
#elif TRANSPOSE_PROBE_MASK == 8
.equ ORIGINAL_MCFG, MCFG_INT4
.equ ORIGINAL_K, 127
.equ SOURCE_STRIDE, 64
#else
#error "TRANSPOSE_PROBE_MASK must be 1, 2, 4, or 8"
#endif

.equ LOAD_MAJOR_DIM, 128
.equ COMPUTE_K, 64
.equ COMPUTE_N, 128
.equ RESULT_STRIDE, (COMPUTE_N * 4)

#if TRANSPOSE_PROBE_MASK == 8
.equ TRANSPOSE_DONE, 2
.equ A_PROBE_DONE, 3
.equ B_PROBE_DONE, 4
#else
.equ TRANSPOSE_DONE, 1
.equ A_PROBE_DONE, 2
.equ B_PROBE_DONE, 3
#endif

matrix_init:
        // Enable matrix state (and the FP/vector state bits used by AM).
        lui     a0, 8194
        addiw   a0, a0, 512
        csrs    mstatus, a0
        ret

transpose_probe:
        fence   iorw, iorw
        mfence
        msyncregreset   sync1

#if TRANSPOSE_PROBE_MASK == 8
        // Seed every raw byte before the odd-K e4 transpose.  The transpose
        // must preserve byte 63's known high nibble while replacing its low
        // nibble.  a_result/b_result are only scratch sources at this point.
        li      t0, LOAD_MAJOR_DIM
        msettilem       t0
        li      t0, LOAD_MAJOR_DIM
        msettilen       t0
        li      t0, COMPUTE_K
        msettilek       t0
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        la      t0, a_result
        li      t1, COMPUTE_K
        la      t2, b_result
        li      t3, COMPUTE_K
        mla     tr0, (t0), t1
        mlb     tr1, (t2), t3
        WAIT_MATRIX 1
#endif

        // Full-width original load shape.  In transpose mode the source has
        // ORIGINAL_K rows and LOAD_MAJOR_DIM elements per row; the resulting
        // register has LOAD_MAJOR_DIM rows and ORIGINAL_K elements per row.
        li      t0, LOAD_MAJOR_DIM
        msettilem       t0
        li      t0, LOAD_MAJOR_DIM
        msettilen       t0
        li      t0, ORIGINAL_K
        msettilek       t0
        SETCFG  mcfg0, ORIGINAL_MCFG
        SETCFG  mcfg1, ORIGINAL_MCFG

        la      t0, a_source
        li      t1, SOURCE_STRIDE
        la      t2, b_source
        li      t3, SOURCE_STRIDE
        fence   iorw, iorw
        mlat    tr0, (t0), t1
        mlbt    tr1, (t2), t3
        WAIT_MATRIX TRANSPOSE_DONE

        // Retag the loaded raw bits as int8 and configure the supported
        // int8*int8->int32 observation path.
        SETCFG  mcfg0, MCFG_INT8
        SETCFG  mcfg1, MCFG_INT8
        SETCFG  mcfg2, MCFG_INT8
        SETCFG  mcfg4, MCFG_INT32

        // -------- A transpose: A_raw * I -> a_result --------
        li      t0, COMPUTE_N
        msettilem       t0
        li      t0, COMPUTE_N
        msettilen       t0
        li      t0, COMPUTE_K
        msettilek       t0
        la      t0, identity
        li      t1, COMPUTE_K
        fence   iorw, iorw
        mlb     tr2, (t0), t1
        mzero   acc0
        mmacc   acc0, tr2, tr0
        la      t0, a_result
        li      t1, RESULT_STRIDE
        msc     acc0, (t0), t1
        WAIT_MATRIX A_PROBE_DONE

        // -------- B transpose: I * B_raw -> b_result --------
        // Load all 128 identity rows so the auxiliary mla overwrites the
        // complete tile register observed by difftest.
        li      t0, COMPUTE_N
        msettilem       t0
        li      t0, COMPUTE_N
        msettilen       t0
        li      t0, COMPUTE_K
        msettilek       t0
        la      t0, identity
        li      t1, COMPUTE_K
        fence   iorw, iorw
        mla     tr2, (t0), t1
        mzero   acc0
        mmacc   acc0, tr1, tr2
        la      t0, b_result
        li      t1, RESULT_STRIDE
        msc     acc0, (t0), t1
        WAIT_MATRIX B_PROBE_DONE
        mfence
        ret
