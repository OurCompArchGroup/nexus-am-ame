#include "ame.h"

void ame_init(void) {
  const unsigned long mstatus_mask = AME_MSTATUS_ENABLE_MASK;
  const unsigned long m = AME_TILE_M;
  const unsigned long k = AME_TILE_K;
  const unsigned long n = AME_TILE_N;
  const unsigned long cfg_i8 = AME_MCFG_INT8;
  const unsigned long cfg_i32 = AME_MCFG_INT32;

  asm volatile(
      "csrs mstatus, %[mask]\n\t"
      "csrwi vcsr, 0\n\t"
      :
      : [mask] "r"(mstatus_mask)
      : "memory");

  AME_MSETTILEM(m);
  AME_MSETTILEK(k);
  AME_MSETTILEN(n);
  AME_MSETCFG(mcfg0, cfg_i8);
  AME_MSETCFG(mcfg1, cfg_i8);
  AME_MSETCFG(mcfg2, cfg_i8);
  AME_MSETCFG(mcfg3, cfg_i8);
  AME_MSETCFG(mcfg4, cfg_i32);
  AME_MSETCFG(mcfg5, cfg_i32);
  AME_MSETCFG(mcfg6, cfg_i32);
  AME_MSETCFG(mcfg7, cfg_i32);
}

void ggml_ame_gemm_tile_i8_i32_bT(const int8_t *A, const int8_t *B, int32_t *C) {
  const unsigned long a_stride = AME_TILE_K * sizeof(int8_t);
  const unsigned long b_stride = AME_TILE_K * sizeof(int8_t);
  const unsigned long c_stride = AME_TILE_N * sizeof(int32_t);
  const unsigned long sync_done = 1;

  AME_MSYNCREGRESET(sync1);
  AME_MFENCE();
  AME_MLA(tr0, A, a_stride);
  AME_MLB(tr1, B, b_stride);
  AME_MLC(acc0, C, c_stride);
  AME_MMACC(acc0, tr1, tr0);
  AME_MSC(acc0, C, c_stride);
  AME_MRELEASE(sync1);
  AME_MACQUIRE(sync1, sync_done);
  AME_MFENCE();
}
