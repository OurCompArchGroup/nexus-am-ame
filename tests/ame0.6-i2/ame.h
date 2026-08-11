#ifndef XSAI_AM_TEST_AME_I2_H
#define XSAI_AM_TEST_AME_I2_H

#include <stdint.h>

#define AME_TILE_M 128
#define AME_TILE_K 64
#define AME_TILE_N 128

#define AME_I2_PACK 4
#define AME_A_I2_STRIDE_BYTES ((AME_TILE_K + AME_I2_PACK - 1) / AME_I2_PACK)

#define AME_MCFG_INT8     0x02UL
#define AME_MCFG_INT32    0x04UL
#define AME_MCFG_FP2PACK4 0x0dUL

#define AME_MSTATUS_ENABLE_MASK ((1UL << 25) | (1UL << 13) | (1UL << 9))

#define AME_MSYNCREGRESET(SYNC) \
  asm volatile("msyncregreset " #SYNC "\n\t" ::: "memory")

#define AME_MFENCE() \
  asm volatile("mfence\n\t" ::: "memory")

#define AME_MSETTILEM(M) \
  asm volatile("msettilem %0\n\t" : : "r"(M) : "memory")

#define AME_MSETTILEK(K) \
  asm volatile("msettilek %0\n\t" : : "r"(K) : "memory")

#define AME_MSETTILEN(N) \
  asm volatile("msettilen %0\n\t" : : "r"(N) : "memory")

#define AME_MSETCFG(MCFG, VALUE) \
  asm volatile("msetcfg " #MCFG ", %0\n\t" : : "r"(VALUE) : "memory")

#define AME_MLA(TR, SRC, STRIDE) \
  asm volatile("mla " #TR ", (%0), %1\n\t" : : "r"(SRC), "r"(STRIDE) : "memory")

#define AME_MLB(TR, SRC, STRIDE) \
  asm volatile("mlb " #TR ", (%0), %1\n\t" : : "r"(SRC), "r"(STRIDE) : "memory")

#define AME_MLC(ACC, SRC, STRIDE) \
  asm volatile("mlc " #ACC ", (%0), %1\n\t" : : "r"(SRC), "r"(STRIDE) : "memory")

#define AME_MMACC(ACC, MS2, MS1) \
  asm volatile("mmacc " #ACC ", " #MS2 ", " #MS1 "\n\t" ::: "memory")

#define AME_MSC(ACC, DST, STRIDE) \
  asm volatile("msc " #ACC ", (%0), %1\n\t" : : "r"(DST), "r"(STRIDE) : "memory")

#define AME_MRELEASE(SYNC) \
  asm volatile("mrelease " #SYNC "\n\t" ::: "memory")

#define AME_MACQUIRE(SYNC, VALUE) \
  asm volatile("macquire " #SYNC ", %0\n\t" : : "r"(VALUE) : "memory")

void ame_i2_init(void);
void ggml_ame_gemm_tile_i2_i8_i32_bT(const uint8_t *A, const int8_t *B, int32_t *C);

#endif
