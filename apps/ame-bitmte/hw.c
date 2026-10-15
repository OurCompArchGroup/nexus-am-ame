#include "bitmte.h"
#include <stdbool.h>

/* Matrix registers are literal operands; only addresses/strides use GPRs.
 * No compiler-managed matrix values cross this function's C interface.
 */
#define LOAD(OP, REG, BASE, STRIDE) \
  __asm__ volatile(OP " " REG ", (%0), %1" :: "r"(BASE), "r"(STRIDE) : "memory")
#define MMA_CASE(S, ACC, B, A) \
  case S: __asm__ volatile("mmacc " ACC ", " B ", " A ", " #S ::: "memory"); break
#define DEFINE_MMA(NAME, ACC, B, A) \
  static inline void NAME(unsigned slice) { \
    switch (slice) { \
      MMA_CASE(0, ACC, B, A); MMA_CASE(1, ACC, B, A); \
      MMA_CASE(2, ACC, B, A); MMA_CASE(3, ACC, B, A); \
      MMA_CASE(4, ACC, B, A); MMA_CASE(5, ACC, B, A); \
      MMA_CASE(6, ACC, B, A); MMA_CASE(7, ACC, B, A); \
      default: __builtin_unreachable(); \
    } \
  }
DEFINE_MMA(mma00, "acc0", "tr1", "tr0")
DEFINE_MMA(mma01, "acc1", "tr3", "tr0")
DEFINE_MMA(mma10, "acc2", "tr1", "tr2")
DEFINE_MMA(mma11, "acc3", "tr3", "tr2")

void bitmte_init(void) {
  __asm__ volatile("csrs mstatus, %0\n\tminit" :: "r"(1UL << 25) : "memory");
}

static void configure(bitmte_mode mode) {
  __asm__ volatile(
    "msettilem %0\n\tmsettilen %0\n\tmsettilek %1\n\t"
    "msetcfg mcfg0, %2\n\tmsetcfg mcfg2, %2\n\t"
    "msetcfg mcfg1, %3\n\tmsetcfg mcfg3, %3\n\t"
    "msetcfg mcfg4, %4\n\tmsetcfg mcfg5, %4\n\t"
    "msetcfg mcfg6, %4\n\tmsetcfg mcfg7, %4\n\tcsrw 0x80a, zero"
    :: "r"(128UL), "r"((unsigned long)bitmte_k_slot(mode)),
       "r"(bitmte_a_bits(mode) == 4 ? 0UL : 2UL),
       "r"(bitmte_b_bits(mode) == 2 ? 13UL : 15UL), "r"(4UL) : "memory");
}

int bitmte_gemm(uint32_t *c, const uint8_t *a, const uint8_t *bt,
                size_t m, size_t n, size_t k, bitmte_mode mode) {
  if ((unsigned)mode > BMTE_I4_BINARY || !a || !bt || !c || !m || !n || !k) return -1;
  if (m % BMTE_TILE || n % BMTE_TILE || k % bitmte_k_resident(mode)) return -1;
  if (((uintptr_t)a | (uintptr_t)bt | (uintptr_t)c) % BMTE_ROW_BYTES) return -1;
  const size_t a_stride = k / (8 / bitmte_a_bits(mode));
  const size_t b_stride = k / (8 / bitmte_b_bits(mode));
  if (n > SIZE_MAX / sizeof(*c) || m > SIZE_MAX / (n * sizeof(*c)) ||
      m > SIZE_MAX / a_stride || n > SIZE_MAX / b_stride) return -1;
  const size_t c_stride = n * sizeof(*c);
  const unsigned slices = bitmte_slices(mode), kres = bitmte_k_resident(mode);

  configure(mode);
  __asm__ volatile("msyncregreset sync0\n\tmfence" ::: "memory");
  for (size_t mi = 0; mi < m; mi += 2 * BMTE_TILE) {
    const bool row1 = m - mi > BMTE_TILE;
    for (size_t nj = 0; nj < n; nj += 2 * BMTE_TILE) {
      const bool col1 = n - nj > BMTE_TILE;
      __asm__ volatile("mzero acc0" ::: "memory");
      if (col1) __asm__ volatile("mzero acc1" ::: "memory");
      if (row1) __asm__ volatile("mzero acc2" ::: "memory");
      if (row1 && col1) __asm__ volatile("mzero acc3" ::: "memory");

      /* Four C tiles stay resident across all K blocks. Each A serves two N
       * tiles, each packed B serves two M tiles and all of its K slices.
       */
      for (size_t kk = 0; kk < k; kk += kres) {
        const uint8_t *ap = a + mi * a_stride + kk / (8 / bitmte_a_bits(mode));
        const uint8_t *bp = bt + nj * b_stride + kk / (8 / bitmte_b_bits(mode));
        LOAD("mlb", "tr1", bp, b_stride);
        if (col1) LOAD("mlb", "tr3", bp + BMTE_TILE * b_stride, b_stride);
        LOAD("mla", "tr0", ap, a_stride);
        if (row1) LOAD("mla", "tr2", ap + BMTE_TILE * a_stride, a_stride);

        for (unsigned s = 0; s < slices; ++s) {
          mma00(s);
          if (col1) mma01(s);
          /* Queue A0's next slice after both consumers, before A1's compute.
           * The existing scoreboard enforces WAR readDone; the source memory
           * is immutable, so no CPU scratch-buffer synchronization is needed.
           */
          if (s + 1 < slices) LOAD("mla", "tr0", ap + (s + 1) * BMTE_ROW_BYTES, a_stride);
          if (row1) {
            mma10(s);
            if (col1) mma11(s);
            if (s + 1 < slices)
              LOAD("mla", "tr2", ap + BMTE_TILE * a_stride + (s + 1) * BMTE_ROW_BYTES, a_stride);
          }
        }
      }
      LOAD("msc", "acc0", c + mi * n + nj, c_stride);
      if (col1) LOAD("msc", "acc1", c + mi * n + nj + BMTE_TILE, c_stride);
      if (row1) LOAD("msc", "acc2", c + (mi + BMTE_TILE) * n + nj, c_stride);
      if (row1 && col1) LOAD("msc", "acc3", c + (mi + BMTE_TILE) * n + nj + BMTE_TILE, c_stride);
    }
  }
  /* A single completion wait drains every output store, including edge tiles.
   * mfence above is CPU->matrix visibility, not a replacement for this wait.
   */
  __asm__ volatile("mrelease sync0\n\tmacquire sync0, %0" :: "r"(1UL) : "memory");
  return 0;
}
