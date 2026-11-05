#include <klib.h>
#include <stdbool.h>
#include <stdint.h>

/* One binary is one fresh-process GEMM with a host-loaded input fixture.
 * Numerical operands are shared; packing and scheduling vary across modes.
 */
#if PPA_MODE == 0
#define A_BITS 8u
#define B_BITS 8u
#define A_CFG 2ul
#define B_CFG 2ul
#define C_CFG 4ul
#elif PPA_MODE == 1
#define A_BITS 16u
#define B_BITS 16u
#define A_CFG 10ul
#define B_CFG 10ul
#define C_CFG 12ul
#elif PPA_MODE == 2 || PPA_MODE == 4
#define A_BITS 8u
#define A_CFG 2ul
#define C_CFG 4ul
#if PPA_MODE == 2
#define B_BITS 2u
#define B_CFG 13ul
#else
#define B_BITS 1u
#define B_CFG 15ul
#endif
#else
#define A_BITS 4u
#define A_CFG 0ul
#define C_CFG 4ul
#if PPA_MODE == 3
#define B_BITS 2u
#define B_CFG 13ul
#else
#define B_BITS 1u
#define B_CFG 15ul
#endif
#endif

#define TILE 128u
#define ROW_BYTES 64u
#define K_SLOT (512u / A_BITS)
#define K_RESIDENT (512u / B_BITS)
#define SLICES (K_RESIDENT / K_SLOT)
#define A_STRIDE ((size_t)PPA_K * A_BITS / 8u)
#define B_STRIDE ((size_t)PPA_K * B_BITS / 8u)
#define C_STRIDE ((size_t)PPA_N * 4u)

_Static_assert(PPA_MODE >= 0 && PPA_MODE <= 5, "invalid mode");
_Static_assert(PPA_CHECK == 0 || PPA_CHECK == 1, "CHECK must be 0 or 1");
_Static_assert(PPA_M > 0 && PPA_M % TILE == 0, "M must be a positive multiple of128");
_Static_assert(PPA_N > 0 && PPA_N % TILE == 0, "N must be a positive multiple of128");
_Static_assert(PPA_K > 0 && PPA_K % 512u == 0, "common K must be a positive multiple of512");
/* BF16 operands are integral; every reduction stays an exact FP32 integer. */
_Static_assert(PPA_K <= 2097152u, "K exceeds the common exact-FP32 oracle domain");
#if PPA_SCHEDULE == 4
_Static_assert(PPA_MODE == 0 || PPA_MODE == 1, "mpair-bdefer is ordinary MTE only");
_Static_assert((PPA_M == 256u || PPA_M == 512u) && PPA_N == 512u &&
               (PPA_K == 512u || PPA_K == 1024u) && PPA_SEED == 0x20261103u,
               "mpair-bdefer is frozen to the C0/CK/CG/CKG finite shape set");
#endif

#include "expected.h"

static uint32_t output[PPA_M][PPA_N]
  __attribute__((section(".data"), aligned(64))) = {0};

#if PPA_CHECK && PPA_MODE == 1
/* Exact integer-to-FP32 encoding, independent of matrix operations and scalar FP. */
static uint32_t fp32_integer(int32_t value) {
  if (!value) return 0;
  const uint32_t sign = value < 0 ? UINT32_C(0x80000000) : 0;
  uint32_t magnitude = value < 0 ? (uint32_t)(-(int64_t)value) : (uint32_t)value;
  unsigned top = 31u - (unsigned)__builtin_clz(magnitude);
  uint32_t significand = top <= 23u ? magnitude << (23u - top) : magnitude >> (top - 23u);
  return sign | ((top + 127u) << 23) | (significand & UINT32_C(0x7fffff));
}
#endif

#define LOAD(OP, REG, PTR, STRIDE) \
  __asm__ volatile(OP " " REG ", (%0), %1" :: "r"(PTR), "r"(STRIDE) : "memory")
#define CASE_MMA(S, C, B, A) \
  case S: __asm__ volatile("mmacc " C ", " B ", " A ", " #S ::: "memory"); break
#define DEFINE_MMA(FN, C, B, A) \
  static inline void FN(unsigned slice) { \
    switch (slice) { \
      CASE_MMA(0, C, B, A); CASE_MMA(1, C, B, A); \
      CASE_MMA(2, C, B, A); CASE_MMA(3, C, B, A); \
      CASE_MMA(4, C, B, A); CASE_MMA(5, C, B, A); \
      CASE_MMA(6, C, B, A); CASE_MMA(7, C, B, A); \
      default: __builtin_unreachable(); \
    } \
  }
#define DEFINE_C_MMAS(N) \
  DEFINE_MMA(mma##N##_20, "acc" #N, "tr2", "tr0") \
  DEFINE_MMA(mma##N##_21, "acc" #N, "tr2", "tr1") \
  DEFINE_MMA(mma##N##_30, "acc" #N, "tr3", "tr0") \
  DEFINE_MMA(mma##N##_31, "acc" #N, "tr3", "tr1")
DEFINE_C_MMAS(0)
DEFINE_C_MMAS(1)
DEFINE_C_MMAS(2)
DEFINE_C_MMAS(3)
#define DISPATCH_C(N) \
  case N: \
    if (b) { if (a) mma##N##_31(slice); else mma##N##_30(slice); } \
    else { if (a) mma##N##_21(slice); else mma##N##_20(slice); } \
    break
static inline void mmacc(unsigned c, unsigned b, unsigned a, unsigned slice) {
  switch (c) { DISPATCH_C(0); DISPATCH_C(1); DISPATCH_C(2); DISPATCH_C(3); }
}
static inline void load_a(unsigned reg, const uint8_t *ptr) {
  if (reg) LOAD("mla", "tr1", ptr, A_STRIDE);
  else LOAD("mla", "tr0", ptr, A_STRIDE);
}
static inline void load_b(unsigned reg, const uint8_t *ptr) {
  if (reg) LOAD("mlb", "tr3", ptr, B_STRIDE);
  else LOAD("mlb", "tr2", ptr, B_STRIDE);
}
static inline void zero_acc(unsigned c) {
  switch (c) {
    case 0: __asm__ volatile("mzero acc0" ::: "memory"); break;
    case 1: __asm__ volatile("mzero acc1" ::: "memory"); break;
    case 2: __asm__ volatile("mzero acc2" ::: "memory"); break;
    case 3: __asm__ volatile("mzero acc3" ::: "memory"); break;
  }
}
static inline void store_acc(unsigned c, uint32_t *ptr) {
  switch (c) {
    case 0: LOAD("msc", "acc0", ptr, C_STRIDE); break;
    case 1: LOAD("msc", "acc1", ptr, C_STRIDE); break;
    case 2: LOAD("msc", "acc2", ptr, C_STRIDE); break;
    case 3: LOAD("msc", "acc3", ptr, C_STRIDE); break;
  }
}

typedef struct {
  uint32_t *first, *second;
  unsigned c, count, issued;
} Pending;

/* Keep older output stores behind the next block's first real arithmetic work.
 * These are issue phases, not extra waits; CML/scoreboard owns completion.
 */
#if PPA_SCHEDULE != 3
static inline void pending_stores(Pending *p, unsigned compute_pairs) {
  if (!p->count) return;
  if (!p->issued && compute_pairs >= 1u) {
    store_acc(p->c, p->first);
    p->issued = 1u;
  }
  if (p->count == 2u && p->issued == 1u && compute_pairs >= 2u) {
    store_acc(p->c + 1u, p->second);
    p->issued = 2u;
  }
}

#endif

static void configure(void) {
  __asm__ volatile("csrs mstatus, %0\n\tcsrwi vcsr, 0\n\tminit\n\t"
    "msettilem %1\n\tmsettilen %1\n\tmsettilek %2\n\t"
    "msetcfg mcfg0, %3\n\tmsetcfg mcfg1, %3\n\t"
    "msetcfg mcfg2, %4\n\tmsetcfg mcfg3, %4\n\t"
    "msetcfg mcfg4, %5\n\tmsetcfg mcfg5, %5\n\t"
    "msetcfg mcfg6, %5\n\tmsetcfg mcfg7, %5\n\t"
    "csrw 0x80a, zero\n\tmsyncregreset sync0\n\tmfence"
    :: "r"((1ul << 25) | (1ul << 13) | (1ul << 9)), "r"(128ul), "r"((unsigned long)K_SLOT),
       "r"(A_CFG), "r"(B_CFG), "r"(C_CFG) : "memory");
}

static inline uint64_t cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value) :: "memory");
  return value;
}

#if PPA_SCHEDULE == 3
/* Same TR roles/configuration as the existing pair schedules:
 * TR0/1 hold A0/A1, TR2/3 hold B0/B1; ACC0/1/2/3 are C00/01/10/11.
 */
typedef struct {
  size_t mi, nj;
  bool valid, row1, col1;
} QuadSite;

static inline QuadSite quad_site(size_t mi, size_t nj) {
  return (QuadSite){mi, nj, mi < PPA_M && nj < PPA_N,
                    mi + TILE < PPA_M, nj + TILE < PPA_N};
}
static inline QuadSite quad_next(QuadSite current) {
  size_t mi = current.mi, nj = current.nj + 2u * TILE;
  if (nj >= PPA_N) { nj = 0; mi += 2u * TILE; }
  return quad_site(mi, nj);
}
static inline void quad_load_a(unsigned row, QuadSite site, size_t k) {
  load_a(row, &packed_a[site.mi + row * TILE][k * A_BITS / 8u]);
}
static inline void quad_load_b(unsigned column, QuadSite site, size_t k) {
  load_b(column, &packed_b[site.nj + column * TILE][k * B_BITS / 8u]);
}
static inline void quad_store(unsigned row, unsigned column, QuadSite site) {
  store_acc(row * 2u + column, &output[site.mi + row * TILE][site.nj + column * TILE]);
}

/* Seed every active output with its first Kslot exactly once. For a rolling
 * successor, A0/B0 were queued in the previous group's compact final tail.
 * The next group never requires all four old stores to finish together.
 */
static inline void quad_seed(QuadSite site, bool bootstrap) {
  if (bootstrap) { quad_load_a(0, site, 0); quad_load_b(0, site, 0); }
  zero_acc(0);
  mmacc(0, 0, 0, 0);                         /* F00 */
  if (site.row1) quad_load_a(1, site, 0);    /* after the old F11 */
  if (site.col1) quad_load_b(1, site, 0);    /* after the old F11 */
  if (site.col1) {
    zero_acc(1);
    mmacc(1, 1, 0, 0);                       /* F01, last A0 reader */
  }
  if (K_SLOT < PPA_K) quad_load_a(0, site, K_SLOT);
  if (site.row1) {
    zero_acc(2);
    mmacc(2, 0, 1, 0);                       /* F10, last B0 row reader */
  }
  if (SLICES == 1u && K_RESIDENT < PPA_K) quad_load_b(0, site, K_RESIDENT);
  if (site.row1 && site.col1) {
    zero_acc(3);
    mmacc(3, 1, 1, 0);                       /* F11 */
  }
  if (site.row1 && K_SLOT < PPA_K) quad_load_a(1, site, K_SLOT);
  if (site.col1 && SLICES == 1u && K_RESIDENT < PPA_K)
    quad_load_b(1, site, K_RESIDENT);
}

static inline void quad_finish(QuadSite site, QuadSite next) {
  /* quad_seed already issued Kslot0 for each active ACC. */
  for (size_t k = K_SLOT; k < PPA_K; k += K_SLOT) {
    const unsigned slice = (unsigned)((k / K_SLOT) % SLICES);
    const size_t next_k = k + K_SLOT;
    const bool last = next_k == PPA_K;
    const bool resident_end = slice + 1u == SLICES;
    mmacc(0, 0, 0, slice);                  /* F00 */
    if (last) quad_store(0, 0, site);        /* S0 */
    if (site.col1) {
      mmacc(1, 1, 0, slice);                /* F01 */
      if (last) quad_store(0, 1, site);      /* S1 */
    }
    if (!last) quad_load_a(0, site, next_k);
    else if (next.valid) quad_load_a(0, next, 0);
    if (site.row1) {
      mmacc(2, 0, 1, slice);                /* F10 */
      if (last) quad_store(1, 0, site);      /* S2 */
    }
    if (!last && resident_end) quad_load_b(0, site, next_k);
    else if (last && next.valid) quad_load_b(0, next, 0);
    if (site.row1 && site.col1) {
      mmacc(3, 1, 1, slice);                /* F11 */
      if (last) quad_store(1, 1, site);      /* S3 */
    }
    if (!last && site.row1) quad_load_a(1, site, next_k);
    if (!last && resident_end && site.col1) quad_load_b(1, site, next_k);
  }
}
#endif

#if PPA_SCHEDULE != 3
#if PPA_SCHEDULE == 0 || PPA_SCHEDULE == 2
static void issue_block(size_t mi, size_t nj, unsigned c, Pending *pending) {
  const unsigned count = nj + TILE < PPA_N ? 2u : 1u;
  zero_acc(c);
  if (count == 2u) zero_acc(c + 1u);
  const uint8_t *a = &packed_a[mi][0];
  const uint8_t *b = &packed_b[nj][0];
  load_a(0, a);
  load_b(0, b);
  if (count == 2u) load_b(1, b + TILE * B_STRIDE);
  unsigned b0 = 0, a0 = 0, pairs = 0;
  for (size_t kr = 0; kr < PPA_K; kr += K_RESIDENT) {
#pragma clang loop unroll(full)
    for (unsigned s = 0; s < SLICES; ++s) {
      const size_t next_k = kr + (s + 1u) * K_SLOT;
      mmacc(c, b0, a0, s);
      if (next_k < PPA_K) load_a(a0 ^ 1u, a + next_k * A_BITS / 8u);
      if (count == 2u) {
#if PPA_SCHEDULE == 2
        /* Keep each B register assigned to its output column. */
        if (s + 1u == SLICES && next_k < PPA_K)
          load_b(b0, b + next_k * B_BITS / 8u);
        mmacc(c + 1u, b0 ^ 1u, a0, s);
        if (s + 1u == SLICES && next_k < PPA_K)
          load_b(b0 ^ 1u, b + TILE * B_STRIDE + next_k * B_BITS / 8u);
#else
        /* After the last N0 consumer, its B register becomes next N1. */
        if (s + 1u == SLICES && next_k < PPA_K)
          load_b(b0, b + TILE * B_STRIDE + next_k * B_BITS / 8u);
        mmacc(c + 1u, b0 ^ 1u, a0, s);
        if (s + 1u == SLICES && next_k < PPA_K)
          load_b(b0 ^ 1u, b + next_k * B_BITS / 8u);
#endif
      } else if (s + 1u == SLICES && next_k < PPA_K) {
        load_b(b0 ^ 1u, b + next_k * B_BITS / 8u);
      }
      a0 ^= 1u;
      pending_stores(pending, ++pairs);
    }
#if PPA_SCHEDULE == 2
    /* A singleton keeps the existing unused-B-register prefetch path. */
    if (count == 1u) b0 ^= 1u;
#else
    b0 ^= 1u;
#endif
  }
  pending_stores(pending, 2u);
  *pending = (Pending){&output[mi][nj], &output[mi][nj + (count == 2u ? TILE : 0u)], c, count, 0};
}
#else
static void issue_block(size_t mi, size_t nj, unsigned c, Pending *pending) {
  const unsigned count = mi + TILE < PPA_M ? 2u : 1u;
  zero_acc(c);
  if (count == 2u) zero_acc(c + 1u);
  const uint8_t *a = &packed_a[mi][0];
  const uint8_t *b = &packed_b[nj][0];
  load_a(0, a);
  if (count == 2u) load_a(1, a + TILE * A_STRIDE);
  load_b(0, b);
  unsigned b0 = 0, pairs = 0;
  for (size_t kr = 0; kr < PPA_K; kr += K_RESIDENT) {
#pragma clang loop unroll(full)
    for (unsigned s = 0; s < SLICES; ++s) {
      const size_t next_k = kr + (s + 1u) * K_SLOT;
      mmacc(c, b0, 0, s);
      if (next_k < PPA_K) load_a(0, a + next_k * A_BITS / 8u);
#if PPA_SCHEDULE == 4
      /* Ordinary MTE finite challenger: let current TR2/3 compute precede
       * the unrelated other-B-register preload for the next K resident. */
      if (count == 2u)
        mmacc(c + 1u, b0, 1, s);
      if (s == 0u && kr + K_RESIDENT < PPA_K)
        load_b(b0 ^ 1u, b + (kr + K_RESIDENT) * B_BITS / 8u);
      if (count == 2u && next_k < PPA_K)
        load_a(1, a + TILE * A_STRIDE + next_k * A_BITS / 8u);
#else
      /* Independent B buffer gets a whole resident vector ahead of use. */
      if (s == 0u && kr + K_RESIDENT < PPA_K)
        load_b(b0 ^ 1u, b + (kr + K_RESIDENT) * B_BITS / 8u);
      if (count == 2u) {
        mmacc(c + 1u, b0, 1, s);
        if (next_k < PPA_K)
          load_a(1, a + TILE * A_STRIDE + next_k * A_BITS / 8u);
      }
#endif
      pending_stores(pending, ++pairs);
    }
    b0 ^= 1u;
  }
  pending_stores(pending, 2u);
  *pending = (Pending){&output[mi][nj], &output[mi + (count == 2u ? TILE : 0u)][nj], c, count, 0};
}
#endif

#endif

static __attribute__((noinline)) uint64_t run_kernel(void) {
#if PPA_SCHEDULE != 3
  Pending pending = {0};
  unsigned c = 0;
#endif
  __asm__ volatile(".global ppa_roi_begin\nppa_roi_begin:" ::: "memory");
  const uint64_t begin = cycles();
#if PPA_SCHEDULE == 3
  QuadSite site = quad_site(0, 0);
  quad_seed(site, true);
  while (site.valid) {
    const QuadSite next = quad_next(site);
    quad_finish(site, next);
    if (next.valid) quad_seed(next, false);
    site = next;
  }
#else
#if PPA_SCHEDULE == 0 || PPA_SCHEDULE == 2
  for (size_t mi = 0; mi < PPA_M; mi += TILE)
    for (size_t nj = 0; nj < PPA_N; nj += 2u * TILE) {
#else
  for (size_t mi = 0; mi < PPA_M; mi += 2u * TILE)
    for (size_t nj = 0; nj < PPA_N; nj += TILE) {
#endif
      issue_block(mi, nj, c, &pending);
      c ^= 2u;
    }
  pending_stores(&pending, 2u);
#endif
  __asm__ volatile("mrelease sync0\n\tmacquire sync0, %0\n\tmfence" :: "r"(1ul) : "memory");
  const uint64_t end = cycles();
  __asm__ volatile(".global ppa_roi_end\nppa_roi_end:" ::: "memory");
  return end - begin;
}

#if PPA_CHECK
static int verify(uint32_t *numerical_hash) {
  uint32_t hash = UINT32_C(2166136261);
  for (size_t i = 0; i < PPA_M; ++i) {
    for (size_t j = 0; j < PPA_N; ++j) {
      const int32_t expected = ppa_expected[i * PPA_N + j];
#if PPA_MODE == 1
      const uint32_t raw = fp32_integer(expected);
#else
      const uint32_t raw = (uint32_t)expected;
#endif
      if (output[i][j] != raw) {
        printf("PPA_FAIL row=%u col=%u got=%08x expected=%08x\n", (unsigned)i, (unsigned)j, output[i][j], raw);
        return 1;
      }
      hash = (hash ^ (uint32_t)expected) * UINT32_C(16777619);
    }
  }
  *numerical_hash = hash;
  return hash == PPA_ORACLE_RESULT_HASH ? 0 : 1;
}

#endif

int main(const char *args) {
  (void)args;
  const uint32_t input_hash = PPA_ORACLE_INPUT_HASH;
  const uint64_t mt = PPA_M / TILE, nt = PPA_N / TILE;
  const uint64_t tasks = mt * nt * (PPA_K / K_SLOT);
#if PPA_SCHEDULE == 3
  const uint64_t a_loads = mt * ((nt + 1u) / 2u) * (PPA_K / K_SLOT);
  const uint64_t b_loads = ((mt + 1u) / 2u) * nt * (PPA_K / K_RESIDENT);
#elif PPA_SCHEDULE == 0 || PPA_SCHEDULE == 2
  const uint64_t a_loads = mt * ((nt + 1u) / 2u) * (PPA_K / K_SLOT);
  const uint64_t b_loads = mt * nt * (PPA_K / K_RESIDENT);
#else
  const uint64_t a_loads = mt * nt * (PPA_K / K_SLOT);
  const uint64_t b_loads = ((mt + 1u) / 2u) * nt * (PPA_K / K_RESIDENT);
#endif
  printf("PPA_BEGIN mode=%s schedule=%s M=%u N=%u K=%u seed=%08x input_hash=%08x input_hash_source=host_fixture kslot=%u kresident=%u slices=%u check=%u\n",
         PPA_MODE_NAME, PPA_SCHEDULE_NAME, PPA_M, PPA_N, PPA_K, (unsigned)PPA_SEED, input_hash, K_SLOT, K_RESIDENT, SLICES, PPA_CHECK);
  configure();
  const uint64_t elapsed = run_kernel();
#if PPA_CHECK
  uint32_t numerical_hash;
  if (verify(&numerical_hash)) return 1;
  printf("PPA_PASS"
#else
  printf("PPA_DONE"
#endif
         " mode=%s schedule=%s M=%u N=%u K=%u seed=%08x input_hash=%08x input_hash_source=host_fixture"
#if PPA_CHECK
         " numerical_hash=%08x"
#endif
         " check=%u elements=%llu cycles=%llu ops=%llu mmacc=%llu aload=%llu bload=%llu mzero=%llu cstore=%llu release=1 acquire=1\n",
         PPA_MODE_NAME, PPA_SCHEDULE_NAME, PPA_M, PPA_N, PPA_K, (unsigned)PPA_SEED,
         input_hash,
#if PPA_CHECK
         numerical_hash,
#endif
         PPA_CHECK, (unsigned long long)(mt * nt * TILE * TILE),
         (unsigned long long)elapsed, (unsigned long long)(2u * (uint64_t)PPA_M * PPA_N * PPA_K),
         (unsigned long long)tasks, (unsigned long long)a_loads, (unsigned long long)b_loads,
         (unsigned long long)(mt * nt), (unsigned long long)(mt * nt));
  return 0;
}
