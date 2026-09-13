/* Full-tile GEMM; extracted from the measured BitVLA OpenLLC schedule. */
#define BENCH_M_GROUP_TILES 2
#ifndef BENCH_VERIFY_FULL
#define BENCH_VERIFY_FULL 0
#endif
#include <am.h>
#include <klib.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef BENCH_M
#define BENCH_M 256u
#endif

#ifndef BENCH_N
#define BENCH_N 256u
#endif

#ifndef BENCH_K
#define BENCH_K 128u
#endif

#ifndef BENCH_K_BATCH_TILES
#define BENCH_K_BATCH_TILES 0u
#endif

/*
 * The full two-N-panel path can use fixed A/B/C register assignments.  Keep
 * the singleton N tail on the generic path: it has one fewer B panel and
 * must not issue a speculative second load or MMACC.
 */

#define STRINGIFY_IMPL(value) #value
#define STRINGIFY(value) STRINGIFY_IMPL(value)

#define BENCH_MODE_NAME "fp2pack4"
#define K_BATCH_MODE_NAME "unbounded"
#define SCHEDULE_NAME "mgroup" STRINGIFY(BENCH_M_GROUP_TILES) \
    "_npair_abpingpong_crossb_n1first_bfirst_staticfull_deferred_store"

_Static_assert(BENCH_M > 0 && BENCH_N > 0 && BENCH_K > 0,
               "M, N and K must be positive");
_Static_assert(BENCH_M % 128 == 0 && BENCH_N % 128 == 0,
               "M and N must be multiples of 128");
_Static_assert(BENCH_K % 64 == 0, "K must be a multiple of 64");
_Static_assert(BENCH_VERIFY_FULL == 0 || BENCH_VERIFY_FULL == 1,
               "VERIFY_FULL must be 0 or 1");

#define TILE_M 128u
#define TILE_N 128u
#define TILE_K 64u
#define PACKED_B_ROW_BYTES (TILE_K / 4u)
#define I2_LANES_PER_BYTE 4u
#define PACKED_B_PANEL_BYTES (TILE_N * PACKED_B_ROW_BYTES)
#define PACKED_B_ROWS_PER_LINE 4u
#define PACKED_B_PAGE_BYTES 4096u
#define INIT_FENCE_BYTES 4096u
#ifndef BENCH_PENDING_STORE0_K_TILE
#define BENCH_PENDING_STORE0_K_TILE 12u
#endif

#ifndef BENCH_PENDING_STORE1_K_TILE
#define BENCH_PENDING_STORE1_K_TILE 24u
#endif

#define PENDING_STORE0_K_TILE BENCH_PENDING_STORE0_K_TILE
#define PENDING_STORE1_K_TILE BENCH_PENDING_STORE1_K_TILE

#define MAX_N 6912u
#define MAX_K 6912u

#if BENCH_N != 0u
#define ALLOC_N BENCH_N
#else
#define ALLOC_N MAX_N
#endif

#if BENCH_K != 0u
#define ALLOC_K BENCH_K
#else
#define ALLOC_K MAX_K
#endif

#define ALLOC_WEIGHT_BYTES ((size_t)ALLOC_N * ALLOC_K)
#define ALLOC_A_BYTES ((size_t)BENCH_M * ALLOC_K)

#define MCFG_INT8 0x02UL
#define MCFG_INT32 0x04UL
#define MCFG_FP2PACK4 0x0dUL

typedef struct {
  const char *name;
  unsigned int n;
  unsigned int k;
} GemmShape;

typedef struct {
  uint64_t cycles;
  uint64_t instret;
  uint64_t macs;
  uint64_t a_traffic_bytes;
  uint64_t b_unique_bytes;
  uint64_t b_traffic_bytes;
  uint64_t b_panel_loads;
  uint64_t c_store_bytes;
  uint64_t mmaccs;
  uint64_t kernel_syncs;
  uint64_t intermediate_syncs;
  bool packed;
} BenchmarkResult;

typedef struct {
  int32_t *c0;
  int32_t *c1;
  unsigned int c_base;
  unsigned long c_stride;
  bool valid;
  bool has_c1;
  bool c0_issued;
  bool c1_issued;
} PendingStores;

_Static_assert(BENCH_M % TILE_M == 0, "M must be an integer number of 128-row tiles");
_Static_assert(MAX_N % TILE_N == 0, "N panels must be 128 rows");
_Static_assert(MAX_K % TILE_K == 0, "K panels must be 64 elements");
_Static_assert(PACKED_B_ROW_BYTES == 16u, "fp2pack4 B rows are fixed at 16 bytes");
_Static_assert(TILE_K == PACKED_B_ROW_BYTES * I2_LANES_PER_BYTE,
               "fp2pack4 must encode four i2 lanes per byte");
_Static_assert(PACKED_B_ROWS_PER_LINE * PACKED_B_ROW_BYTES == 64u,
               "one packed B cache line must hold four rows");
_Static_assert(PACKED_B_PANEL_BYTES * 2u == PACKED_B_PAGE_BYTES,
               "two fp2pack4 B panels must fill one 4-KiB page");
_Static_assert(PENDING_STORE0_K_TILE < PENDING_STORE1_K_TILE,
               "deferred C stores must retain program order");
_Static_assert(BENCH_K_BATCH_TILES == 0u,
               "K batch must be unbounded for this kernel-level sync benchmark");

/*
 * B uses [NtilePair][Ktile][NwithinPair][Nwithin][Kwithin] storage. This
 * outer-N-pair/inner-K swizzle keeps the two B panels consumed by tr2/tr3 at
 * one K step adjacent, while each packed B panel remains a dense 128 x 16 B
 * object with the hardware-mandated 16 B row stride.
 */
static int8_t matrix_a[ALLOC_A_BYTES] __attribute__((aligned(64)));
/*
 * Every packed 128x64 panel occupies 2 KiB within one 4-KiB page.
 * Aligning the backing array to a page places consecutive panels at
 * page offsets 0x000 or 0x800.
 */
static uint8_t matrix_b_fp2[ALLOC_WEIGHT_BYTES / 4u]
    __attribute__((aligned(PACKED_B_PAGE_BYTES)));
static int32_t matrix_c[BENCH_M * ALLOC_N] __attribute__((aligned(64)));

static inline uint64_t read_mcycle(void) {
  uint64_t value;
  asm volatile("rdcycle %0" : "=r"(value));
  return value;
}

static inline uint64_t read_minstret(void) {
  uint64_t value;
  asm volatile("rdinstret %0" : "=r"(value));
  return value;
}

static int8_t decode_i2(uint8_t code) {
  code &= 3u;
  return (int8_t)((code & 2u) ? (int)code - 4 : (int)code);
}

static uint8_t weight_code(unsigned int n_tile, unsigned int k_tile,
                           unsigned int n_in_tile, unsigned int k_in_tile) {
  return (uint8_t)((n_tile + 2u * k_tile + n_in_tile + k_in_tile) & 3u);
}

static int8_t activation_code(uint8_t code) {
  switch (code & 3u) {
    case 0: return -2;
    case 1: return -1;
    case 2: return 1;
    default: return 2;
  }
}

static int8_t activation_value(unsigned int m_tile, unsigned int m_in_tile,
                               unsigned int k_tile, unsigned int k_in_tile) {
  return activation_code((uint8_t)(m_tile + 3u * m_in_tile +
                                   2u * k_tile + k_in_tile));
}

static size_t b_panel_index(unsigned int n_tile, unsigned int k_tile,
                            unsigned int n_tiles, unsigned int k_tiles) {
  const unsigned int n_pair = n_tile / 2u;
  const bool pair_has_second = n_pair * 2u + 1u < n_tiles;
  const size_t pair_base = (size_t)n_pair * 2u * k_tiles;

  if (pair_has_second) {
    return pair_base + (size_t)k_tile * 2u + (n_tile & 1u);
  }
  return pair_base + k_tile;
}

#define MATRIX_SYNC_RESET() asm volatile("msyncregreset sync1" ::: "memory")
#define MATRIX_FENCE() asm volatile("mfence" ::: "memory")
#define MATRIX_ZERO_ACC0() asm volatile("mzero acc0" ::: "memory")
#define MATRIX_ZERO_ACC1() asm volatile("mzero acc1" ::: "memory")
#define MATRIX_ZERO_ACC2() asm volatile("mzero acc2" ::: "memory")
#define MATRIX_ZERO_ACC3() asm volatile("mzero acc3" ::: "memory")
#define MATRIX_LOAD_A0(ptr, stride) \
  asm volatile("mla tr0, (%0), %1" :: "r"(ptr), "r"(stride) : "memory")
#define MATRIX_LOAD_A1(ptr, stride) \
  asm volatile("mla tr1, (%0), %1" :: "r"(ptr), "r"(stride) : "memory")
#define MATRIX_LOAD_B0(ptr, stride) \
  asm volatile("mlb tr2, (%0), %1" :: "r"(ptr), "r"(stride) : "memory")
#define MATRIX_LOAD_B1(ptr, stride) \
  asm volatile("mlb tr3, (%0), %1" :: "r"(ptr), "r"(stride) : "memory")
#define MATRIX_MACC_ACC0_B0() asm volatile("mmacc acc0, tr2, tr0" ::: "memory")
#define MATRIX_MACC_ACC0_B1() asm volatile("mmacc acc0, tr3, tr0" ::: "memory")
#define MATRIX_MACC_ACC1_B0() asm volatile("mmacc acc1, tr2, tr1" ::: "memory")
#define MATRIX_MACC_ACC1_B1() asm volatile("mmacc acc1, tr3, tr1" ::: "memory")
#define MATRIX_MACC_ACC2_B0() asm volatile("mmacc acc2, tr2, tr0" ::: "memory")
#define MATRIX_MACC_ACC2_B1() asm volatile("mmacc acc2, tr3, tr0" ::: "memory")
#define MATRIX_MACC_ACC3_B0() asm volatile("mmacc acc3, tr2, tr1" ::: "memory")
#define MATRIX_MACC_ACC3_B1() asm volatile("mmacc acc3, tr3, tr1" ::: "memory")
#define MATRIX_STORE_C0(ptr, stride) \
  asm volatile("msc acc0, (%0), %1" :: "r"(ptr), "r"(stride) : "memory")
#define MATRIX_STORE_C1(ptr, stride) \
  asm volatile("msc acc1, (%0), %1" :: "r"(ptr), "r"(stride) : "memory")
#define MATRIX_STORE_C2(ptr, stride) \
  asm volatile("msc acc2, (%0), %1" :: "r"(ptr), "r"(stride) : "memory")
#define MATRIX_STORE_C3(ptr, stride) \
  asm volatile("msc acc3, (%0), %1" :: "r"(ptr), "r"(stride) : "memory")
#define MATRIX_RELEASE() asm volatile("mrelease sync1" ::: "memory")
#define MATRIX_ACQUIRE(target) \
  asm volatile("macquire sync1, %0" :: "r"(target) : "memory")

static inline void zero_acc(unsigned int acc) {
  switch (acc) {
    case 0u: MATRIX_ZERO_ACC0(); break;
    case 1u: MATRIX_ZERO_ACC1(); break;
    case 2u: MATRIX_ZERO_ACC2(); break;
    default: MATRIX_ZERO_ACC3(); break;
  }
}

static inline void store_acc(unsigned int acc, int32_t *ptr,
                             unsigned long stride) {
  switch (acc) {
    case 0u: MATRIX_STORE_C0(ptr, stride); break;
    case 1u: MATRIX_STORE_C1(ptr, stride); break;
    case 2u: MATRIX_STORE_C2(ptr, stride); break;
    default: MATRIX_STORE_C3(ptr, stride); break;
  }
}

static inline void load_a_buffer(unsigned int buffer, const int8_t *ptr,
                                 unsigned long stride) {
  if (buffer == 0u) {
    MATRIX_LOAD_A0(ptr, stride);
  } else {
    MATRIX_LOAD_A1(ptr, stride);
  }
}

static inline void load_b_buffer(unsigned int buffer, const void *ptr,
                                 unsigned long stride) {
  if (buffer == 0u) {
    MATRIX_LOAD_B0(ptr, stride);
  } else {
    MATRIX_LOAD_B1(ptr, stride);
  }
}

static inline void mmacc_acc_with_regs(unsigned int acc,
                                       unsigned int b_buffer,
                                       unsigned int a_buffer) {
  switch (acc) {
    case 0u:
      if (b_buffer == 0u) {
        if (a_buffer == 0u) {
          asm volatile("mmacc acc0, tr2, tr0" ::: "memory");
        } else {
          asm volatile("mmacc acc0, tr2, tr1" ::: "memory");
        }
      } else if (a_buffer == 0u) {
        asm volatile("mmacc acc0, tr3, tr0" ::: "memory");
      } else {
        asm volatile("mmacc acc0, tr3, tr1" ::: "memory");
      }
      break;
    case 1u:
      if (b_buffer == 0u) {
        if (a_buffer == 0u) {
          asm volatile("mmacc acc1, tr2, tr0" ::: "memory");
        } else {
          asm volatile("mmacc acc1, tr2, tr1" ::: "memory");
        }
      } else if (a_buffer == 0u) {
        asm volatile("mmacc acc1, tr3, tr0" ::: "memory");
      } else {
        asm volatile("mmacc acc1, tr3, tr1" ::: "memory");
      }
      break;
    case 2u:
      if (b_buffer == 0u) {
        if (a_buffer == 0u) {
          asm volatile("mmacc acc2, tr2, tr0" ::: "memory");
        } else {
          asm volatile("mmacc acc2, tr2, tr1" ::: "memory");
        }
      } else if (a_buffer == 0u) {
        asm volatile("mmacc acc2, tr3, tr0" ::: "memory");
      } else {
        asm volatile("mmacc acc2, tr3, tr1" ::: "memory");
      }
      break;
    default:
      if (b_buffer == 0u) {
        if (a_buffer == 0u) {
          asm volatile("mmacc acc3, tr2, tr0" ::: "memory");
        } else {
          asm volatile("mmacc acc3, tr2, tr1" ::: "memory");
        }
      } else if (a_buffer == 0u) {
        asm volatile("mmacc acc3, tr3, tr0" ::: "memory");
      } else {
        asm volatile("mmacc acc3, tr3, tr1" ::: "memory");
      }
      break;
  }
}

static inline void mmacc_n0(unsigned int c_base, unsigned int a_buffer,
                            unsigned int b_buffer) {
  mmacc_acc_with_regs(c_base, b_buffer, a_buffer);
}

static inline void mmacc_n1(unsigned int c_base, unsigned int a_buffer,
                            unsigned int b_buffer) {
  mmacc_acc_with_regs(c_base + 1u, b_buffer, a_buffer);
}

static inline void issue_pending_stores(PendingStores *pending,
                                        unsigned int completed_k_tiles) {
  if (!pending->valid) return;
  if (!pending->c0_issued && completed_k_tiles >= PENDING_STORE0_K_TILE) {
    store_acc(pending->c_base, pending->c0, pending->c_stride);
    pending->c0_issued = true;
  }
  if (pending->has_c1 && !pending->c1_issued &&
      completed_k_tiles >= PENDING_STORE1_K_TILE) {
    store_acc(pending->c_base + 1u, pending->c1, pending->c_stride);
    pending->c1_issued = true;
  }
}

static inline void flush_pending_stores(PendingStores *pending) {
  if (!pending->valid) return;
  if (!pending->c0_issued) {
    store_acc(pending->c_base, pending->c0, pending->c_stride);
    pending->c0_issued = true;
  }
  if (pending->has_c1 && !pending->c1_issued) {
    store_acc(pending->c_base + 1u, pending->c1, pending->c_stride);
    pending->c1_issued = true;
  }
}

/*
 * This specializes the full-N-pair inner loop.  The generic implementation
 * below has to select its C, A, and B registers at run time; here the two
 * CReg-pair variants emit those mappings directly.  B is still loaded in
 * the contract order N0/N1 for each K tile, but N1 computes first so its
 * BReg can immediately receive the next K tile's N0 panel.
 */
#define MATRIX_FAST_C0_EVEN_N1() \
  asm volatile("mmacc acc1, tr3, tr0" ::: "memory")
#define MATRIX_FAST_C0_EVEN_N0() \
  asm volatile("mmacc acc0, tr2, tr0" ::: "memory")
#define MATRIX_FAST_C0_ODD_N1() \
  asm volatile("mmacc acc1, tr2, tr1" ::: "memory")
#define MATRIX_FAST_C0_ODD_N0() \
  asm volatile("mmacc acc0, tr3, tr1" ::: "memory")
#define MATRIX_FAST_C2_EVEN_N1() \
  asm volatile("mmacc acc3, tr3, tr0" ::: "memory")
#define MATRIX_FAST_C2_EVEN_N0() \
  asm volatile("mmacc acc2, tr2, tr0" ::: "memory")
#define MATRIX_FAST_C2_ODD_N1() \
  asm volatile("mmacc acc3, tr2, tr1" ::: "memory")
#define MATRIX_FAST_C2_ODD_N0() \
  asm volatile("mmacc acc2, tr3, tr1" ::: "memory")
#define MATRIX_FAST_LOAD_INITIAL(base, panel_bytes, stride) \
  do { \
    MATRIX_LOAD_B0((base), (stride)); \
    MATRIX_LOAD_B1((base) + (panel_bytes), (stride)); \
  } while (0)
#define MATRIX_FAST_LOAD_EVEN_AFTER_N1(base, next_k, panel_bytes, stride) \
  MATRIX_LOAD_B1((base) + (size_t)(next_k) * 2u * (panel_bytes), (stride))
#define MATRIX_FAST_LOAD_EVEN_AFTER_N0(base, next_k, panel_bytes, stride) \
  MATRIX_LOAD_B0((base) + ((size_t)(next_k) * 2u + 1u) * (panel_bytes), (stride))
#define MATRIX_FAST_LOAD_ODD_AFTER_N1(base, next_k, panel_bytes, stride) \
  MATRIX_LOAD_B0((base) + (size_t)(next_k) * 2u * (panel_bytes), (stride))
#define MATRIX_FAST_LOAD_ODD_AFTER_N0(base, next_k, panel_bytes, stride) \
  MATRIX_LOAD_B1((base) + ((size_t)(next_k) * 2u + 1u) * (panel_bytes), (stride))

#define DEFINE_FAST_FULL_NPAIR_BLOCK(name, cbase_value, zero_n0, zero_n1, \
                                     even_n1, even_n0, odd_n1, odd_n0) \
static void name(const GemmShape *shape, bool packed, unsigned int mt, \
                 unsigned int nt, PendingStores *pending) { \
  const unsigned int n_tiles = shape->n / TILE_N; \
  const unsigned int k_tiles = shape->k / TILE_K; \
  const unsigned long a_stride = shape->k; \
  const unsigned long b_stride = PACKED_B_ROW_BYTES; \
  const unsigned long c_stride = (unsigned long)shape->n * sizeof(int32_t); \
  const size_t b_panel_bytes = PACKED_B_PANEL_BYTES; \
  const size_t pair_base = (size_t)(nt / 2u) * 2u * k_tiles; \
  const uint8_t *const b_pair_base = matrix_b_fp2 + pair_base * PACKED_B_PANEL_BYTES; \
  int32_t *const c0 = matrix_c + \
      ((size_t)mt * TILE_M * shape->n + (size_t)nt * TILE_N); \
  int32_t *const c1 = c0 + TILE_N; \
  const int8_t *const a_base = matrix_a + \
      (size_t)mt * TILE_M * shape->k; \
  (void)n_tiles; \
  zero_n0(); \
  zero_n1(); \
  MATRIX_FAST_LOAD_INITIAL(b_pair_base, b_panel_bytes, b_stride); \
  MATRIX_LOAD_A0(a_base, a_stride); \
  for (unsigned int kt = 0u; kt < k_tiles; kt += 2u) { \
    even_n1(); \
    if (kt + 1u < k_tiles) { \
      MATRIX_FAST_LOAD_EVEN_AFTER_N1(b_pair_base, kt + 1u, b_panel_bytes, \
                                     b_stride); \
    } \
    even_n0(); \
    if (kt + 1u < k_tiles) { \
      MATRIX_FAST_LOAD_EVEN_AFTER_N0(b_pair_base, kt + 1u, b_panel_bytes, \
                                     b_stride); \
      MATRIX_LOAD_A1(a_base + (size_t)(kt + 1u) * TILE_K, a_stride); \
    } \
    issue_pending_stores(pending, kt + 1u); \
    if (kt + 1u == k_tiles) break; \
    odd_n1(); \
    if (kt + 2u < k_tiles) { \
      MATRIX_FAST_LOAD_ODD_AFTER_N1(b_pair_base, kt + 2u, b_panel_bytes, \
                                    b_stride); \
    } \
    odd_n0(); \
    if (kt + 2u < k_tiles) { \
      MATRIX_FAST_LOAD_ODD_AFTER_N0(b_pair_base, kt + 2u, b_panel_bytes, \
                                    b_stride); \
      MATRIX_LOAD_A0(a_base + (size_t)(kt + 2u) * TILE_K, a_stride); \
    } \
    issue_pending_stores(pending, kt + 2u); \
  } \
  flush_pending_stores(pending); \
  *pending = (PendingStores){ \
      .c0 = c0, \
      .c1 = c1, \
      .c_base = cbase_value, \
      .c_stride = c_stride, \
      .valid = true, \
      .has_c1 = true, \
  }; \
}

DEFINE_FAST_FULL_NPAIR_BLOCK(issue_full_npair_c0_fast, 0u,
                             MATRIX_ZERO_ACC0, MATRIX_ZERO_ACC1,
                             MATRIX_FAST_C0_EVEN_N1, MATRIX_FAST_C0_EVEN_N0,
                             MATRIX_FAST_C0_ODD_N1, MATRIX_FAST_C0_ODD_N0)

DEFINE_FAST_FULL_NPAIR_BLOCK(issue_full_npair_c2_fast, 2u,
                             MATRIX_ZERO_ACC2, MATRIX_ZERO_ACC3,
                             MATRIX_FAST_C2_EVEN_N1, MATRIX_FAST_C2_EVEN_N0,
                             MATRIX_FAST_C2_ODD_N1, MATRIX_FAST_C2_ODD_N0)

#undef DEFINE_FAST_FULL_NPAIR_BLOCK
#undef MATRIX_FAST_C0_EVEN_N1
#undef MATRIX_FAST_C0_EVEN_N0
#undef MATRIX_FAST_C0_ODD_N1
#undef MATRIX_FAST_C0_ODD_N0
#undef MATRIX_FAST_C2_EVEN_N1
#undef MATRIX_FAST_C2_EVEN_N0
#undef MATRIX_FAST_C2_ODD_N1
#undef MATRIX_FAST_C2_ODD_N0
#undef MATRIX_FAST_LOAD_INITIAL
#undef MATRIX_FAST_LOAD_EVEN_AFTER_N1
#undef MATRIX_FAST_LOAD_EVEN_AFTER_N0
#undef MATRIX_FAST_LOAD_ODD_AFTER_N1
#undef MATRIX_FAST_LOAD_ODD_AFTER_N0

static inline void throttle_initialization(size_t *pending_bytes,
                                           size_t written_bytes) {
  *pending_bytes += written_bytes;
  if (*pending_bytes >= INIT_FENCE_BYTES) {
    asm volatile("fence rw, rw" ::: "memory");
    *pending_bytes = 0;
  }
}

static void configure_matrix(bool packed) {
  const unsigned long mstatus_mask = (1UL << 25) | (1UL << 13) | (1UL << 9);
  const unsigned long a_type = MCFG_INT8;
  const unsigned long b_type = MCFG_FP2PACK4;
  const unsigned long c_type = MCFG_INT32;

  asm volatile("csrs mstatus, %0\n\t"
               "csrwi vcsr, 0\n\t"
               :
               : "r"(mstatus_mask)
               : "memory");
  asm volatile("msettilem %0\n\t"
               "msettilen %1\n\t"
               "msettilek %2\n\t"
               :
               : "r"((unsigned long)TILE_M), "r"((unsigned long)TILE_N),
                 "r"((unsigned long)TILE_K)
               : "memory");
  asm volatile("msetcfg mcfg0, %0\n\t"
               "msetcfg mcfg1, %0\n\t"
               "msetcfg mcfg2, %1\n\t"
               "msetcfg mcfg3, %1\n\t"
               "msetcfg mcfg4, %2\n\t"
               "msetcfg mcfg5, %2\n\t"
               "msetcfg mcfg6, %2\n\t"
               "msetcfg mcfg7, %2\n\t"
               :
               : "r"(a_type), "r"(b_type), "r"(c_type)
               : "memory");
}

static void initialize_operands(const GemmShape *shape) {
  const unsigned int m_tiles = BENCH_M / TILE_M;
  const unsigned int n_tiles = shape->n / TILE_N;
  const unsigned int k_tiles = shape->k / TILE_K;
  int8_t a_patterns[4][TILE_K];
  uint8_t fp2_b_patterns[4][PACKED_B_ROW_BYTES];
  uint8_t fp2_line_patterns[4][64];
  size_t pending_init_bytes = 0;

  for (unsigned int base = 0; base < 4u; ++base) {
    for (unsigned int ki = 0; ki < TILE_K; ++ki) {
      a_patterns[base][ki] = activation_code((uint8_t)(base + ki));
    }
    for (unsigned int byte = 0; byte < PACKED_B_ROW_BYTES; ++byte) {
      uint8_t packed = 0;
      for (unsigned int lane = 0; lane < I2_LANES_PER_BYTE; ++lane) {
        const unsigned int ki = byte * I2_LANES_PER_BYTE + lane;
        packed |= (uint8_t)(((base + ki) & 3u) << (2u * lane));
      }
      fp2_b_patterns[base][byte] = packed;
    }
  }
  for (unsigned int base = 0; base < 4u; ++base) {
    for (unsigned int row_in_line = 0; row_in_line < PACKED_B_ROWS_PER_LINE;
         ++row_in_line) {
      memcpy(fp2_line_patterns[base] + row_in_line * PACKED_B_ROW_BYTES,
             fp2_b_patterns[(base + row_in_line) & 3u], PACKED_B_ROW_BYTES);
    }
  }

  // Every M tile is initialized independently. The kernel later consumes
  // adjacent tiles in pairs, but that pairing must not omit the odd tile.
  for (unsigned int mt = 0; mt < m_tiles; ++mt) {
    for (unsigned int mi = 0; mi < TILE_M; ++mi) {
      int8_t *const a_row = matrix_a +
          ((size_t)mt * TILE_M + mi) * shape->k;
      for (unsigned int kt = 0; kt < k_tiles; ++kt) {
        const unsigned int base = (mt + 3u * mi + 2u * kt) & 3u;
        memcpy(a_row + kt * TILE_K, a_patterns[base], TILE_K);
        throttle_initialization(&pending_init_bytes, TILE_K);
      }
    }
  }

  for (unsigned int nt = 0; nt < n_tiles; ++nt) {
    for (unsigned int kt = 0; kt < k_tiles; ++kt) {
      const size_t panel = b_panel_index(nt, kt, n_tiles, k_tiles);
      uint8_t *const fp2_panel = matrix_b_fp2 +
          panel * PACKED_B_PANEL_BYTES;

      for (unsigned int ni = 0; ni < TILE_N; ni += PACKED_B_ROWS_PER_LINE) {
        const unsigned int base = (nt + 2u * kt + ni) & 3u;
        uint8_t *const fp2_line = fp2_panel + (size_t)ni * PACKED_B_ROW_BYTES;
        memcpy(fp2_line, fp2_line_patterns[base], sizeof(fp2_line_patterns[0]));
        throttle_initialization(&pending_init_bytes, sizeof(fp2_line_patterns[0]));
      }
    }
  }
  if (pending_init_bytes != 0u) asm volatile("fence rw, rw" ::: "memory");
  MATRIX_FENCE();
}

static int32_t expected_output(const GemmShape *shape, size_t row,
                               size_t column) {
  const unsigned int m_tile = row / TILE_M;
  const unsigned int m_in_tile = row % TILE_M;
  const unsigned int n_tile = column / TILE_N;
  const unsigned int n_in_tile = column % TILE_N;
  const unsigned int k_tiles = shape->k / TILE_K;
  int32_t result = 0;

  for (unsigned int kt = 0; kt < k_tiles; ++kt) {
    for (unsigned int ki = 0; ki < TILE_K; ++ki) {
      result += (int32_t)activation_value(m_tile, m_in_tile, kt, ki) *
          (int32_t)decode_i2(weight_code(n_tile, kt, n_in_tile, ki));
    }
  }
  return result;
}

static bool verify_one_sample(const GemmShape *shape, size_t row,
                              size_t column) {
  const size_t index = row * shape->n + column;
  const int32_t expected = expected_output(shape, row, column);

  if (matrix_c[index] != expected) {
    printf("[FAIL] C[%lu][%lu] got=%d expected=%d\n",
           (unsigned long)row, (unsigned long)column, matrix_c[index], expected);
    return false;
  }
  return true;
}

static bool verify_samples(const GemmShape *shape) {
#if BENCH_VERIFY_FULL
  for (size_t row = 0; row < BENCH_M; ++row) {
    for (size_t column = 0; column < shape->n; ++column) {
      if (!verify_one_sample(shape, row, column)) return false;
    }
  }
  return true;
#else
  const unsigned int n_tiles = shape->n / TILE_N;
  const unsigned int m_tiles = BENCH_M / TILE_M;
  const size_t fixed_samples[][2] = {
      {0u, 0u},
      {0u, shape->n - 1u},
      {BENCH_M - 1u, 0u},
      {BENCH_M - 1u, shape->n - 1u},
      {BENCH_M / 2u, shape->n / 2u},
  };

  for (size_t i = 0; i < sizeof(fixed_samples) / sizeof(fixed_samples[0]); ++i) {
    if (!verify_one_sample(shape, fixed_samples[i][0], fixed_samples[i][1])) {
      return false;
    }
  }
  for (unsigned int nt = 0; nt < n_tiles; ++nt) {
    const size_t row = ((size_t)nt * 37u) % BENCH_M;
    const size_t column = (size_t)nt * TILE_N + (nt * 29u) % TILE_N;
    if (!verify_one_sample(shape, row, column)) return false;
  }
  for (unsigned int mt = 0; mt < m_tiles; ++mt) {
    const size_t row = (size_t)mt * TILE_M + (mt * 17u) % TILE_M;
    const size_t column = ((size_t)mt * 19u) % shape->n;
    if (!verify_one_sample(shape, row, column)) return false;
  }
  // Cover every output tile, including the singleton K/V N-tile pair and
  // every odd M tile consumed by an M-pair block.
  for (unsigned int mt = 0; mt < m_tiles; ++mt) {
    for (unsigned int nt = 0; nt < n_tiles; ++nt) {
      const size_t row = (size_t)mt * TILE_M +
          (mt * 37u + nt * 17u) % TILE_M;
      const size_t column = (size_t)nt * TILE_N +
          (mt * 29u + nt * 43u) % TILE_N;
      if (!verify_one_sample(shape, row, column)) return false;
    }
  }
  return true;
#endif
}

static __attribute__((unused)) void issue_mtile_npair_block(
    const GemmShape *shape, bool packed, unsigned int mt, unsigned int nt,
    unsigned int c_base, PendingStores *pending) {
  const unsigned int n_tiles = shape->n / TILE_N;
  const unsigned int k_tiles = shape->k / TILE_K;
  const bool has_n1 = nt + 1u < n_tiles;
  const unsigned long a_stride = shape->k;
  const unsigned long b_stride = PACKED_B_ROW_BYTES;
  const unsigned long c_stride = (unsigned long)shape->n * sizeof(int32_t);
  int32_t *const c0 = matrix_c +
      ((size_t)mt * TILE_M * shape->n + (size_t)nt * TILE_N);
  int32_t *const c1 = c0 + TILE_N;
  const int8_t *const a_base = matrix_a +
      (size_t)mt * TILE_M * shape->k;

  // N1 computes first; its B register receives next K's N0. Then N0's
  // register receives next K's N1. B ownership swaps on every K tile.
  zero_acc(c_base);
  if (has_n1) zero_acc(c_base + 1u);

  const size_t first_panel0 = b_panel_index(nt, 0u, n_tiles, k_tiles);
  const void *const first_b0 = (const void *)(matrix_b_fp2 + first_panel0 * PACKED_B_PANEL_BYTES);
  load_b_buffer(0u, first_b0, b_stride);
  if (has_n1) {
    const size_t first_panel1 = b_panel_index(nt + 1u, 0u, n_tiles, k_tiles);
    const void *const first_b1 = (const void *)(matrix_b_fp2 + first_panel1 * PACKED_B_PANEL_BYTES);
    load_b_buffer(1u, first_b1, b_stride);
  }
  load_a_buffer(0u, a_base, a_stride);

  for (unsigned int kt = 0; kt < k_tiles; ++kt) {
    const unsigned int a_buffer = kt & 1u;
    const unsigned int n0_b_buffer = kt & 1u;
    const unsigned int n1_b_buffer = n0_b_buffer ^ 1u;

    if (has_n1) {
      // B panels are stored N0 then N1 within one K tile. Starting with N1
      // releases tr3 for the next K tile's N0 panel, so BML sees ascending
      // panel addresses: K+1/N0 before K+1/N1.
      mmacc_n1(c_base, a_buffer, n1_b_buffer);
      if (kt + 1u < k_tiles) {
        const size_t next_panel0 = b_panel_index(nt, kt + 1u, n_tiles, k_tiles);
        const void *const next_b0 = (const void *)(matrix_b_fp2 + next_panel0 * PACKED_B_PANEL_BYTES);
        load_b_buffer(n1_b_buffer, next_b0, b_stride);
      }
      mmacc_n0(c_base, a_buffer, n0_b_buffer);
      if (kt + 1u < k_tiles) {
        const size_t next_panel1 = b_panel_index(nt + 1u, kt + 1u,
                                                  n_tiles, k_tiles);
        const void *const next_b1 = (const void *)(matrix_b_fp2 + next_panel1 * PACKED_B_PANEL_BYTES);
        load_b_buffer(n0_b_buffer, next_b1, b_stride);
      }
    } else {
      mmacc_n0(c_base, a_buffer, n0_b_buffer);
      if (kt + 1u < k_tiles) {
        const size_t next_panel0 = b_panel_index(nt, kt + 1u, n_tiles, k_tiles);
        const void *const next_b0 = (const void *)(matrix_b_fp2 + next_panel0 * PACKED_B_PANEL_BYTES);
        load_b_buffer(n1_b_buffer, next_b0, b_stride);
      }
    }
    if (kt + 1u < k_tiles) {
      load_a_buffer(a_buffer ^ 1u, a_base + (size_t)(kt + 1u) * TILE_K,
                    a_stride);
    }
    // Current compute uses the opposite CReg pair, so the prior block's CML
    // stores can drain after its new A/B work has entered the issue window.
    issue_pending_stores(pending, kt + 1u);
  }
  flush_pending_stores(pending);
  *pending = (PendingStores){
      .c0 = c0,
      .c1 = c1,
      .c_base = c_base,
      .c_stride = c_stride,
      .valid = true,
      .has_c1 = has_n1,
  };
}

static BenchmarkResult run_gemm(const GemmShape *shape, bool packed) {
  const unsigned int m_tiles = BENCH_M / TILE_M;
  const unsigned int n_tiles = shape->n / TILE_N;
  const unsigned int k_tiles = shape->k / TILE_K;
  PendingStores pending = {0};
  BenchmarkResult result = {0};

  configure_matrix(packed);
  MATRIX_SYNC_RESET();
  MATRIX_FENCE();
  const uint64_t cycle_begin = read_mcycle();
  const uint64_t instret_begin = read_minstret();

  // Process an M group consecutively for each N pair. This keeps B panels
  // hot across the selected number of M tiles while every output block still
  // uses the same output-stationary CReg-pair schedule.
  unsigned int c_base = 0u;
  for (unsigned int mt_base = 0; mt_base < m_tiles;
      mt_base += BENCH_M_GROUP_TILES) {
    for (unsigned int nt = 0; nt < n_tiles; nt += 2u) {
      for (unsigned int mt = mt_base;
           mt < mt_base + BENCH_M_GROUP_TILES
#if (BENCH_M / TILE_M) % 2u != 0u
           && mt < m_tiles
#endif
           ; ++mt) {
        if (nt + 1u < n_tiles) {
          if (c_base == 0u) {
            issue_full_npair_c0_fast(shape, packed, mt, nt, &pending);
          } else {
            issue_full_npair_c2_fast(shape, packed, mt, nt, &pending);
          }
        } else {
          issue_mtile_npair_block(shape, packed, mt, nt, c_base, &pending);
        }
        c_base ^= 2u;
      }
    }
  }

  // The final CReg pair has no successor block to overlap its stores.
  flush_pending_stores(&pending);

  // This is the sole completion boundary for all pipelined output blocks.
  MATRIX_RELEASE();
  MATRIX_ACQUIRE(1UL);
  MATRIX_FENCE();
  result.cycles = read_mcycle() - cycle_begin;
  result.instret = read_minstret() - instret_begin;
  result.macs = (uint64_t)BENCH_M * shape->n * shape->k;
  result.a_traffic_bytes = (uint64_t)BENCH_M * shape->k *
      ((n_tiles + 1u) / 2u);
  result.b_unique_bytes = (uint64_t)shape->n * shape->k / (packed ? 4u : 1u);
  result.b_traffic_bytes = result.b_unique_bytes * m_tiles;
  result.b_panel_loads = (uint64_t)m_tiles * n_tiles * k_tiles;
  result.c_store_bytes = (uint64_t)BENCH_M * shape->n * sizeof(int32_t);
  result.mmaccs = (uint64_t)m_tiles * n_tiles * k_tiles;
  result.kernel_syncs = 1u;
  result.intermediate_syncs = 0u;
  result.packed = packed;
  return result;
}

static void print_result(const GemmShape *shape, const BenchmarkResult *result) {
  printf("BITVLA_RESULT case=%s mode=%s schedule=%s m=%u n=%u k=%u k_batch=%s cycles=%llu instret=%llu macs=%llu mmaccs=%llu a_traffic_bytes=%llu b_unique_bytes=%llu b_traffic_bytes=%llu b_panel_loads=%llu c_store_bytes=%llu kernel_syncs=%llu intermediate_syncs=%llu\n",
         shape->name, result->packed ? "fp2pack4" : "i8",
         SCHEDULE_NAME, BENCH_M, shape->n, shape->k, K_BATCH_MODE_NAME,
         (unsigned long long)result->cycles,
         (unsigned long long)result->instret,
         (unsigned long long)result->macs,
         (unsigned long long)result->mmaccs,
         (unsigned long long)result->a_traffic_bytes,
         (unsigned long long)result->b_unique_bytes,
         (unsigned long long)result->b_traffic_bytes,
         (unsigned long long)result->b_panel_loads,
         (unsigned long long)result->c_store_bytes,
         (unsigned long long)result->kernel_syncs,
         (unsigned long long)result->intermediate_syncs);
}

int main(void) {
  /* Preserve the measured kernel's shape setup and compiler specialization. */
  GemmShape shape = {"custom", 2560u, 2560u};
  if (BENCH_N != 0u) shape.n = BENCH_N;
  if (BENCH_K != 0u) shape.k = BENCH_K;
  if (shape.n > ALLOC_N || shape.k > ALLOC_K ||
      shape.n % TILE_N != 0u || shape.k % TILE_K != 0u) {
    printf("[FAIL] N=%u K=%u must fit swizzled 128x64 tiles\n", shape.n, shape.k);
    return 1;
  }

  printf("BITVLA_GEMM case=%s mode=%s schedule=%s M=%u N=%u K=%u k_batch=%s\n",
         shape.name, BENCH_MODE_NAME, SCHEDULE_NAME, BENCH_M, shape.n, shape.k,
         K_BATCH_MODE_NAME);
  initialize_operands(&shape);

  const BenchmarkResult fp2 = run_gemm(&shape, true);
  if (!verify_samples(&shape)) return 1;
  print_result(&shape, &fp2);

  printf("[PASS] %s M=%u N=%u K=%u\n",
         shape.name, BENCH_M, shape.n, shape.k);
  return 0;
}
