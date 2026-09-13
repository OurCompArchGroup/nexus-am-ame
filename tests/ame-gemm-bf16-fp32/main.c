/* Full-tile GEMM; extracted from the measured BitVLA OpenLLC schedule. */
#ifndef BENCH_VERIFY_FULL
#define BENCH_VERIFY_FULL 0
#endif
#include <am.h>
#include <klib.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef BENCH_M
#define BENCH_M 128u
#endif

#ifndef BENCH_N
#define BENCH_N 256u
#endif

#ifndef BENCH_K
#define BENCH_K 32u
#endif

#ifndef BENCH_NAME
#define BENCH_NAME "custom"
#endif

#ifndef BENCH_VERIFY_FULL
#define BENCH_VERIFY_FULL 0
#endif

#ifndef BENCH_K_BATCH_TILES
#define BENCH_K_BATCH_TILES 0u
#endif

_Static_assert(BENCH_M > 0 && BENCH_N > 0 && BENCH_K > 0,
               "M, N and K must be positive");
_Static_assert(BENCH_M % 128 == 0 && BENCH_N % 128 == 0,
               "M and N must be multiples of 128");
_Static_assert(BENCH_K % 32 == 0, "K must be a multiple of 32");
_Static_assert(BENCH_VERIFY_FULL == 0 || BENCH_VERIFY_FULL == 1,
               "VERIFY_FULL must be 0 or 1");

#define TILE_M 128u
#define TILE_N 128u
#define TILE_K 32u
#define BF16_BYTES 2u
#define FP32_BYTES 4u
#define MCFG_BF16 0x0aUL
#define MCFG_FP32 0x0cUL
#define BF16_PEAK_OPS_PER_CYCLE 2048u

#define SCHEDULE_NAME "1m2n-a-pingpong-crossb-cpair"

#if BENCH_M == 0u || BENCH_N == 0u || BENCH_K == 0u
#error "BENCH_M, BENCH_N, and BENCH_K must be nonzero"
#endif

#if BENCH_K_BATCH_TILES != 0u
#error "This benchmark measures an unbounded output-stationary reduction; set K_BATCH_TILES=0"
#endif

typedef struct {
  uint64_t cycles;
  uint64_t instret;
  uint64_t macs;
  uint64_t mmaccs;
  uint64_t c_stores;
  uint64_t sync_stores;
  uint64_t sync_store_bytes;
} BenchmarkResult;

typedef struct {
  uint32_t *c0;
  uint32_t *c1;
  unsigned int c_base;
  unsigned int n_count;
  unsigned long c_stride;
  bool valid;
  bool c0_issued;
  bool c1_issued;
} PendingStores;

/* The operands are real, nonzero BF16 values. C remains FP32 throughout. */
static uint16_t matrix_a[(size_t)BENCH_M * BENCH_K] __attribute__((aligned(64)));
static uint16_t matrix_b[(size_t)BENCH_N * BENCH_K] __attribute__((aligned(64)));
static uint32_t matrix_c[(size_t)BENCH_M * BENCH_N] __attribute__((aligned(64)));

static const int8_t kValueCodes[4] = {-2, -1, 1, 2};
static const uint16_t kBf16Values[4] = {
    0xbf80u, /* -1.0 */
    0xbf00u, /* -0.5 */
    0x3f00u, /* +0.5 */
    0x3f80u, /* +1.0 */
};

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

static inline unsigned int a_code_index(unsigned int row, unsigned int k) {
  return (row * 5u + k * 3u + 1u) & 3u;
}

static inline unsigned int b_code_index(unsigned int column, unsigned int k) {
  return (column * 7u + k * 11u + 2u) & 3u;
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
#define MATRIX_MACC_00() asm volatile("mmacc acc0, tr2, tr0" ::: "memory")
#define MATRIX_MACC_01() asm volatile("mmacc acc1, tr3, tr0" ::: "memory")
#define MATRIX_MACC_ACC0_A1() asm volatile("mmacc acc0, tr2, tr1" ::: "memory")
#define MATRIX_MACC_ACC1_A1() asm volatile("mmacc acc1, tr3, tr1" ::: "memory")
#define MATRIX_MACC_ACC2_A0() asm volatile("mmacc acc2, tr2, tr0" ::: "memory")
#define MATRIX_MACC_ACC3_A0() asm volatile("mmacc acc3, tr3, tr0" ::: "memory")
#define MATRIX_MACC_ACC2_A1() asm volatile("mmacc acc2, tr2, tr1" ::: "memory")
#define MATRIX_MACC_ACC3_A1() asm volatile("mmacc acc3, tr3, tr1" ::: "memory")
#define MATRIX_MACC_ACC0_B1_A0() \
  asm volatile("mmacc acc0, tr3, tr0" ::: "memory")
#define MATRIX_MACC_ACC1_B0_A0() \
  asm volatile("mmacc acc1, tr2, tr0" ::: "memory")
#define MATRIX_MACC_ACC0_B1_A1() \
  asm volatile("mmacc acc0, tr3, tr1" ::: "memory")
#define MATRIX_MACC_ACC1_B0_A1() \
  asm volatile("mmacc acc1, tr2, tr1" ::: "memory")
#define MATRIX_MACC_ACC2_B1_A0() \
  asm volatile("mmacc acc2, tr3, tr0" ::: "memory")
#define MATRIX_MACC_ACC3_B0_A0() \
  asm volatile("mmacc acc3, tr2, tr0" ::: "memory")
#define MATRIX_MACC_ACC2_B1_A1() \
  asm volatile("mmacc acc2, tr3, tr1" ::: "memory")
#define MATRIX_MACC_ACC3_B0_A1() \
  asm volatile("mmacc acc3, tr2, tr1" ::: "memory")
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

static inline void set_tile(unsigned int m, unsigned int n, unsigned int k) {
  asm volatile("msettilem %0\n\t"
               "msettilen %1\n\t"
               "msettilek %2\n\t"
               :
               : "r"((unsigned long)m), "r"((unsigned long)n),
                 "r"((unsigned long)k)
               : "memory");
}

static void configure_matrix(void) {
  const unsigned long mstatus_mask = (1UL << 25) | (1UL << 13) | (1UL << 9);
  const unsigned long a_b_type = MCFG_BF16;
  const unsigned long c_type = MCFG_FP32;

  asm volatile("csrs mstatus, %0\n\t"
               "csrwi vcsr, 0\n\t"
               :
               : "r"(mstatus_mask)
               : "memory");
  asm volatile("msetcfg mcfg0, %0\n\t"
               "msetcfg mcfg1, %0\n\t"
               "msetcfg mcfg2, %0\n\t"
               "msetcfg mcfg3, %0\n\t"
               "msetcfg mcfg4, %1\n\t"
               "msetcfg mcfg5, %1\n\t"
               "msetcfg mcfg6, %1\n\t"
               "msetcfg mcfg7, %1\n\t"
               :
               : "r"(a_b_type), "r"(c_type)
               : "memory");
}

static void initialize_operands(void) {
  for (unsigned int m = 0; m < BENCH_M; ++m) {
    for (unsigned int k = 0; k < BENCH_K; ++k) {
      matrix_a[(size_t)m * BENCH_K + k] =
          kBf16Values[a_code_index(m, k)];
    }
  }
  for (unsigned int n = 0; n < BENCH_N; ++n) {
    for (unsigned int k = 0; k < BENCH_K; ++k) {
      matrix_b[(size_t)n * BENCH_K + k] =
          kBf16Values[b_code_index(n, k)];
    }
  }
  asm volatile("fence rw, rw" ::: "memory");
}

static inline void zero_output_pair(unsigned int c_base,
                                    unsigned int n_count) {
  if (c_base == 0u) {
    MATRIX_ZERO_ACC0();
    if (n_count == 2u) MATRIX_ZERO_ACC1();
  } else {
    MATRIX_ZERO_ACC2();
    if (n_count == 2u) MATRIX_ZERO_ACC3();
  }
}

static inline void issue_mmacc_n0(unsigned int c_base,
                                  unsigned int a_buffer,
                                  unsigned int b_buffer,
                                  BenchmarkResult *result) {
  if (c_base == 0u) {
    if (a_buffer == 0u) {
      if (b_buffer == 0u) {
        MATRIX_MACC_00();
      } else {
        MATRIX_MACC_ACC0_B1_A0();
      }
    } else {
      if (b_buffer == 0u) {
        MATRIX_MACC_ACC0_A1();
      } else {
        MATRIX_MACC_ACC0_B1_A1();
      }
    }
  } else {
    if (a_buffer == 0u) {
      if (b_buffer == 0u) {
        MATRIX_MACC_ACC2_A0();
      } else {
        MATRIX_MACC_ACC2_B1_A0();
      }
    } else {
      if (b_buffer == 0u) {
        MATRIX_MACC_ACC2_A1();
      } else {
        MATRIX_MACC_ACC2_B1_A1();
      }
    }
  }
  ++result->mmaccs;
}

static inline void issue_mmacc_n1(unsigned int c_base,
                                  unsigned int a_buffer,
                                  unsigned int b_buffer,
                                  BenchmarkResult *result) {
  if (c_base == 0u) {
    if (a_buffer == 0u) {
      if (b_buffer == 0u) {
        MATRIX_MACC_ACC1_B0_A0();
      } else {
        MATRIX_MACC_01();
      }
    } else {
      if (b_buffer == 0u) {
        MATRIX_MACC_ACC1_B0_A1();
      } else {
        MATRIX_MACC_ACC1_A1();
      }
    }
  } else {
    if (a_buffer == 0u) {
      if (b_buffer == 0u) {
        MATRIX_MACC_ACC3_B0_A0();
      } else {
        MATRIX_MACC_ACC3_A0();
      }
    } else {
      if (b_buffer == 0u) {
        MATRIX_MACC_ACC3_B0_A1();
      } else {
        MATRIX_MACC_ACC3_A1();
      }
    }
  }
  ++result->mmaccs;
}

static inline void load_a_buffer(unsigned int buffer, const uint16_t *ptr,
                                 unsigned long stride) {
  if (buffer == 0u) {
    MATRIX_LOAD_A0(ptr, stride);
  } else {
    MATRIX_LOAD_A1(ptr, stride);
  }
}

static inline void load_b_buffer(unsigned int buffer, const uint16_t *ptr,
                                 unsigned long stride) {
  if (buffer == 0u) {
    MATRIX_LOAD_B0(ptr, stride);
  } else {
    MATRIX_LOAD_B1(ptr, stride);
  }
}

static inline void store_acc(unsigned int acc, uint32_t *ptr,
                             unsigned long stride) {
  switch (acc) {
    case 0u: MATRIX_STORE_C0(ptr, stride); break;
    case 1u: MATRIX_STORE_C1(ptr, stride); break;
    case 2u: MATRIX_STORE_C2(ptr, stride); break;
    default: MATRIX_STORE_C3(ptr, stride); break;
  }
}

/* tr0/tr1 and tr2/tr3 are independent A/B operand pairs. */

static inline void issue_pending_stores(PendingStores *pending,
                                        unsigned int completed_k_tiles,
                                        BenchmarkResult *result) {
  if (!pending->valid) return;
  if (!pending->c0_issued && completed_k_tiles >= 1u) {
    store_acc(pending->c_base, pending->c0, pending->c_stride);
    pending->c0_issued = true;
    ++result->c_stores;
  }
  if (pending->n_count == 2u && !pending->c1_issued &&
      completed_k_tiles >= 2u) {
    store_acc(pending->c_base + 1u, pending->c1, pending->c_stride);
    pending->c1_issued = true;
    ++result->c_stores;
  }
}

static inline void flush_pending_stores(PendingStores *pending,
                                        BenchmarkResult *result) {
  if (!pending->valid) return;
  if (!pending->c0_issued) {
    store_acc(pending->c_base, pending->c0, pending->c_stride);
    pending->c0_issued = true;
    ++result->c_stores;
  }
  if (pending->n_count == 2u && !pending->c1_issued) {
    store_acc(pending->c_base + 1u, pending->c1, pending->c_stride);
    pending->c1_issued = true;
    ++result->c_stores;
  }
}

/*
 * One M tile and up to two N tiles stay output-stationary for the full K
 * reduction. tr0/tr1 ping-pong A across K; every loaded A tile therefore
 * feeds both B panels before it can be overwritten. Each output accumulator retains the entire
 * reduction while the other CReg pair drains its final stores.
 */
static void issue_mtile_npair_block(unsigned int m_base, unsigned int n_base,
                                    unsigned int m_extent,
                                    unsigned int n_extent,
                                    unsigned int n_count,
                                    unsigned int c_base,
                                    PendingStores *pending,
                                    BenchmarkResult *result) {
  const unsigned long a_stride = (unsigned long)BENCH_K * BF16_BYTES;
  const unsigned long b_stride = (unsigned long)BENCH_K * BF16_BYTES;
  const unsigned long c_stride = (unsigned long)BENCH_N * FP32_BYTES;
  const unsigned int first_k_extent = BENCH_K < TILE_K ? BENCH_K : TILE_K;
  const uint16_t *const a_base = matrix_a + (size_t)m_base * BENCH_K;
  const uint16_t *const b0_base = matrix_b + (size_t)n_base * BENCH_K;

  set_tile(m_extent, n_extent, first_k_extent);
  zero_output_pair(c_base, n_count);

  load_a_buffer(0u, a_base, a_stride);
  load_b_buffer(0u, b0_base, b_stride);
  if (n_count == 2u) {
    load_b_buffer(1u, b0_base + (size_t)TILE_N * BENCH_K, b_stride);
  }

  bool operands_ready = true;
  unsigned int k_tile_index = 0u;
  for (unsigned int k_base = 0; k_base < BENCH_K; ++k_tile_index) {
    const unsigned int k_extent =
        BENCH_K - k_base < TILE_K ? BENCH_K - k_base : TILE_K;
    const unsigned int a_buffer = k_tile_index & 1u;
    const unsigned int n0_b_buffer = k_tile_index & 1u;
    const unsigned int n1_b_buffer = n0_b_buffer ^ 1u;

    if (k_extent != first_k_extent) {
      set_tile(m_extent, n_extent, k_extent);
    }
    if (!operands_ready) {
      const uint16_t *const current_a = a_base + k_base;
      const uint16_t *const current_b0 = b0_base + k_base;

      load_a_buffer(a_buffer, current_a, a_stride);
      load_b_buffer(n0_b_buffer, current_b0, b_stride);
      if (n_count == 2u) {
        load_b_buffer(n1_b_buffer,
                      current_b0 + (size_t)TILE_N * BENCH_K, b_stride);
      }
      operands_ready = true;
    }
    /*
     * Retire the B registers in cross order. Once N0 consumes its B register,
     * turn it into next K's N1 panel; it then has the current N1 and next N0
     * compute lifecycles to fill. The other B register receives next K's N0
     * after N1 retires. This keeps the slower B loads independent from their
     * consumer for as long as the four matrix registers allow.
     */
    issue_mmacc_n0(c_base, a_buffer, n0_b_buffer, result);
    k_base += k_extent;

    if (n_count == 2u) {
      if (k_base < BENCH_K) {
        const unsigned int next_k_extent =
            BENCH_K - k_base < TILE_K ? BENCH_K - k_base : TILE_K;
        const uint16_t *const next_b0 = b0_base + k_base;
        if (next_k_extent == k_extent) {
          load_b_buffer(n0_b_buffer,
                        next_b0 + (size_t)TILE_N * BENCH_K, b_stride);
        }
      }
      issue_mmacc_n1(c_base, a_buffer, n1_b_buffer, result);
      if (k_base < BENCH_K) {
        const unsigned int next_k_extent =
            BENCH_K - k_base < TILE_K ? BENCH_K - k_base : TILE_K;
        const uint16_t *const next_b0 = b0_base + k_base;
        if (next_k_extent == k_extent) {
          load_b_buffer(n1_b_buffer, next_b0, b_stride);
        }
      }
    } else if (k_base < BENCH_K) {
      const unsigned int next_k_extent =
          BENCH_K - k_base < TILE_K ? BENCH_K - k_base : TILE_K;
      const uint16_t *const next_b0 = b0_base + k_base;
      if (next_k_extent == k_extent) {
        load_b_buffer(n1_b_buffer, next_b0, b_stride);
      }
    }

    if (k_base < BENCH_K) {
      const unsigned int next_k_extent =
          BENCH_K - k_base < TILE_K ? BENCH_K - k_base : TILE_K;
      const uint16_t *const next_a = a_base + k_base;
      if (next_k_extent == k_extent) {
        load_a_buffer(a_buffer ^ 1u, next_a, a_stride);
      } else {
        operands_ready = false;
      }
    }

    issue_pending_stores(pending, k_tile_index + 1u, result);
  }

  uint32_t *const c00 = matrix_c + (size_t)m_base * BENCH_N + n_base;
  flush_pending_stores(pending, result);
  *pending = (PendingStores){
      .c0 = c00,
      .c1 = c00 + TILE_N,
      .c_base = c_base,
      .n_count = n_count,
      .c_stride = c_stride,
      .valid = true,
  };
}

static BenchmarkResult run_gemm_npair(void) {
  BenchmarkResult result = {0};
  unsigned long sync_target = 0;
  unsigned int c_base = 0u;
  PendingStores pending = {0};

  configure_matrix();
  MATRIX_SYNC_RESET();
  MATRIX_FENCE();
  const uint64_t cycle_begin = read_mcycle();
  const uint64_t instret_begin = read_minstret();

  for (unsigned int m_base = 0; m_base < BENCH_M;) {
    const unsigned int m_extent =
        BENCH_M - m_base < TILE_M ? BENCH_M - m_base : TILE_M;

    for (unsigned int n_base = 0; n_base < BENCH_N;) {
      const unsigned int n_remaining = BENCH_N - n_base;
      const unsigned int n_count = n_remaining >= 2u * TILE_N ? 2u : 1u;
      const unsigned int n_extent = n_remaining < TILE_N ? n_remaining : TILE_N;

      issue_mtile_npair_block(m_base, n_base, m_extent, n_extent, n_count,
                              c_base,
                              &pending,
                              &result);
      n_base += n_count * TILE_N;
      c_base ^= 2u;
    }
    m_base += m_extent;
  }

  flush_pending_stores(&pending, &result);

  /* One kernel-level drain makes all final C stores visible to verification. */
  MATRIX_RELEASE();
  ++sync_target;
  MATRIX_ACQUIRE(sync_target);
  result.cycles = read_mcycle() - cycle_begin;
  result.instret = read_minstret() - instret_begin;
  result.macs = (uint64_t)BENCH_M * BENCH_N * BENCH_K;
  return result;
}

static BenchmarkResult run_gemm(void) {
  return run_gemm_npair();
}

static uint32_t expected_output_bits(unsigned int row, unsigned int column) {
  int32_t scaled_sum = 0;
  union {
    float f;
    uint32_t u;
  } result;

  for (unsigned int k = 0; k < BENCH_K; ++k) {
    const int32_t a = kValueCodes[a_code_index(row, k)];
    const int32_t b = kValueCodes[b_code_index(column, k)];
    scaled_sum += a * b;
  }
  result.f = (float)scaled_sum * 0.25f;
  return result.u;
}

static bool verify_one(unsigned int row, unsigned int column) {
  const uint32_t actual = matrix_c[(size_t)row * BENCH_N + column];
  const uint32_t expected = expected_output_bits(row, column);

  if (actual != expected) {
    printf("[FAIL] C[%u][%u] got=0x%08x expected=0x%08x\n",
           row, column, actual, expected);
    return false;
  }
  return true;
}

static bool verify_output(void) {
#if BENCH_VERIFY_FULL
  for (unsigned int m = 0; m < BENCH_M; ++m) {
    for (unsigned int n = 0; n < BENCH_N; ++n) {
      if (!verify_one(m, n)) return false;
    }
  }
  return true;
#else
  const unsigned int samples[][2] = {
      {0u, 0u},
      {0u, BENCH_N - 1u},
      {BENCH_M - 1u, 0u},
      {BENCH_M - 1u, BENCH_N - 1u},
      {BENCH_M / 2u, BENCH_N / 2u},
  };

  for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i) {
    if (!verify_one(samples[i][0], samples[i][1])) return false;
  }
  for (unsigned int m = 0; m < BENCH_M; m += TILE_M) {
    const unsigned int row = m + (m + 37u) % (BENCH_M - m < TILE_M
                                                    ? BENCH_M - m : TILE_M);
    const unsigned int column = (m * 13u + 29u) % BENCH_N;
    if (!verify_one(row, column)) return false;
  }
  for (unsigned int n = 0; n < BENCH_N; n += TILE_N) {
    const unsigned int row = (n * 17u + 11u) % BENCH_M;
    const unsigned int column = n + (n + 19u) % (BENCH_N - n < TILE_N
                                                       ? BENCH_N - n : TILE_N);
    if (!verify_one(row, column)) return false;
  }
  return true;
#endif
}

/* Keep extra coverage out of main's register allocation and timed kernel. */
static __attribute__((noinline)) bool verify_output_tiles(void) {
  for (unsigned int mt = 0; mt < BENCH_M / TILE_M; ++mt) {
    for (unsigned int nt = 0; nt < BENCH_N / TILE_N; ++nt) {
      const unsigned int row = mt * TILE_M + (mt * 37u + nt * 17u) % TILE_M;
      const unsigned int column = nt * TILE_N + (mt * 29u + nt * 43u) % TILE_N;
      if (!verify_one(row, column)) return false;
    }
  }
  printf("[PASS] every BF16 output tile checked\n");
  return true;
}

static void print_result(const BenchmarkResult *result) {
  const uint64_t operations = result->macs * 2u;
  const uint64_t utilization_bp = result->cycles == 0u ? 0u :
      operations * 10000u / (result->cycles * BF16_PEAK_OPS_PER_CYCLE);

  printf("BF16_RESULT case=%s schedule=%s m=%u n=%u k=%u k_batch_tiles=%u "
         "cycles=%llu instret=%llu macs=%llu operations=%llu mmaccs=%llu "
         "c_stores=%llu sync_stores=%llu sync_store_bytes=%llu "
         "peak_ops_per_cycle=%u utilization_bp=%llu\n",
         BENCH_NAME, SCHEDULE_NAME, BENCH_M, BENCH_N, BENCH_K,
         BENCH_K_BATCH_TILES,
         (unsigned long long)result->cycles,
         (unsigned long long)result->instret,
         (unsigned long long)result->macs,
         (unsigned long long)operations,
         (unsigned long long)result->mmaccs,
         (unsigned long long)result->c_stores,
         (unsigned long long)result->sync_stores,
         (unsigned long long)result->sync_store_bytes,
         BF16_PEAK_OPS_PER_CYCLE,
         (unsigned long long)utilization_bp);
}

int main(void) {
  printf("BF16_GEMM case=%s M=%u N=%u K=%u tile=128x128x32 k_batch_tiles=%u\n",
         BENCH_NAME, BENCH_M, BENCH_N, BENCH_K, BENCH_K_BATCH_TILES);
  initialize_operands();
  const BenchmarkResult result = run_gemm();
  if (!verify_output()) return 1;
  print_result(&result);
  printf("[PASS] BF16 GEMM case=%s M=%u N=%u K=%u\n",
         BENCH_NAME, BENCH_M, BENCH_N, BENCH_K);
  return verify_output_tiles() ? 0 : 1;
}
