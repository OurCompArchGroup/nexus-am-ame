#include <am.h>
#include <klib.h>
#include <stdint.h>

#define MATRIX_M 128u
#define MATRIX_N 128u
#define MATRIX_K 64u
#define PACKED_B_ROW_BYTES (MATRIX_K / 4u)

#define MCFG_INT8 0x02UL
#define MCFG_INT32 0x04UL
#define MCFG_FP2PACK4 0x0dUL

static int8_t matrix_a[MATRIX_M][MATRIX_K] __attribute__((aligned(64)));
static uint8_t matrix_b[MATRIX_N][PACKED_B_ROW_BYTES]
    __attribute__((aligned(4096)));
static int32_t matrix_c[MATRIX_M][MATRIX_N] __attribute__((aligned(64)));

_Static_assert(sizeof(matrix_a) == 8192, "A must be one full i8 tile");
_Static_assert(sizeof(matrix_b) == 2048,
               "B must be one dense fp2pack4 panel");
_Static_assert(sizeof(matrix_c) == 65536, "C must be one full i32 tile");
static int8_t make_a_value(unsigned int m, unsigned int k) {
  int8_t activation = m < MATRIX_K ? 1 : -3;
  return (m % MATRIX_K) == k ? activation : 0;
}

/*
 * The rotation in the first four packed bytes encodes all seven N bits, so
 * every B row has a distinct 64-lane pattern. Every byte is a rotation of
 * codes {0,1,2,3}, which guarantees coverage of all signed-i2 values.
 */
static uint8_t make_b_code(unsigned int n, unsigned int k) {
  unsigned int byte = k >> 2;
  unsigned int rotation;

  if (byte < 4u) {
    rotation = (n >> (2u * byte)) & 0x3u;
  } else {
    rotation = (n + 3u * byte + (n >> (byte & 7u))) & 0x3u;
  }
  return (uint8_t)(((k & 0x3u) + rotation) & 0x3u);
}

static int8_t decode_i2(uint8_t code) {
  code &= 0x3u;
  return (int8_t)((code & 0x2u) ? (int)code - 4 : (int)code);
}

static int32_t initial_c_value(unsigned int m, unsigned int n) {
  return (int32_t)((m * 19u + n * 23u + 7u) % 257u) - 128;
}

static int initialize_matrices(void) {
  unsigned int seen_codes = 0;

  for (unsigned int m = 0; m < MATRIX_M; ++m) {
    for (unsigned int k = 0; k < MATRIX_K; ++k) {
      matrix_a[m][k] = make_a_value(m, k);
    }
  }

  for (unsigned int n = 0; n < MATRIX_N; ++n) {
    for (unsigned int byte = 0; byte < PACKED_B_ROW_BYTES; ++byte) {
      uint8_t packed = 0;
      for (unsigned int lane = 0; lane < 4u; ++lane) {
        unsigned int k = byte * 4u + lane;
        uint8_t code = make_b_code(n, k);
        packed |= (uint8_t)(code << (2u * lane));
        seen_codes |= 1u << code;
      }
      matrix_b[n][byte] = packed;
    }
  }

  for (unsigned int m = 0; m < MATRIX_M; ++m) {
    for (unsigned int n = 0; n < MATRIX_N; ++n) {
      matrix_c[m][n] = initial_c_value(m, n);
    }
  }

  return seen_codes == 0x0fu ? 0 : 1;
}

static void configure_matrix_unit(void) {
  const unsigned long mstatus_mask =
      (1UL << 25) | (1UL << 13) | (1UL << 9);

  asm volatile("csrs mstatus, %0\n\t"
               "csrwi vcsr, 0\n\t"
               :
               : "r"(mstatus_mask)
               : "memory");

  asm volatile("msettilem %0\n\t"
               "msettilen %1\n\t"
               "msettilek %2\n\t"
               :
               : "r"((unsigned long)MATRIX_M), "r"((unsigned long)MATRIX_N),
                 "r"((unsigned long)MATRIX_K)
               : "memory");
  asm volatile("msetcfg mcfg0, %0\n\t"
               "msetcfg mcfg1, %1\n\t"
               "msetcfg mcfg4, %2\n\t"
               :
               : "r"(MCFG_INT8), "r"(MCFG_FP2PACK4), "r"(MCFG_INT32)
               : "memory");
}

static void run_matrix_multiply(void) {
  const unsigned long a_stride = MATRIX_K;
  const unsigned long b_stride = PACKED_B_ROW_BYTES;
  const unsigned long c_stride = MATRIX_N * sizeof(int32_t);
  const unsigned long sync_done = 1;

  asm volatile("msyncregreset sync1\n\t" ::: "memory");
  asm volatile("mfence\n\t" ::: "memory");
  asm volatile("mla tr0, (%0), %1\n\t"
               :
               : "r"(matrix_a), "r"(a_stride)
               : "memory");
  asm volatile("mlb tr1, (%0), %1\n\t"
               :
               : "r"(matrix_b), "r"(b_stride)
               : "memory");
  asm volatile("mlc acc0, (%0), %1\n\t"
               :
               : "r"(matrix_c), "r"(c_stride)
               : "memory");
  asm volatile("mmacc acc0, tr1, tr0\n\t" ::: "memory");
  asm volatile("msc acc0, (%0), %1\n\t"
               :
               : "r"(matrix_c), "r"(c_stride)
               : "memory");
  asm volatile("mrelease sync1\n\t" ::: "memory");
  asm volatile("macquire sync1, %0\n\t"
               :
               : "r"(sync_done)
               : "memory");
  asm volatile("mfence\n\t" ::: "memory");
}

static int verify_result(void) {
  for (unsigned int m = 0; m < MATRIX_M; ++m) {
    for (unsigned int n = 0; n < MATRIX_N; ++n) {
      int32_t expected = initial_c_value(m, n);
      unsigned int k = m % MATRIX_K;
      uint8_t packed = matrix_b[n][k >> 2];
      uint8_t code = (packed >> (2u * (k & 0x3u))) & 0x3u;
      expected += (int32_t)matrix_a[m][k] * (int32_t)decode_i2(code);

      if (matrix_c[m][n] != expected) {
        printf("[FAIL] C[%u][%u]: got %d, expected %d\n", m, n,
               matrix_c[m][n], expected);
        return 1;
      }
    }
  }
  return 0;
}

int main(void) {
  printf("AME i8 x fp2pack4 -> i32 directed test\n");

  if (initialize_matrices() != 0) {
    printf("[FAIL] packed-B generator did not cover all i2 codes\n");
    return 1;
  }

  configure_matrix_unit();
  run_matrix_multiply();

  if (verify_result() != 0) {
    return 1;
  }

  printf("[PASS] full 128x64 packed-B tile\n");
  /* The AM trap runtime converts this return value to _halt(0)/GOOD_TRAP. */
  return 0;
}
