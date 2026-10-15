#include "bitmte.h"
#include <klib.h>

/* The Makefile selects one of four examples and sizes its static buffers.
 * All use C[M][N] = A[M][K] * B[K][N]; bt[j][k] stores B[k][j].
 */
_Static_assert(BITMTE_M > 0 && BITMTE_M % BMTE_TILE == 0, "M must be a positive multiple of 128");
_Static_assert(BITMTE_N > 0 && BITMTE_N % BMTE_TILE == 0, "N must be a positive multiple of 128");
_Static_assert(BITMTE_K > 0 && BITMTE_K % (512 / BITMTE_B_BITS) == 0,
               "K must be a positive multiple of 256 (ternary) or 512 (binary)");

static int8_t a[BITMTE_M][BITMTE_K];
static int8_t bt[BITMTE_N][BITMTE_K];
static uint8_t packed_a[BITMTE_M][BITMTE_K / (8 / BITMTE_A_BITS)] __attribute__((aligned(64)));
static uint8_t packed_bt[BITMTE_N][BITMTE_K / (8 / BITMTE_B_BITS)] __attribute__((aligned(64)));
static uint32_t c[BITMTE_M][BITMTE_N] __attribute__((aligned(64)));

static uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= UINT32_C(0x7feb352d);
  x ^= x >> 15;
  x *= UINT32_C(0x846ca68b);
  return x ^ (x >> 16);
}

static int prepare(void) {
  const bitmte_mode mode = BITMTE_MODE;
  for (size_t i = 0; i < BITMTE_M; ++i) {
    for (size_t x = 0; x < BITMTE_K; ++x) {
      uint32_t value = mix((uint32_t)i * 65537u + (uint32_t)x + 42u);
      a[i][x] = (int)(value & ((1u << BITMTE_A_BITS) - 1)) - (1 << (BITMTE_A_BITS - 1));
    }
    if (bitmte_pack_a(packed_a[i], a[i], BITMTE_K, mode)) return -1;
  }
  for (size_t j = 0; j < BITMTE_N; ++j) {
    for (size_t x = 0; x < BITMTE_K; ++x) {
      uint32_t value = mix((uint32_t)j * 104729u + (uint32_t)x + 17u);
      bt[j][x] = BITMTE_B_BITS == 2 ? (int)(value % 3) - 1 : (int)(value & 1);
    }
    if (bitmte_pack_b(packed_bt[j], bt[j], BITMTE_K, mode)) return -1;
  }
  for (size_t i = 0; i < BITMTE_M; ++i)
    for (size_t j = 0; j < BITMTE_N; ++j) c[i][j] = UINT32_C(0xdeadbeef);
  return 0;
}

/* Independent reference uses original numerical inputs, never packed bytes,
 * unpack helpers, selectors, or the hardware kernel's blocking scheme.
 */
static int verify(void) {
  uint32_t checksum = 0;
  for (size_t i = 0; i < BITMTE_M; ++i) {
    for (size_t j = 0; j < BITMTE_N; ++j) {
      uint32_t expected = 0;
      for (size_t x = 0; x < BITMTE_K; ++x)
        expected += (uint32_t)((int)a[i][x] * (int)bt[j][x]);
      if (c[i][j] != expected) {
        printf("FAIL %s [%u,%u]: got=%08x expected=%08x\n",
               BITMTE_NAME, (unsigned)i, (unsigned)j, c[i][j], expected);
        return 1;
      }
      checksum += c[i][j];
    }
  }
  printf("PASS %s M=%u N=%u K=%u elements=%u checksum=%08x\n",
         BITMTE_NAME, BITMTE_M, BITMTE_N, BITMTE_K, BITMTE_M * BITMTE_N, checksum);
  return 0;
}

int main(const char *args) {
  (void)args;
  printf("GEMM %s M=%u N=%u K=%u (scalar check enabled)\n",
         BITMTE_NAME, BITMTE_M, BITMTE_N, BITMTE_K);
  if (prepare()) {
    printf("FAIL packing\n");
    return 1;
  }
  bitmte_init();
  if (bitmte_gemm(&c[0][0], &packed_a[0][0], &packed_bt[0][0],
                  BITMTE_M, BITMTE_N, BITMTE_K, BITMTE_MODE)) {
    printf("FAIL kernel arguments\n");
    return 1;
  }
  return verify();
}
