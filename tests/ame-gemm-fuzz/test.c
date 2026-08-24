#include <stdint.h>
#include <string.h>
#include <klib.h>
#include "ame.h"

static int8_t A[AME_TILE_M * AME_TILE_K] __attribute__((aligned(64)));
static int8_t B[AME_TILE_N * AME_TILE_K] __attribute__((aligned(64)));
static int32_t C_ame[AME_TILE_M * AME_TILE_N] __attribute__((aligned(64)));
static int32_t C_ref[AME_TILE_M * AME_TILE_N] __attribute__((aligned(64)));

static uint32_t rng_state = 0x12345678u;

static uint32_t next_rand(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state;
}

static void reference_gemm_i8_i32_bT(const int8_t *a, const int8_t *b, int32_t *c) {
  for (int row = 0; row < AME_TILE_M; row++) {
    for (int col = 0; col < AME_TILE_N; col++) {
      int32_t sum = 0;
      for (int kk = 0; kk < AME_TILE_K; kk++) {
        sum += (int32_t)a[row * AME_TILE_K + kk] *
               (int32_t)b[col * AME_TILE_K + kk];
      }
      c[row * AME_TILE_N + col] = sum;
    }
  }
}

static void fill_case(int test_id) {
  if (test_id == 0) {
    memset(A, 0, sizeof(A));
    memset(B, 0, sizeof(B));
    return;
  }
  if (test_id == 1) {
    memset(A, 1, sizeof(A));
    memset(B, 1, sizeof(B));
    return;
  }
  if (test_id == 2) {
    memset(A, 127, sizeof(A));
    memset(B, 127, sizeof(B));
    return;
  }
  if (test_id == 3) {
    memset(A, -128, sizeof(A));
    memset(B, -128, sizeof(B));
    return;
  }

  for (int i = 0; i < AME_TILE_M * AME_TILE_K; i++) {
    A[i] = (int8_t)(next_rand() & 0xff);
  }
  for (int i = 0; i < AME_TILE_N * AME_TILE_K; i++) {
    B[i] = (int8_t)(next_rand() & 0xff);
  }
}

static int compare_results(int test_id) {
  int errors = 0;
  int first = -1;

  for (int i = 0; i < AME_TILE_M * AME_TILE_N; i++) {
    if (C_ame[i] != C_ref[i]) {
      if (first < 0) {
        first = i;
      }
      errors++;
    }
  }

  if (errors != 0) {
    printf("[FAIL] case %d: %d mismatches, first index %d AME=%d REF=%d\n",
           test_id, errors, first, C_ame[first], C_ref[first]);
    return 1;
  }

  printf("[PASS] case %d\n", test_id);
  return 0;
}

int main() {
  const int num_tests = 20;
  int failures = 0;

  ame_init();
  printf("AME proposal-14 GEMM fuzz test: C(128x128)=A(128x64)*B^T(128x64)\n");

  for (int t = 0; t < num_tests; t++) {
    fill_case(t);
    memset(C_ame, 0, sizeof(C_ame));
    memset(C_ref, 0, sizeof(C_ref));

    reference_gemm_i8_i32_bT(A, B, C_ref);
    ggml_ame_gemm_tile_i8_i32_bT(A, B, C_ame);

    failures += compare_results(t);
    if (failures > 0) {
      break;
    }
  }

  if (failures == 0) {
    printf("All %d AME proposal-14 GEMM tests PASSED.\n", num_tests);
  }

  return failures == 0 ? 0 : 1;
}
