#include <stdint.h>
#include <string.h>
#include <klib.h>
#include "ame.h"

#if defined(__clang__)
#define XSAI_NOINLINE __attribute__((noinline))
#define XSAI_SCALAR_LOOP \
  _Pragma("clang loop vectorize(disable)") \
  _Pragma("clang loop interleave(disable)")
#else
#define XSAI_NOINLINE
#define XSAI_SCALAR_LOOP
#endif

static uint8_t A[AME_TILE_M * AME_A_I2_STRIDE_BYTES] __attribute__((aligned(64)));
static int8_t B[AME_TILE_N * AME_TILE_K] __attribute__((aligned(64)));
static int32_t C_ame[AME_TILE_M * AME_TILE_N] __attribute__((aligned(64)));
static int32_t C_ref[AME_TILE_M * AME_TILE_N] __attribute__((aligned(64)));

static uint32_t rng_state = 0x23456789u;

static uint32_t next_rand(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state;
}

static inline uint8_t encode_i2(int8_t v) {
  return (uint8_t)v & 0x3;
}

static void set_a_i2(int row, int k, int8_t value) {
  const int byte_idx = row * AME_A_I2_STRIDE_BYTES + k / AME_I2_PACK;
  const int bit_off = (k & (AME_I2_PACK - 1)) * 2;
  A[byte_idx] &= (uint8_t)~(0x3u << bit_off);
  A[byte_idx] |= (uint8_t)(encode_i2(value) << bit_off);
}

static void fill_ref_constant(int32_t value) {
  XSAI_SCALAR_LOOP
  for (int i = 0; i < AME_TILE_M * AME_TILE_N; i++) {
    C_ref[i] = value;
  }
}

static void fill_ref_row_col(const int8_t *a_by_row, const int8_t *b_by_col) {
  XSAI_SCALAR_LOOP
  for (int row = 0; row < AME_TILE_M; row++) {
    XSAI_SCALAR_LOOP
    for (int col = 0; col < AME_TILE_N; col++) {
      C_ref[row * AME_TILE_N + col] =
          (int32_t)a_by_row[row] * (int32_t)b_by_col[col] * AME_TILE_K;
    }
  }
}

static void fill_ref_k_pattern(const int8_t *a_vals, const int8_t *b_vals) {
  int32_t dot_by_phase[4][4];

  for (int a_phase = 0; a_phase < 4; a_phase++) {
    for (int b_phase = 0; b_phase < 4; b_phase++) {
      int32_t dot4 = 0;
      for (int kk = 0; kk < 4; kk++) {
        dot4 += (int32_t)a_vals[(a_phase + kk) & 3] *
                (int32_t)b_vals[(b_phase + kk) & 3];
      }
      dot_by_phase[a_phase][b_phase] = dot4 * (AME_TILE_K / 4);
    }
  }

  XSAI_SCALAR_LOOP
  for (int row = 0; row < AME_TILE_M; row++) {
    XSAI_SCALAR_LOOP
    for (int col = 0; col < AME_TILE_N; col++) {
      C_ref[row * AME_TILE_N + col] = dot_by_phase[row & 3][col & 3];
    }
  }
}

static void fill_case(int test_id) {
  memset(A, 0, sizeof(A));
  memset(B, 0, sizeof(B));
  memset(C_ref, 0, sizeof(C_ref));

  if (test_id == 0) {
    fill_ref_constant(0);
    return;
  }

  if (test_id == 1) {
    for (int row = 0; row < AME_TILE_M; row++) {
      for (int kk = 0; kk < AME_TILE_K; kk++) {
        set_a_i2(row, kk, 1);
      }
    }
    memset(B, 1, sizeof(B));
    fill_ref_constant(AME_TILE_K);
    return;
  }

  if (test_id == 2) {
    for (int row = 0; row < AME_TILE_M; row++) {
      for (int kk = 0; kk < AME_TILE_K; kk++) {
        set_a_i2(row, kk, -1);
      }
    }
    memset(B, -1, sizeof(B));
    fill_ref_constant(AME_TILE_K);
    return;
  }

  if (test_id == 3) {
    for (int row = 0; row < AME_TILE_M; row++) {
      for (int kk = 0; kk < AME_TILE_K; kk++) {
        set_a_i2(row, kk, -2);
      }
    }
    memset(B, 1, sizeof(B));
    fill_ref_constant(-2 * AME_TILE_K);
    return;
  }

  if (test_id == 4) {
    static int8_t a_by_row[AME_TILE_M];
    static int8_t b_by_col[AME_TILE_N];
    static const int8_t a_vals[4] = {-2, -1, 0, 1};

    for (int row = 0; row < AME_TILE_M; row++) {
      a_by_row[row] = a_vals[row & 3];
      for (int kk = 0; kk < AME_TILE_K; kk++) {
        set_a_i2(row, kk, a_by_row[row]);
      }
    }
    for (int col = 0; col < AME_TILE_N; col++) {
      b_by_col[col] = (int8_t)((col % 7) - 3);
      for (int kk = 0; kk < AME_TILE_K; kk++) {
        B[col * AME_TILE_K + kk] = b_by_col[col];
      }
    }
    fill_ref_row_col(a_by_row, b_by_col);
    return;
  }

  if (test_id == 5) {
    static const int8_t a_vals[4] = {-2, -1, 0, 1};
    static const int8_t b_vals[4] = {1, -2, 3, -4};

    for (int row = 0; row < AME_TILE_M; row++) {
      for (int kk = 0; kk < AME_TILE_K; kk++) {
        set_a_i2(row, kk, a_vals[(row + kk) & 3]);
      }
    }
    for (int col = 0; col < AME_TILE_N; col++) {
      for (int kk = 0; kk < AME_TILE_K; kk++) {
        B[col * AME_TILE_K + kk] = b_vals[(col + kk) & 3];
      }
    }
    fill_ref_k_pattern(a_vals, b_vals);
    return;
  }

  for (int row = 0; row < AME_TILE_M; row++) {
    for (int kk = 0; kk < AME_TILE_K; kk++) {
      set_a_i2(row, kk, (int8_t)((next_rand() & 0x3u) - 2));
    }
  }
  for (int i = 0; i < AME_TILE_N * AME_TILE_K; i++) {
    B[i] = (int8_t)(next_rand() & 0xffu);
  }
}

static int compare_results(int test_id) {
  int errors = 0;
  int first = -1;

  XSAI_SCALAR_LOOP
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
  const int num_tests = 6;
  int failures = 0;

  ame_i2_init();
  printf("AME proposal-12 GEMM i2*i8->i32 test: C(128x128)=A(128x64)*B^T(128x64)\n");
  printf("A uses mcfg fp2pack4(0x0d) as signed int2, packed four lanes per byte.\n");

  for (int t = 0; t < num_tests; t++) {
    fill_case(t);
    memset(C_ame, 0, sizeof(C_ame));

    ggml_ame_gemm_tile_i2_i8_i32_bT(A, B, C_ame);

    failures += compare_results(t);
    if (failures > 0) {
      break;
    }
  }

  if (failures == 0) {
    printf("All %d AME proposal-12 i2 GEMM tests PASSED.\n", num_tests);
  }

  return failures == 0 ? 0 : 1;
}
