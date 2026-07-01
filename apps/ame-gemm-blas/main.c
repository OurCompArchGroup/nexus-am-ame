#include "xsai_blas.h"
#include <klib.h>
#include <stdint.h>

// Demonstrate driving the xsai-blas int8 GEMM (xsai_gemm_i8_f32) from a
// NEXUS-AM test app, replacing the hand-written matop.S kernels.
//
// xsai_gemm_i8_f32 computes  C = dequant(A x B^T):
//   A:        [M x K] int8, row-major
//   B:        [N x K] int8, row-major (B is stored transposed)
//   scales_a: [M x nb] f32,  scales_b: [N x nb] f32,  nb = K / scale_block_k
//   C:        [N x M] f32,  COLUMN-major  ->  C[n*M + m]
//
// We set every scale to 1.0 so the f32 output equals the raw int32 dot product,
// which keeps verification exact against an integer scalar reference oracle.

#define M 256
#define K 128
#define N 256
#define SCALE_BLOCK_K 64
#define NB (K / SCALE_BLOCK_K)

static int8_t A[M * K];
static int8_t B[N * K];
static float C[N * M];
static float scales_a[M * NB];
static float scales_b[N * NB];

static void fill_inputs(void) {
  // All-1s inputs: every dot product over K terms equals K (=128), so each
  // C entry must be exactly 128. Deterministic and matches the original
  // ame-gemm expectation.
  for (int i = 0; i < M * K; i++)
    A[i] = 1;
  for (int i = 0; i < N * K; i++)
    B[i] = 1;
  for (int i = 0; i < M * NB; i++)
    scales_a[i] = 1.0f;
  for (int i = 0; i < N * NB; i++)
    scales_b[i] = 1.0f;
}

// Scalar integer oracle, written into column-major layout to match the library.
static int verify(void) {
  int bad = 0;
  for (int n = 0; n < N && bad < 8; n++) {
    for (int m = 0; m < M; m++) {
      int32_t expect = 0;
      for (int k = 0; k < K; k++)
        expect += (int32_t)A[m * K + k] * (int32_t)B[n * K + k];
      float got = C[n * M + m];
      if (got != (float)expect) {
        if (bad < 8)
          printf("mismatch C[n=%d,m=%d] got=%d expect=%d\n", n, m, (int)got,
                 (int)expect);
        bad++;
      }
    }
  }
  return bad;
}

int main() {
  printf("GEMM (xsai-blas) Test started! M=%d K=%d N=%d\n", M, K, N);
  fill_inputs();

  size_t ws_sz = xsai_gemm_i8_f32_workspace_size(M, K, N);
  void *ws = malloc(ws_sz);

  xsai_gemm_i8_f32(A, B, scales_a, scales_b, C, M, K, N, SCALE_BLOCK_K, ws,
                   ws_sz);

  if (verify() == 0) {
    printf("C[0]=%d C[last]=%d (expect %d)\n", (int)C[0], (int)C[N * M - 1], K);
    printf("GEMM (xsai-blas) Test passed!\n");
  } else {
    printf("GEMM (xsai-blas) Test failed!\n");
  }

  free(ws);
  return 0;
}
