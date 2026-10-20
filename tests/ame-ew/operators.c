#include "ame-ew.h"

/* These are operator building blocks using only explicit physical registers.
 * They deliberately keep allocation/lifetime visible; no matrix calling ABI
 * or automatic spill behavior is assumed across arbitrary C function calls.
 */
static uint32_t wide_image[AME_SLOTS] __attribute__((aligned(64)));

static void linear_epilogue(void) {
  enum { M = 3, N = 5 };
  int32_t x[M][N], residual[M][N], bias[N] = {-3, 1, 4, -1, 2};
  int32_t gain[M] = {1, 2, 3}, result[M][N];
  case_begin("operator: column bias + residual + ReLU + row gain");
  for (unsigned i = 0; i < M; i++)
    for (unsigned j = 0; j < N; j++) {
      x[i][j] = (int)i * 5 + (int)j - 7;
      residual[i][j] = 2 - (int)j;
    }
  AME_CSR_WRITE(0x80a, 0);
  AME_CFG(mcfg4, AME_I32);
  AME_CFG(mcfg5, AME_I32);
  ame_shape(M, N, 1);
  AME_LOAD(mlc, acc0, x, N * 4);
  AME_LOAD(mlc, acc1, residual, N * 4);
  AME_LOAD(vl.row, lv7, bias, 4);
  AME_LOAD(vl.col, lv23, gain, 4);
  AME_EW(madd.row, acc0, lv7, acc0);
  AME_EW(madd, acc0, acc1, acc0);
  AME_X(mmax.x, acc0, 0, acc0);
  AME_EW(mmul.col, acc0, lv23, acc0);
  AME_STORE(msc, acc0, result, N * 4);
  for (unsigned i = 0; i < M; i++)
    for (unsigned j = 0; j < N; j++) {
      int32_t v = x[i][j] + bias[j] + residual[i][j];
      if (v < 0) v = 0;
      CHECK_U32("fused epilogue", result[i][j], v * gain[i]);
    }
}

static int32_t round_even_small(float x) {
  /* Independent finite reference for this example's small half-integer data. */
  int32_t t = (int32_t)x;
  float fraction = x - (float)t;
  if (fraction > 0.5f || (fraction == 0.5f && (t & 1))) t++;
  if (fraction < -0.5f || (fraction == -0.5f && (t & 1))) t--;
  return t;
}

static void requantize(void) {
  enum { M = 2, N = 7 };
  const int8_t x[M][N] = {{-100, -3, -1, 0, 1, 3, 100},
                          {100, 3, 1, 0, -1, -3, -100}};
  const float bias[N] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  int8_t result[M][N];
  case_begin("operator: INT8 dequantize + FP32 affine + RNE requantize + signed clip");
  for (unsigned i = 0; i < AME_SLOTS; i++) wide_image[i] = 0;
  AME_LOAD_WHOLE(mlc.whole, acc0, wide_image);
  AME_CFG(mcfg4, AME_I8);
  AME_CSR_WRITE(0x809, 0);
  AME_CSR_WRITE(0x80a, 1);
  ame_flags_clear();
  ame_shape(M, N, 1);
  AME_LOAD(mlc, acc0, x, N);
  AME_CVT(mcfg4, acc0, AME_F32);
  AME_X(mmul.x, acc0, f32_bits(2.5f), acc0);
  AME_LOAD(vl.row, lv3, bias, 4);
  AME_EW(madd.row, acc0, lv3, acc0);
  AME_CVT(mcfg4, acc0, AME_I8);
  AME_STORE(msc, acc0, result, N);
  for (unsigned i = 0; i < M; i++)
    for (unsigned j = 0; j < N; j++) {
      int32_t q = round_even_small((float)x[i][j] * 2.5f + bias[j]);
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      CHECK_U32("requantized byte", (uint8_t)result[i][j], (uint8_t)q);
    }
  CHECK_U32("INT8 clip accrued", AME_CSR_READ(0x807), 1);
  CHECK_U32("fractional FCVT accrued NX", AME_CSR_READ(0x808), AME_NX);
  AME_CSR_WRITE(0x80a, 0);
}

static void row_moments(void) {
  enum { M = 3, N = 4 };
  const float input[M][N] = {{-3, -1, 1, 3}, {0, 2, 4, 6}, {1, 1, 1, 1}};
  float means[M], variances[M], centered[M][N];
  case_begin("operator: row mean / centered variance for normalization");
  AME_CFG(mcfg4, AME_F32);
  AME_CFG(mcfg5, AME_F32);
  AME_CFG(mcfg6, AME_F32);
  AME_CFG(mcfg7, AME_F32);
  AME_CSR_WRITE(0x809, 0);
  ame_shape(M, N, 1);
  AME_LOAD(mlc, acc0, input, N * 4);
  AME_X(mmul.x, acc1, f32_bits(0.25f), acc0);
  AME_REDUCE(vredadd.row, lv0, acc1, 0);
  AME_EW(msub.col, acc2, lv0, acc0);
  AME_EW(mmul, acc3, acc2, acc2);
  AME_X(mmul.x, acc3, f32_bits(0.25f), acc3);
  AME_REDUCE(vredadd.row, lv1, acc3, 0);
  AME_STORE(vs.col, lv0, means, 4);
  AME_STORE(vs.col, lv1, variances, 4);
  AME_STORE(msc, acc2, centered, N * 4);
  for (unsigned i = 0; i < M; i++) {
    float mean = 0, variance = 0;
    for (unsigned j = 0; j < N; j++) mean += input[i][j] * 0.25f;
    for (unsigned j = 0; j < N; j++) {
      float d = input[i][j] - mean;
      variance += d * d * 0.25f;
      CHECK_U32("centered row", f32_bits(centered[i][j]), f32_bits(d));
    }
    CHECK_U32("row mean", f32_bits(means[i]), f32_bits(mean));
    CHECK_U32("row variance", f32_bits(variances[i]), f32_bits(variance));
  }
}

static void softmax2(void) {
  enum { M = 4, N = 6 };
  const int exponents[M][N] = {{0, -1, -2, -3, -4, -5},
                              {3, 2, 1, 0, -1, -2},
                              {1, 1, 1, 1, 1, 1},
                              {-4, -2, 0, -1, -3, -5}};
  float input[M][N], denominator[M], result[M][N], estimate[M];
  case_begin("operator: stable base-2 softmax using Reduce / EXP2 / REC7");
  for (unsigned i = 0; i < M; i++)
    for (unsigned j = 0; j < N; j++) input[i][j] = (float)exponents[i][j];
  AME_CFG(mcfg4, AME_F32);
  AME_CFG(mcfg5, AME_F32);
  AME_CFG(mcfg6, AME_F32);
  AME_CFG(mcfg7, AME_F32);
  AME_CSR_WRITE(0x809, 0);
  ame_shape(M, N, 1);
  AME_LOAD(mlc, acc0, input, N * 4);
  AME_REDUCE(vredmax.row, lv0, acc0, 0);
  AME_EW(msub.col, acc0, lv0, acc0);
  AME_UNARY(mexp2, acc1, acc0);
  AME_REDUCE(vredadd.row, lv1, acc1, 0);
  AME_STORE(vs.col, lv1, denominator, 4);

  /* No pure-LV reciprocal instruction: explicitly stage M values through C.
   * This single-column region is the same M rows, not a reinterpretation of LV.
   */
  ame_shape(M, 1, 1);
  AME_LOAD(mlc, acc2, denominator, 4);
  AME_UNARY(mrec, acc3, acc2);
  AME_REDUCE(vredadd.row, lv2, acc3, 0);
  AME_STORE(vs.col, lv2, estimate, 4);

  /* One optional software Newton step: x <- x * (2 - d*x).
   * It is explicit EW software, not a promise of extra precision in mrec.
   */
  AME_EW(mmul, acc2, acc3, acc2);
  AME_X(mmul.x, acc2, f32_bits(-1.0f), acc2);
  AME_X(madd.x, acc2, f32_bits(2.0f), acc2);
  AME_EW(mmul, acc3, acc2, acc3);
  AME_REDUCE(vredadd.row, lv3, acc3, 0);
  ame_shape(M, N, 1);
  AME_EW(mmul.col, acc0, lv2, acc1);
  AME_STORE(msc, acc0, result, N * 4);
  for (unsigned i = 0; i < M; i++) {
    int maximum = exponents[i][0];
    float total = 0;
    for (unsigned j = 1; j < N; j++)
      if (exponents[i][j] > maximum) maximum = exponents[i][j];
    for (unsigned j = 0; j < N; j++)
      total += 1.0f / (float)(1u << (maximum - exponents[i][j]));
    CHECK_U32("integer EXP2 denominator", f32_bits(denominator[i]), f32_bits(total));
    for (unsigned j = 0; j < N; j++) {
      float probability = (1.0f / (float)(1u << (maximum - exponents[i][j]))) / total;
      /* Example-specific tolerances; not a normative ISA ULP/error bound. */
      CHECK_CLOSE("REC7 softmax2", result[i][j], probability, 0.005f);
    }
  }

  case_begin("operator: softmax2 with an explicit software Newton refinement");
  AME_EW(mmul.col, acc0, lv3, acc1);
  AME_STORE(msc, acc0, result, N * 4);
  for (unsigned i = 0; i < M; i++) {
    float total = 0;
    for (unsigned j = 0; j < N; j++) total += result[i][j];
    CHECK_CLOSE("refined probability sum", total, 1.0f, 0.0001f);
    for (unsigned j = 0; j < N; j++) {
      int maximum = exponents[i][0];
      for (unsigned k = 1; k < N; k++)
        if (exponents[i][k] > maximum) maximum = exponents[i][k];
      float probability = (1.0f / (float)(1u << (maximum - exponents[i][j]))) / denominator[i];
      CHECK_CLOSE("refined softmax2", result[i][j], probability, 0.0001f);
    }
  }
}

static void natural_softmax(void) {
  enum { M = 2, N = 5 };
  const int logits[M][N] = {{3, 2, 1, 0, -1}, {-1, 0, 1, 2, 3}};
  /* Independent constants exp(-k), k=0..4; no target libm dependency. */
  const float exp_negative[N] = {1.0f, 0.3678794411714423f, 0.1353352832366127f,
                                 0.04978706836786394f, 0.01831563888873418f};
  float input[M][N], output[M][N], denominator = 0;
  case_begin("operator: natural softmax with LV broadcast inside acc and Newton");
  for (unsigned j = 0; j < N; j++) denominator += exp_negative[j];
  for (unsigned i = 0; i < M; i++)
    for (unsigned j = 0; j < N; j++) input[i][j] = (float)logits[i][j];
  AME_CFG(mcfg4, AME_F32);
  AME_CFG(mcfg5, AME_F32);
  AME_CFG(mcfg6, AME_F32);
  AME_CSR_WRITE(0x809, 0);
  ame_shape(M, N, 1);
  AME_LOAD(mlc, acc0, input, N * 4);
  AME_REDUCE(vredmax.row, lv0, acc0, 0);
  AME_EW(msub.col, acc0, lv0, acc0);
  AME_X(mmul.x, acc0, 0x3fb8aa3bu, acc0); /* FP32 log2(e). */
  AME_UNARY(mexp2, acc0, acc0);
  AME_REDUCE(vredadd.row, lv1, acc0, 0);
  /* Expand M denominators through zero + column broadcast without a CPU
   * memory round trip. mzero initializes the temporary, not a new LV move. */
  asm volatile("mzero acc1" ::: "acc1", "memory");
  AME_EW(madd.col, acc1, lv1, acc1);
  AME_UNARY(mrec, acc2, acc1);
  AME_EW(mmul, acc1, acc2, acc1);
  AME_X(mmul.x, acc1, f32_bits(-1.0f), acc1);
  AME_X(madd.x, acc1, f32_bits(2.0f), acc1);
  AME_EW(mmul, acc2, acc1, acc2);
  AME_EW(mmul, acc0, acc2, acc0);
  AME_STORE(msc, acc0, output, N * 4);
  for (unsigned i = 0; i < M; i++) {
    float total = 0;
    for (unsigned j = 0; j < N; j++) {
      float expected = exp_negative[3 - logits[i][j]] / denominator;
      CHECK_CLOSE("natural exp softmax", output[i][j], expected, 0.0001f);
      total += output[i][j];
    }
    CHECK_CLOSE("natural softmax row sum", total, 1.0f, 0.0001f);
  }
}

static void fp16_mmacc_epilogue(void) {
  const uint16_t a = 0x3c00, b = 0x0c00, initial = 0x3c00; /* 1,2^-12,1 */
  uint16_t result;
  case_begin("operator: nominal FP16 mmacc retains FP32 before EW epilogue/store");
  AME_CSR_WRITE(0x809, 0);
  AME_CFG(mcfg0, AME_F16);
  AME_CFG(mcfg1, AME_F16);
  AME_CFG(mcfg4, AME_F16);
  ame_shape(1, 1, 1);
  AME_LOAD(mla, tr0, &a, 2);
  AME_LOAD(mlb, tr1, &b, 2);
  AME_LOAD(mlc, acc0, &initial, 2);
  asm volatile("mmacc acc0, tr1, tr0\n\tmmacc acc0, tr1, tr0" ::: "acc0", "memory");
  AME_X(msub.x, acc0, f32_bits(1.0f), acc0);
  AME_STORE(msc, acc0, &result, 2);
  /* (1 + 2^-12 + 2^-12) - 1 = 2^-11, half encoding0x1000.
   * Rounding after each old nominalFP16 mmacc would instead lose both terms.
   */
  CHECK_U32("FP16 epilogue residual", result, 0x1000);
  AME_STORE_WHOLE(msc.whole, acc0, wide_image);
  CHECK_U32("FP32 wide residual", wide_image[0], 0x3a000000);
}

void test_operators(void) {
  linear_epilogue();
  requantize();
  row_moments();
  softmax2();
  natural_softmax();
  fp16_mmacc_epilogue();
}
