#include "ame-ew.h"

/* Exact small integers keep these FP32 sums independent of reduction-tree
 * choices.  Each LV image also verifies every element outside the output. */
static const int32_t input[2][3] = {{2, -3, 5}, {-7, 11, -13}};
static uint32_t seed[AME_DIM] __attribute__((aligned(64)));
static uint32_t result[AME_DIM] __attribute__((aligned(64)));

static void init_seed(unsigned floating) {
  for (unsigned i = 0; i < AME_DIM; i++) seed[i] = 0x6e120000u + i;
  seed[0] = floating ? f32_bits(17.0f) : 17;
  seed[1] = floating ? f32_bits(-19.0f) : (uint32_t)-19;
  seed[2] = floating ? f32_bits(23.0f) : 23;
}

static void check_form(unsigned floating, unsigned maximum, unsigned column,
                       unsigned accumulate) {
  unsigned outputs = column ? 3 : 2, terms = column ? 2 : 3;
  for (unsigned k = 0; k < AME_DIM; k++) {
    uint32_t expected = seed[k];
    if (k < outputs) {
      int32_t value = column ? input[0][k] : input[k][0];
      for (unsigned t = 1; t < terms; t++) {
        int32_t x = column ? input[t][k] : input[k][t];
        if (maximum) { if (x > value) value = x; }
        else value += x;
      }
      if (accumulate) {
        int32_t old = k == 0 ? 17 : k == 1 ? -19 : 23;
        if (maximum) { if (old > value) value = old; }
        else value += old;
      }
      expected = floating ? f32_bits((float)value) : (uint32_t)value;
    }
    CHECK_U32("reduced value / preserved LV tail", result[k], expected);
  }
}

#define REDUCE_FORM(op, maximum, column, accumulate, floating) do { \
  case_begin((floating) ? "FP32 " #op " acc=" AME_STR(accumulate) \
                        : "INT32 " #op " acc=" AME_STR(accumulate)); \
  AME_LOAD_WHOLE(vl.whole, lv31, seed); \
  AME_REDUCE(op, lv31, acc0, accumulate); \
  AME_STORE_WHOLE(vs.whole, lv31, result); \
  check_form(floating, maximum, column, accumulate); \
} while (0)

#define ALL_REDUCTIONS(floating) do { \
  REDUCE_FORM(vredadd.row, 0, 0, 0, floating); \
  REDUCE_FORM(vredadd.row, 0, 0, 1, floating); \
  REDUCE_FORM(vredadd.col, 0, 1, 0, floating); \
  REDUCE_FORM(vredadd.col, 0, 1, 1, floating); \
  REDUCE_FORM(vredmax.row, 1, 0, 0, floating); \
  REDUCE_FORM(vredmax.row, 1, 0, 1, floating); \
  REDUCE_FORM(vredmax.col, 1, 1, 0, floating); \
  REDUCE_FORM(vredmax.col, 1, 1, 1, floating); \
} while (0)

static void basic_forms(void) {
  float fp[2][3];
  AME_CSR_WRITE(0x80a, 0);
  AME_CSR_WRITE(0x809, 0);
  ame_flags_clear();
  ame_shape(2, 3, 0); /* Nonempty reductions do not depend on K. */
  init_seed(0);
  AME_CFG(mcfg4, AME_I32);
  AME_LOAD(mlc, acc0, input, 12);
  ALL_REDUCTIONS(0);
  for (unsigned i = 0; i < 2; i++)
    for (unsigned j = 0; j < 3; j++) fp[i][j] = (float)input[i][j];
  init_seed(1);
  AME_CFG(mcfg4, AME_F32);
  AME_LOAD(mlc, acc0, fp, 12);
  ALL_REDUCTIONS(1);
  CHECK_U32("exact finite reductions have no FP exceptions", AME_CSR_READ(0x808), 0);
}

/* The same logical 2x3 matrix, split once along each reduction axis.
 * acc=0 initializes a streaming reduction; each later tile uses acc=1. */
static void streamed_tiles(void) {
  static const int32_t first_row[2][2] = {{2, -3}, {-7, 11}};
  static const int32_t second_row[2] = {5, -13};
  static const int32_t first_col[3] = {2, -3, 5};
  static const int32_t second_col[3] = {-7, 11, -13};
  static const int32_t row_sum[2] = {4, -9}, row_max[2] = {5, 11};
  static const int32_t col_sum[3] = {-5, 8, -8}, col_max[3] = {2, 11, 5};
  uint32_t row_first[2][2], row_second[2], col_first[3], col_second[3];
  uint32_t sums[3], maxima[3];
  AME_CSR_WRITE(0x80a, 0);
  AME_CSR_WRITE(0x809, 0);
  for (unsigned floating = 0; floating < 2; floating++) {
    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++)
        row_first[i][j] = floating ? f32_bits((float)first_row[i][j])
                                   : (uint32_t)first_row[i][j];
      row_second[i] = floating ? f32_bits((float)second_row[i])
                               : (uint32_t)second_row[i];
    }
    for (unsigned j = 0; j < 3; j++) {
      col_first[j] = floating ? f32_bits((float)first_col[j])
                              : (uint32_t)first_col[j];
      col_second[j] = floating ? f32_bits((float)second_col[j])
                               : (uint32_t)second_col[j];
    }
    case_begin(floating ? "FP32 streaming Reduce across two real tiles, row and column"
                        : "INT32 streaming Reduce across two real tiles, row and column");
    AME_CFG(mcfg4, floating ? AME_F32 : AME_I32);
    ame_flags_clear();

    /* Column partition: first two columns, then the final column. */
    ame_shape(2, 2, 1);
    AME_LOAD(mlc, acc0, row_first, 8);
    AME_REDUCE(vredadd.row, lv0, acc0, 0);
    AME_REDUCE(vredmax.row, lv1, acc0, 0);
    ame_shape(2, 1, 1);
    AME_LOAD(mlc, acc0, row_second, 4);
    AME_REDUCE(vredadd.row, lv0, acc0, 1);
    AME_REDUCE(vredmax.row, lv1, acc0, 1);
    AME_STORE(vs.col, lv0, sums, 4);
    AME_STORE(vs.col, lv1, maxima, 4);
    for (unsigned i = 0; i < 2; i++) {
      CHECK_U32("two-tile row sum", sums[i], floating ? f32_bits((float)row_sum[i])
                                                     : (uint32_t)row_sum[i]);
      CHECK_U32("two-tile row maximum", maxima[i], floating ? f32_bits((float)row_max[i])
                                                           : (uint32_t)row_max[i]);
    }

    /* Row partition: first row, then the second row, preserving each LV column. */
    ame_shape(1, 3, 1);
    AME_LOAD(mlc, acc0, col_first, 12);
    AME_REDUCE(vredadd.col, lv0, acc0, 0);
    AME_REDUCE(vredmax.col, lv1, acc0, 0);
    AME_LOAD(mlc, acc0, col_second, 12);
    AME_REDUCE(vredadd.col, lv0, acc0, 1);
    AME_REDUCE(vredmax.col, lv1, acc0, 1);
    AME_STORE(vs.row, lv0, sums, 4);
    AME_STORE(vs.row, lv1, maxima, 4);
    for (unsigned j = 0; j < 3; j++) {
      CHECK_U32("two-tile column sum", sums[j], floating ? f32_bits((float)col_sum[j])
                                                        : (uint32_t)col_sum[j]);
      CHECK_U32("two-tile column maximum", maxima[j], floating ? f32_bits((float)col_max[j])
                                                              : (uint32_t)col_max[j]);
    }
    CHECK_U32("exact streaming sums have no matrix exceptions", AME_CSR_READ(0x808), 0);
    CHECK_U32("small streaming sums do not saturate", AME_CSR_READ(0x807), 0);
  }
}

static void wide_nominal_source(void) {
  const int8_t bytes[2] = {100, 100};
  const uint16_t halves[2][2] = {{0x3c00, 0x3c00}, {0xbc00, 0xbc00}};
  uintptr_t cfg;
  case_begin("Reduce nominal INT8 source retains INT32 EW results");
  AME_CFG(mcfg4, AME_I8);
  ame_shape(1, 2, 1);
  AME_LOAD(mlc, acc0, bytes, 2);
  AME_X(madd.x, acc0, 1000, acc0);
  AME_REDUCE(vredadd.row, lv0, acc0, 0);
  AME_STORE(vs.col, lv0, result, 4);
  CHECK_U32("INT8 source wide sum", result[0], 2200);
  AME_GETCFG(cfg, mcfg4);
  CHECK_U32("Reduce keeps nominal INT8 config", cfg, AME_I8);

  case_begin("Reduce nominal FP16 source emits raw FP32 without narrowing");
  AME_CFG(mcfg4, AME_F16);
  ame_shape(2, 2, 1);
  AME_LOAD(mlc, acc0, halves, 4);
  AME_X(madd.x, acc0, 0x39800000ul, acc0); /* Add 2^-12 in FP32. */
  AME_REDUCE(vredadd.row, lv0, acc0, 0);
  AME_STORE(vs.col, lv0, result, 4);
  CHECK_U32("positive wide FP16-source sum", result[0], 0x40000800);
  CHECK_U32("negative wide FP16-source sum", result[1], 0xbffff000);
  AME_GETCFG(cfg, mcfg4);
  CHECK_U32("Reduce keeps nominal FP16 config", cfg, AME_F16);
}

static void integer_final_rounding(void) {
  const uint32_t source[2][3] = {{0x7fffffff, 0x7fffffff, 0x80000001},
                                  {0x80000000, 0x80000000, 0x7fffffff}};
  const uint32_t cancellation[2][3] = {{0x7fffffff, 0x7fffffff, 0},
                                        {0x80000000, 0x80000000, 1}};
  const uint32_t cancellation_col[3][2] = {{0x7fffffff, 0x80000000},
                                            {0x7fffffff, 0x80000000}, {0, 1}};
  const uint32_t endpoint[2] = {0x7fffffff, 0x80000000};
  AME_CFG(mcfg4, AME_I32);
  for (unsigned sat = 0; sat < 2; sat++) {
    case_begin(sat ? "Reduce final signed saturation, without intermediate clipping"
                   : "Reduce wraps the complete mathematical sum once");
    AME_CSR_WRITE(0x80a, sat);
    ame_flags_clear();
    AME_CSR_WRITE(0x808, AME_DZ);
    ame_shape(2, 3, 1);
    AME_LOAD(mlc, acc0, source, 12);
    AME_REDUCE(vredadd.row, lv0, acc0, 0);
    AME_STORE(vs.col, lv0, result, 4);
    CHECK_U32("overflow then cancellation", result[0], 0x7fffffff);
    CHECK_U32("negative final overflow", result[1], sat ? 0x80000000 : 0x7fffffff);
    CHECK_U32("final clipping msat", AME_CSR_READ(0x807), sat);
    CHECK_U32("integer Reduce preserves matrix FP flags", AME_CSR_READ(0x808), AME_DZ);

    case_begin("Reduce includes old LV value before deciding overflow");
    AME_CSR_WRITE(0x807, 0);
    seed[0] = 0x80000001; /* -INT32_MAX */
    seed[1] = 0x7fffffff;
    AME_LOAD(vl.col, lv0, seed, 4);
    AME_LOAD(mlc, acc0, cancellation, 12);
    AME_REDUCE(vredadd.row, lv0, acc0, 1);
    AME_STORE(vs.col, lv0, result, 4);
    CHECK_U32("old negative cancels positive partial overflow", result[0], 0x7fffffff);
    CHECK_U32("old positive cancels negative partial overflow", result[1], 0x80000000);
    CHECK_U32("no saturation after complete cancellation", AME_CSR_READ(0x807), 0);
    ame_shape(3, 2, 1);
    AME_LOAD(vl.row, lv0, seed, 4);
    AME_LOAD(mlc, acc0, cancellation_col, 8);
    AME_REDUCE(vredadd.col, lv0, acc0, 1);
    AME_STORE(vs.row, lv0, result, 4);
    CHECK_U32("column cancellation positive", result[0], 0x7fffffff);
    CHECK_U32("column cancellation negative", result[1], 0x80000000);
    CHECK_U32("column final-only saturation", AME_CSR_READ(0x807), 0);

    case_begin("Reduce old destination can cause final overflow; msat stays sticky");
    ame_shape(2, 1, 1);
    seed[0] = 1;
    seed[1] = 0xffffffff;
    AME_LOAD(vl.col, lv0, seed, 4);
    AME_LOAD(mlc, acc0, endpoint, 4);
    AME_REDUCE(vredadd.row, lv0, acc0, 1);
    AME_STORE(vs.col, lv0, result, 4);
    CHECK_U32("old value causes positive overflow", result[0], sat ? 0x7fffffff : 0x80000000);
    CHECK_U32("old value causes negative overflow", result[1], sat ? 0x80000000 : 0x7fffffff);
    CHECK_U32("old-destination clipping msat", AME_CSR_READ(0x807), sat);
    AME_REDUCE(vredmax.row, lv0, acc0, 0);
    CHECK_U32("MAX keeps sticky msat", AME_CSR_READ(0x807), sat);
  }
  AME_CSR_WRITE(0x80a, 0);
  ame_flags_clear();
}

static void check_empty(unsigned outputs, unsigned accumulate) {
  for (unsigned i = 0; i < AME_DIM; i++)
    CHECK_U32("empty input / zero output / LV tail", result[i],
              i < outputs && !accumulate ? 0 : seed[i]);
  CHECK_U32("empty Reduce keeps FP flags", AME_CSR_READ(0x808), AME_DZ);
  CHECK_U32("empty Reduce keeps msat", AME_CSR_READ(0x807), 1);
  CHECK_U32("empty Reduce leaves scalar fcsr", AME_CSR_READ(0x003), 0x28);
}

#define EMPTY_REDUCTION(op, outputs, accumulate) do { \
  AME_LOAD_WHOLE(vl.whole, lv31, seed); \
  AME_REDUCE(op, lv31, acc0, accumulate); \
  AME_STORE_WHOLE(vs.whole, lv31, result); \
  check_empty(outputs, accumulate); \
} while (0)

static void empty_reductions(void) {
  for (unsigned i = 0; i < AME_DIM; i++) seed[i] = 0x7f800001u + i;
  seed[1] = 0x80000000; /* No arithmetic, even with signaling NaNs / signed zero. */
  for (unsigned floating = 0; floating < 2; floating++) {
    AME_CFG(mcfg4, floating ? AME_F32 : AME_I32);
    AME_CSR_WRITE(0x003, 0x28);
    AME_CSR_WRITE(0x808, AME_DZ);
    AME_CSR_WRITE(0x807, 1);
    case_begin(floating ? "FP32 empty columns / zero column outputs, all reductions"
                        : "INT32 empty columns / zero column outputs, all reductions");
    ame_shape(3, 0, 0);
    EMPTY_REDUCTION(vredadd.row, 3, 0);
    EMPTY_REDUCTION(vredadd.row, 3, 1);
    EMPTY_REDUCTION(vredmax.row, 3, 0);
    EMPTY_REDUCTION(vredmax.row, 3, 1);
    EMPTY_REDUCTION(vredadd.col, 0, 0);
    EMPTY_REDUCTION(vredadd.col, 0, 1);
    EMPTY_REDUCTION(vredmax.col, 0, 0);
    EMPTY_REDUCTION(vredmax.col, 0, 1);
    case_begin(floating ? "FP32 empty rows / zero row outputs, all reductions"
                        : "INT32 empty rows / zero row outputs, all reductions");
    ame_shape(0, 3, 0);
    EMPTY_REDUCTION(vredadd.col, 3, 0);
    EMPTY_REDUCTION(vredadd.col, 3, 1);
    EMPTY_REDUCTION(vredmax.col, 3, 0);
    EMPTY_REDUCTION(vredmax.col, 3, 1);
    EMPTY_REDUCTION(vredadd.row, 0, 0);
    EMPTY_REDUCTION(vredadd.row, 0, 1);
    EMPTY_REDUCTION(vredmax.row, 0, 0);
    EMPTY_REDUCTION(vredmax.row, 0, 1);
  }
  AME_CSR_WRITE(0x003, 0);
  ame_flags_clear();
}

static void floating_specials(void) {
  const uint32_t values[4][2] = {{0x7fc12345, 0x40000000},
                                  {0x7f812345, 0x40400000},
                                  {0x80000000, 0}, {0xffc54321, 0x7fc12345}};
  const uint32_t maximum[4] = {0x40000000, 0x40400000, 0, 0x7fc00000};
  const uint32_t combined[4] = {0x40800000, 0x40400000, 0, 0x40a00000};
  const uint32_t snan = 0xff812345;
  const uint32_t infinities[2] = {0x7f800000, 0xff800000};
  const uint32_t half_ulp = 0x33800000;
  const uint32_t rounded[5] = {0x3f800000, 0x3f800000, 0x3f800000,
                                0x3f800001, 0x3f800001};
  AME_CFG(mcfg4, AME_F32);
  AME_CSR_WRITE(0x809, 0);
  case_begin("FP32 Reduce MAX number-preferred NaNs and signed zero");
  ame_shape(4, 2, 1);
  AME_LOAD(mlc, acc0, values, 8);
  ame_flags_clear();
  AME_REDUCE(vredmax.row, lv31, acc0, 0);
  AME_STORE(vs.col, lv31, result, 4);
  for (unsigned i = 0; i < 4; i++) CHECK_U32("row MAX special", result[i], maximum[i]);
  CHECK_U32("MAX signaling NaN NV", AME_CSR_READ(0x808), AME_NV);
  seed[0] = 0x40800000; /* 4 */
  seed[1] = 0x7f812345;
  seed[2] = 0x80000000;
  seed[3] = 0x40a00000; /* 5 */
  AME_LOAD(vl.col, lv31, seed, 4);
  ame_flags_clear();
  AME_REDUCE(vredmax.row, lv31, acc0, 1);
  AME_STORE(vs.col, lv31, result, 4);
  for (unsigned i = 0; i < 4; i++) CHECK_U32("old destination MAX special", result[i], combined[i]);
  CHECK_U32("old signaling NaN NV", AME_CSR_READ(0x808), AME_NV);

  case_begin("singleton ADD and MAX quiet signaling NaNs and accrue NV");
  ame_shape(1, 1, 1);
  AME_LOAD(mlc, acc0, &snan, 4);
  AME_CSR_WRITE(0x808, AME_DZ);
  AME_REDUCE(vredadd.row, lv0, acc0, 0);
  AME_STORE(vs.col, lv0, result, 4);
  CHECK_U32("singleton ADD canonical NaN", result[0], 0x7fc00000);
  CHECK_U32("singleton ADD sticky NV", AME_CSR_READ(0x808), AME_NV | AME_DZ);
  AME_CSR_WRITE(0x808, AME_DZ);
  AME_REDUCE(vredmax.col, lv0, acc0, 0);
  AME_STORE(vs.row, lv0, result, 4);
  CHECK_U32("singleton MAX canonical NaN", result[0], 0x7fc00000);
  CHECK_U32("singleton MAX sticky NV", AME_CSR_READ(0x808), AME_NV | AME_DZ);

  case_begin("FP32 Reduce ADD opposite infinities is a valid NV arithmetic case");
  ame_shape(1, 2, 1);
  AME_LOAD(mlc, acc0, infinities, 8);
  ame_flags_clear();
  AME_REDUCE(vredadd.row, lv0, acc0, 0);
  AME_STORE(vs.col, lv0, result, 4);
  CHECK_U32("infinity cancellation canonical NaN", result[0], 0x7fc00000);
  CHECK_U32("infinity cancellation NV", AME_CSR_READ(0x808), AME_NV);

  case_begin("FP32 singleton acc=1 obeys matrix rounding independently of scalar fcsr");
  AME_CSR_WRITE(0x003, 0x28);
  ame_shape(1, 1, 1);
  AME_LOAD(mlc, acc0, &half_ulp, 4);
  seed[0] = 0x3f800000;
  for (unsigned rm = 0; rm < 5; rm++) {
    AME_CSR_WRITE(0x809, rm);
    AME_CSR_WRITE(0x808, AME_NV);
    AME_LOAD(vl.row, lv0, seed, 4);
    AME_REDUCE(vredadd.col, lv0, acc0, 1);
    AME_STORE(vs.row, lv0, result, 4);
    CHECK_U32("singleton + old FP32 rounded result", result[0], rounded[rm]);
    CHECK_U32("Reduce NX plus sticky NV", AME_CSR_READ(0x808), AME_NV | AME_NX);
    CHECK_U32("Reduce scalar fcsr unchanged", AME_CSR_READ(0x003), 0x28);
  }
  AME_CSR_WRITE(0x809, 0);
  AME_CSR_WRITE(0x003, 0);
  ame_flags_clear();
}

void test_reduce(void) {
  basic_forms();
  streamed_tiles();
  wide_nominal_source();
  integer_final_rounding();
  empty_reductions();
  floating_specials();
}
