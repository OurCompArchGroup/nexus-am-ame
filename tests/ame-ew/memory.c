#include "ame-ew.h"

/* LV is a distinct raw register file. Expand physical names at assembly time;
 * there is deliberately no dynamic-register helper or .word encoding here.
 */
#define EACH_LV(M) \
  M(0); M(1); M(2); M(3); M(4); M(5); M(6); M(7); \
  M(8); M(9); M(10); M(11); M(12); M(13); M(14); M(15); \
  M(16); M(17); M(18); M(19); M(20); M(21); M(22); M(23); \
  M(24); M(25); M(26); M(27); M(28); M(29); M(30); M(31)

enum { LV_ROWS = 3, LV_COLS = 7, HOLES = 32, SMALL_BYTES = 512 };
static uint32_t lv_seed[32][AME_DIM] __attribute__((aligned(64)));
static uint32_t lv_row[32][LV_COLS * 3] __attribute__((aligned(64)));
static uint32_t lv_col[32][LV_ROWS * 2] __attribute__((aligned(64)));
static uint32_t lv_image[AME_DIM] __attribute__((aligned(64)));
static uint32_t holes[HOLES] __attribute__((aligned(64)));
static uint32_t c_seed[AME_SLOTS] __attribute__((aligned(64)));
static uint32_t c_image[AME_SLOTS] __attribute__((aligned(64)));
static uint8_t packed_in[AME_SLOTS * 2] __attribute__((aligned(64)));
static uint8_t packed_out[AME_SLOTS * 2] __attribute__((aligned(64)));
static uint8_t packed_t[AME_SLOTS * 2] __attribute__((aligned(64)));

static void fill_bytes(uint8_t *p, unsigned count, uint8_t value) {
  for (unsigned k = 0; k < count; k++) p[k] = value;
}

/* Memory element bit patterns, not a floating-point conversion oracle. */
static void put_element(uint8_t *p, unsigned bytes, uint32_t value) {
  for (unsigned b = 0; b < bytes; b++) p[b] = (uint8_t)(value >> (8 * b));
}
static uint32_t get_element(const uint8_t *p, unsigned bytes) {
  uint32_t value = 0;
  for (unsigned b = 0; b < bytes; b++) value |= (uint32_t)p[b] << (8 * b);
  return value;
}

static void local_vectors(void) {
  case_begin("LV: all 32 whole images remain independent");
  ame_shape(LV_ROWS, LV_COLS, 1);
  AME_CSR_WRITE(0x808, AME_DZ | AME_NX);
  AME_CSR_WRITE(0x807, 1);
  for (unsigned r = 0; r < 32; r++) {
    for (unsigned k = 0; k < AME_DIM; k++)
      lv_seed[r][k] = 0xa5000000u | (r << 16) | k;
    for (unsigned k = 0; k < LV_COLS * 3; k++)
      lv_row[r][k] = 0x31000000u | (r << 16) | k;
    for (unsigned k = 0; k < LV_ROWS * 2; k++)
      lv_col[r][k] = 0x72000000u | (r << 16) | k;
  }
#define INIT_LV(r) AME_LOAD_WHOLE(vl.whole, lv##r, lv_seed[r])
  EACH_LV(INIT_LV);
#undef INIT_LV
#define CHECK_WHOLE(r) do { \
  AME_STORE_WHOLE(vs.whole, lv##r, lv_image); \
  for (unsigned k = 0; k < AME_DIM; k++) \
    CHECK_U32("independent whole LV", lv_image[k], lv_seed[r][k]); \
} while (0)
  EACH_LV(CHECK_WHOLE);
#undef CHECK_WHOLE

  case_begin("LV: all 32 row loads preserve tails and row stores preserve holes");
#define ROW_LOAD(r) AME_LOAD(vl.row, lv##r, lv_row[r], 12)
  EACH_LV(ROW_LOAD);
#undef ROW_LOAD
#define ROW_CHECK(r) do { \
  AME_STORE_WHOLE(vs.whole, lv##r, lv_image); \
  for (unsigned k = 0; k < AME_DIM; k++) \
    CHECK_U32("row load raw/tail", lv_image[k], \
              k < LV_COLS ? lv_row[r][k * 3] : lv_seed[r][k]); \
  for (unsigned k = 0; k < HOLES; k++) holes[k] = 0xdadadadau; \
  AME_STORE(vs.row, lv##r, holes, 16); \
  for (unsigned k = 0; k < HOLES; k++) \
    CHECK_U32("row store extent/holes", holes[k], \
              k % 4 == 0 && k / 4 < LV_COLS ? lv_row[r][k / 4 * 3] : 0xdadadadau); \
} while (0)
  EACH_LV(ROW_CHECK);
#undef ROW_CHECK

  case_begin("LV: all 32 column loads preserve row data and column stores preserve holes");
#define COL_LOAD(r) AME_LOAD(vl.col, lv##r, lv_col[r], 8)
  EACH_LV(COL_LOAD);
#undef COL_LOAD
#define COL_CHECK(r) do { \
  AME_STORE_WHOLE(vs.whole, lv##r, lv_image); \
  for (unsigned k = 0; k < AME_DIM; k++) { \
    uint32_t expected = k < LV_ROWS ? lv_col[r][k * 2] : \
                        k < LV_COLS ? lv_row[r][k * 3] : lv_seed[r][k]; \
    CHECK_U32("column load raw/tail", lv_image[k], expected); \
  } \
  for (unsigned k = 0; k < HOLES; k++) holes[k] = 0xdadadadau; \
  AME_STORE(vs.col, lv##r, holes, 12); \
  for (unsigned k = 0; k < HOLES; k++) \
    CHECK_U32("column store extent/holes", holes[k], \
              k % 3 == 0 && k / 3 < LV_ROWS ? lv_col[r][k / 3 * 2] : 0xdadadadau); \
} while (0)
  EACH_LV(COL_CHECK);
#undef COL_CHECK
  CHECK_U32("raw LV leaves FP flags", AME_CSR_READ(0x808), AME_DZ | AME_NX);
  CHECK_U32("raw LV leaves saturation", AME_CSR_READ(0x807), 1);

  /* All stores to the repeated address carry the same value, so this example
   * does not assume an ordering among overlapping stores of distinct values.
   */
  case_begin("LV: zero byte row stride broadcasts raw bits");
  const uint32_t value = 0xffc12345u;
  uint32_t guard[3] = {0x11111111, 0, 0x22222222};
  AME_LOAD_WHOLE(vl.whole, lv31, lv_seed[31]);
  AME_LOAD(vl.row, lv31, &value, 0);
  AME_STORE(vs.row, lv31, &guard[1], 0);
  AME_STORE_WHOLE(vs.whole, lv31, lv_image);
  for (unsigned k = 0; k < AME_DIM; k++)
    CHECK_U32("zero stride row/tail", lv_image[k], k < LV_COLS ? value : lv_seed[31][k]);
  CHECK_U32("zero stride row store", guard[1], value);
  CHECK_U32("zero stride leading guard", guard[0], 0x11111111);
  CHECK_U32("zero stride trailing guard", guard[2], 0x22222222);

  case_begin("LV: zero byte column stride broadcasts raw bits");
  guard[1] = 0;
  AME_LOAD_WHOLE(vl.whole, lv0, lv_seed[0]);
  AME_LOAD(vl.col, lv0, &value, 0);
  AME_STORE(vs.col, lv0, &guard[1], 0);
  AME_STORE_WHOLE(vs.whole, lv0, lv_image);
  for (unsigned k = 0; k < AME_DIM; k++)
    CHECK_U32("zero stride column/tail", lv_image[k], k < LV_ROWS ? value : lv_seed[0][k]);
  CHECK_U32("zero stride column store", guard[1], value);
  CHECK_U32("zero stride leading guard", guard[0], 0x11111111);
  CHECK_U32("zero stride trailing guard", guard[2], 0x22222222);
  CHECK_U32("raw NaN bits do not raise NV", AME_CSR_READ(0x808), AME_DZ | AME_NX);

  case_begin("LV: zero row/column lengths preserve raw state and memory; whole ignores dimensions");
  ame_shape(0, 0, 0);
  AME_LOAD_WHOLE(vl.whole, lv0, lv_seed[0]);
  AME_LOAD_WHOLE(vl.whole, lv31, lv_seed[31]);
  AME_LOAD(vl.row, lv0, &value, 4);
  AME_LOAD(vl.col, lv31, &value, 4);
  guard[1] = 0x33333333;
  AME_STORE(vs.row, lv0, &guard[1], 4);
  AME_STORE(vs.col, lv31, &guard[1], 4);
  CHECK_U32("zero length stores leave memory", guard[1], 0x33333333);
  CHECK_U32("zero length leading guard", guard[0], 0x11111111);
  CHECK_U32("zero length trailing guard", guard[2], 0x22222222);
  for (unsigned k = 0; k < AME_DIM; k++) lv_image[k] = 0;
  AME_STORE_WHOLE(vs.whole, lv0, lv_image);
  for (unsigned k = 0; k < AME_DIM; k++)
    CHECK_U32("zero length row retains whole image", lv_image[k], lv_seed[0][k]);
  for (unsigned k = 0; k < AME_DIM; k++) lv_image[k] = 0;
  AME_STORE_WHOLE(vs.whole, lv31, lv_image);
  for (unsigned k = 0; k < AME_DIM; k++)
    CHECK_U32("zero length column retains whole image", lv_image[k], lv_seed[31][k]);
  CHECK_U32("zero length FP flags unchanged", AME_CSR_READ(0x808), AME_DZ | AME_NX);
  CHECK_U32("zero length saturation unchanged", AME_CSR_READ(0x807), 1);

  case_begin("LV: row and column lengths of 128 transfer every raw slot");
  ame_shape(AME_DIM, AME_DIM, 0);
  AME_LOAD(vl.row, lv0, lv_seed[31], 4);
  AME_LOAD(vl.col, lv31, lv_seed[0], 4);
  for (unsigned k = 0; k < AME_DIM; k++) lv_image[k] = 0;
  AME_STORE(vs.row, lv0, lv_image, 4);
  for (unsigned k = 0; k < AME_DIM; k++)
    CHECK_U32("maximum row load/store extent", lv_image[k], lv_seed[31][k]);
  for (unsigned k = 0; k < AME_DIM; k++) lv_image[k] = 0;
  AME_STORE(vs.col, lv31, lv_image, 4);
  for (unsigned k = 0; k < AME_DIM; k++)
    CHECK_U32("maximum column load/store extent", lv_image[k], lv_seed[0][k]);
  CHECK_U32("maximum length FP flags unchanged", AME_CSR_READ(0x808), AME_DZ | AME_NX);
  CHECK_U32("maximum length saturation unchanged", AME_CSR_READ(0x807), 1);
  ame_flags_clear();
}

enum { C_I8, C_I32, C_F16, C_BF16, C_F32 };
static const uintptr_t formats[] = {AME_I8, AME_I32, AME_F16, AME_BF16, AME_F32};
static const unsigned widths[] = {1, 4, 2, 2, 4};
static const uint32_t nominal[5][8] = {
  {0x80, 0xef, 0xff, 0, 1, 17, 63, 127},
  {0xfffeffff, 0xffffffef, 0xffffffff, 0, 1, 17, 65537, 0x007fffff},
  {0xc400, 0xbe00, 0x8000, 0, 0x3800, 0x3c00, 0x4200, 0x4800},
  {0xc080, 0xbfc0, 0x8000, 0, 0x3f00, 0x3f80, 0x4040, 0x4100},
  {0xc0800000, 0xbfc00000, 0x80000000, 0, 0x3f000000, 0x3f800000, 0x40400000, 0x41000000}
};
static const uint32_t absolute_wide[5][8] = {
  {128, 17, 1, 0, 1, 17, 63, 127},
  {65537, 17, 1, 0, 1, 17, 65537, 0x007fffff},
  {0x40800000, 0x3fc00000, 0, 0, 0x3f000000, 0x3f800000, 0x40400000, 0x41000000},
  {0x40800000, 0x3fc00000, 0, 0, 0x3f000000, 0x3f800000, 0x40400000, 0x41000000},
  {0x40800000, 0x3fc00000, 0, 0, 0x3f000000, 0x3f800000, 0x40400000, 0x41000000}
};

static uint32_t absolute_nominal(unsigned kind, unsigned p) {
  if (kind == C_I8) return absolute_wide[kind][p] & 0xffu;
  if (kind == C_I32) return absolute_wide[kind][p];
  return nominal[kind][p] & (kind == C_F32 ? 0x7fffffffu : 0x7fffu);
}

static void typed_c(unsigned kind, int transpose, const char *name) {
  enum { M = 3, N = 5, ROW_PITCH = 8, COL_PITCH = 5 };
  uint8_t expected[SMALL_BYTES];
  unsigned bytes = widths[kind];
  case_begin(name);
  fill_bytes(packed_in, SMALL_BYTES, 0xa5);
  fill_bytes(packed_out, SMALL_BYTES, 0x5a);
  fill_bytes(packed_t, SMALL_BYTES, 0x5a);
  for (unsigned k = 0; k < AME_SLOTS; k++) c_seed[k] = 0x5a000000u + k;
  for (unsigned i = 0; i < M; i++)
    for (unsigned j = 0; j < N; j++) {
      unsigned element = transpose ? j * COL_PITCH + i : i * ROW_PITCH + j;
      put_element(packed_in + element * bytes, bytes, nominal[kind][(i * N + j) % 8]);
    }
  AME_CFG(mcfg4, formats[kind]);
  AME_CSR_WRITE(0x809, 0);
  AME_CSR_WRITE(0x80a, 0);
  ame_flags_clear();
  ame_shape(M, N, 1);
  AME_LOAD_WHOLE(mlc.whole, acc0, c_seed);
  if (transpose) AME_LOAD(mlct, acc0, packed_in, COL_PITCH * bytes);
  else AME_LOAD(mlc, acc0, packed_in, ROW_PITCH * bytes);
  AME_UNARY(mabs, acc0, acc0);
  AME_STORE_WHOLE(msc.whole, acc0, c_image);
  for (unsigned k = 0; k < AME_SLOTS; k++) {
    unsigned i = k / AME_DIM, j = k % AME_DIM;
    uint32_t want = i < M && j < N ? absolute_wide[kind][(i * N + j) % 8] : c_seed[k];
    CHECK_U32("C widened result and untouched wide tail", c_image[k], want);
  }
  AME_STORE(msc, acc0, packed_out, ROW_PITCH * bytes);
  AME_STORE(msct, acc0, packed_t, COL_PITCH * bytes);
  fill_bytes(expected, SMALL_BYTES, 0x5a);
  for (unsigned i = 0; i < M; i++)
    for (unsigned j = 0; j < N; j++)
      put_element(expected + (i * ROW_PITCH + j) * bytes, bytes,
                  absolute_nominal(kind, (i * N + j) % 8));
  for (unsigned k = 0; k < SMALL_BYTES; k++)
    CHECK_U32("C normal narrow/holes", packed_out[k], expected[k]);
  fill_bytes(expected, SMALL_BYTES, 0x5a);
  for (unsigned i = 0; i < M; i++)
    for (unsigned j = 0; j < N; j++)
      put_element(expected + (j * COL_PITCH + i) * bytes, bytes,
                  absolute_nominal(kind, (i * N + j) % 8));
  for (unsigned k = 0; k < SMALL_BYTES; k++)
    CHECK_U32("C transpose narrow/holes", packed_t[k], expected[k]);
  uintptr_t cfg;
  AME_GETCFG(cfg, mcfg4);
  CHECK_U32("C transfer/EW retains cfg", cfg, formats[kind]);
  CHECK_U32("exact typed memory FP flags", AME_CSR_READ(0x808), 0);
  CHECK_U32("wrap memory does not saturate", AME_CSR_READ(0x807), 0);
}

static void full_low_precision(unsigned kind, int transpose, const char *name) {
  static const uint16_t half_in[8] = {0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
  static const uint16_t half_out[8] = {0x3800, 0x3e00, 0x4100, 0x4300, 0x4480, 0x4580, 0x4680, 0x4780};
  static const uint16_t bf_in[8] = {0, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0};
  static const uint16_t bf_out[8] = {0x3f00, 0x3fc0, 0x4020, 0x4060, 0x4090, 0x40b0, 0x40d0, 0x40f0};
  static const int32_t int_in[8] = {-128, -7, -1, 0, 1, 3, 7, 127};
  static const uint32_t fp_out[8] = {0x3f000000, 0x3fc00000, 0x40200000, 0x40600000,
                                     0x40900000, 0x40b00000, 0x40d00000, 0x40f00000};
  unsigned bytes = widths[kind];
  case_begin(name);
  for (unsigned i = 0; i < AME_DIM; i++)
    for (unsigned j = 0; j < AME_DIM; j++) {
      unsigned p = (i * 5 + j) % 8;
      unsigned at = transpose ? j * AME_DIM + i : i * AME_DIM + j;
      uint32_t bits = kind == C_I8 ? (uint8_t)int_in[p] :
                      kind == C_F16 ? half_in[p] : bf_in[p];
      put_element(packed_in + at * bytes, bytes, bits);
    }
  AME_CSR_WRITE(0x809, 0);
  AME_CSR_WRITE(0x80a, 0);
  ame_flags_clear();
  AME_CFG(mcfg4, formats[kind]);
  ame_shape(AME_DIM, AME_DIM, 1);
  if (transpose) AME_LOAD(mlct, acc0, packed_in, AME_DIM * bytes);
  else AME_LOAD(mlc, acc0, packed_in, AME_DIM * bytes);
  AME_X(madd.x, acc0, kind == C_I8 ? 1u : 0x3f000000u, acc0);
  AME_STORE_WHOLE(msc.whole, acc0, c_image);
  AME_STORE(msc, acc0, packed_out, AME_DIM * bytes);
  AME_STORE(msct, acc0, packed_t, AME_DIM * bytes);
  for (unsigned i = 0; i < AME_DIM; i++)
    for (unsigned j = 0; j < AME_DIM; j++) {
      unsigned p = (i * 5 + j) % 8;
      uint32_t wide = kind == C_I8 ? (uint32_t)(int_in[p] + 1) : fp_out[p];
      uint32_t narrow = kind == C_I8 ? (uint8_t)wide :
                        kind == C_F16 ? half_out[p] : bf_out[p];
      CHECK_U32("full 128x128 wide EW", c_image[i * AME_DIM + j], wide);
      CHECK_U32("full low precision normal", get_element(packed_out + (i * AME_DIM + j) * bytes, bytes), narrow);
      CHECK_U32("full low precision transpose", get_element(packed_t + (j * AME_DIM + i) * bytes, bytes), narrow);
    }
  CHECK_U32("full exact FP flags", AME_CSR_READ(0x808), 0);
  CHECK_U32("full I8 EW remains wide", AME_CSR_READ(0x807), 0);
}

static void wide_context(void) {
  case_begin("C: nominal INT8 paired whole context preserves 16384 wide slots");
  for (unsigned k = 0; k < AME_SLOTS; k++) c_seed[k] = 0x13579bdfu ^ (k * 0x10201u);
  AME_CFG(mcfg4, AME_I8);
  AME_CSR_WRITE(0x80a, 0);
  ame_shape(1, 1, 1);
  AME_LOAD_WHOLE(mlc.whole, acc0, c_seed);
  AME_STORE_WHOLE(msc.whole, acc0, c_image);
  AME_X(madd.x, acc0, 1, acc0);
  AME_LOAD_WHOLE(mlc.whole, acc0, c_image);
  AME_STORE_WHOLE(msc.whole, acc0, c_seed);
  for (unsigned k = 0; k < AME_SLOTS; k++)
    CHECK_U32("paired raw wide context", c_seed[k], 0x13579bdfu ^ (k * 0x10201u));
}

static void int8_narrowing(void) {
  static const int32_t values[] = {-1000, -129, -128, -1, 0, 1, 127, 128, 300, 2147483647, (-2147483647 - 1)};
  enum { N = sizeof(values) / sizeof(values[0]) };
  int8_t result[N];
  for (unsigned k = 0; k < AME_SLOTS; k++) c_seed[k] = 0;
  for (unsigned k = 0; k < N; k++) c_seed[k] = (uint32_t)values[k];
  AME_CFG(mcfg4, AME_I8);
  ame_shape(1, N, 1);
  AME_LOAD_WHOLE(mlc.whole, acc0, c_seed);
  case_begin("C INT8: wide values narrow by low-byte wrap");
  AME_CSR_WRITE(0x80a, 0);
  ame_flags_clear();
  AME_STORE(msc, acc0, result, N);
  for (unsigned k = 0; k < N; k++)
    CHECK_U32("INT8 low byte", (uint8_t)result[k], (uint8_t)values[k]);
  CHECK_U32("INT8 wrap msat", AME_CSR_READ(0x807), 0);
  case_begin("C INT8: signed clipping sets sticky msat");
  AME_CSR_WRITE(0x80a, 1);
  AME_STORE(msc, acc0, result, N);
  for (unsigned k = 0; k < N; k++) {
    int32_t v = values[k];
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    CHECK_U32("INT8 signed clip", (uint8_t)result[k], (uint8_t)v);
  }
  CHECK_U32("INT8 clip msat", AME_CSR_READ(0x807), 1);
  case_begin("C INT8: exact stores retain accrued saturation until explicit clear");
  AME_X(mmax.x, acc0, (uint32_t)-128, acc0);
  AME_X(mmin.x, acc0, 127, acc0);
  AME_STORE(msc, acc0, result, N);
  CHECK_U32("INT8 exact store sticky msat", AME_CSR_READ(0x807), 1);
  ame_flags_clear();
  AME_STORE(msc, acc0, result, N);
  CHECK_U32("INT8 exact store after clear", AME_CSR_READ(0x807), 0);
  CHECK_U32("INT8 stores do not raise FP flags", AME_CSR_READ(0x808), 0);
  AME_CSR_WRITE(0x80a, 0);
}

void test_memory(void) {
  local_vectors();
  typed_c(C_I8, 0, "C INT8 normal load: signed widen / ABS / both store layouts");
  typed_c(C_I8, 1, "C INT8 transpose load: signed widen / ABS / both store layouts");
  typed_c(C_I32, 0, "C INT32 normal load feeding new ABS / preserved tail");
  typed_c(C_I32, 1, "C INT32 transpose load feeding new ABS / preserved tail");
  typed_c(C_F16, 0, "C FP16 normal load: FP32 slots / ABS / narrow");
  typed_c(C_F16, 1, "C FP16 transpose load: FP32 slots / ABS / narrow");
  typed_c(C_BF16, 0, "C BF16 normal load: FP32 slots / ABS / narrow");
  typed_c(C_BF16, 1, "C BF16 transpose load: FP32 slots / ABS / narrow");
  typed_c(C_F32, 0, "C FP32 normal load feeding new ABS / preserved tail");
  typed_c(C_F32, 1, "C FP32 transpose load feeding new ABS / preserved tail");
  full_low_precision(C_I8, 0, "C INT8: full 128x128 normal load / wide ADD / both stores");
  full_low_precision(C_I8, 1, "C INT8: full 128x128 transpose load / wide ADD / both stores");
  full_low_precision(C_F16, 0, "C FP16: full 128x128 normal load / FP32 ADD / both stores");
  full_low_precision(C_F16, 1, "C FP16: full 128x128 transpose load / FP32 ADD / both stores");
  full_low_precision(C_BF16, 0, "C BF16: full 128x128 normal load / FP32 ADD / both stores");
  full_low_precision(C_BF16, 1, "C BF16: full 128x128 transpose load / FP32 ADD / both stores");
  wide_context();
  int8_narrowing();
}
