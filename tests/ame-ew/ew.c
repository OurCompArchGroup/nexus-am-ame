#include "ame-ew.h"

/* Small exact scalar oracles for the normal binary examples.  Boundary and
 * special-value examples use explicit bit patterns instead of host overflow. */
enum ew_op { EW_MUL, EW_ADD, EW_SUB, EW_MAX, EW_MIN };
static const int32_t ia[6] = {2, -3, 5, -7, 11, -13};
static const int32_t ib[6] = {3, 5, -2, 2, -3, -4};
static const int32_t ir[3] = {4, -2, 3}, ic[2] = {5, -6};
static uint32_t wide_image[AME_SLOTS] __attribute__((aligned(64)));

static int32_t integer_op(enum ew_op op, int32_t a, int32_t b) {
  switch (op) {
    case EW_MUL: return a * b;
    case EW_ADD: return a + b;
    case EW_SUB: return a - b;
    case EW_MAX: return a > b ? a : b;
    default: return a < b ? a : b;
  }
}

static float floating_op(enum ew_op op, float a, float b) {
  switch (op) {
    case EW_MUL: return a * b;
    case EW_ADD: return a + b;
    case EW_SUB: return a - b;
    case EW_MAX: return a > b ? a : b;
    default: return a < b ? a : b;
  }
}

static int32_t mode_b(unsigned mode, unsigned i, unsigned j) {
  if (mode == 0) return ib[i * 3 + j];
  if (mode == 1) return ir[j];
  if (mode == 2) return ic[i];
  return -2;
}

static void check_integer_form(enum ew_op op, unsigned mode, const int32_t *out) {
  for (unsigned i = 0; i < 2; i++)
    for (unsigned j = 0; j < 3; j++)
      CHECK_U32("INT32 binary result", out[i * 3 + j],
                integer_op(op, ia[i * 3 + j], mode_b(mode, i, j)));
}

static void check_floating_form(enum ew_op op, unsigned mode, const float *out) {
  for (unsigned i = 0; i < 2; i++)
    for (unsigned j = 0; j < 3; j++)
      CHECK_U32("FP32 binary result", f32_bits(out[i * 3 + j]),
                f32_bits(floating_op(op, (float)ia[i * 3 + j],
                                   (float)mode_b(mode, i, j))));
}

#define INTEGER_FORMS(op, kind) do { \
  case_begin("INT32 " #op ": accumulator / row / column / low32 GPR"); \
  AME_EW(op, acc2, acc1, acc0); \
  AME_STORE(msc, acc2, out, 12); check_integer_form(kind, 0, out); \
  AME_EW(op.row, acc2, lv31, acc0); \
  AME_STORE(msc, acc2, out, 12); check_integer_form(kind, 1, out); \
  AME_EW(op.col, acc2, lv17, acc0); \
  AME_STORE(msc, acc2, out, 12); check_integer_form(kind, 2, out); \
  AME_X(op.x, acc2, 0x12345678fffffffeul, acc0); \
  AME_STORE(msc, acc2, out, 12); check_integer_form(kind, 3, out); \
} while (0)

#define FLOATING_FORMS(op, kind) do { \
  case_begin("FP32 " #op ": accumulator / row / column / raw low32 GPR"); \
  AME_EW(op, acc2, acc1, acc0); \
  AME_STORE(msc, acc2, out, 12); check_floating_form(kind, 0, out); \
  AME_EW(op.row, acc2, lv31, acc0); \
  AME_STORE(msc, acc2, out, 12); check_floating_form(kind, 1, out); \
  AME_EW(op.col, acc2, lv17, acc0); \
  AME_STORE(msc, acc2, out, 12); check_floating_form(kind, 2, out); \
  AME_X(op.x, acc2, 0xa5a55a5ac0000000ul, acc0); \
  AME_STORE(msc, acc2, out, 12); check_floating_form(kind, 3, out); \
} while (0)

static void binary_forms(void) {
  int32_t out[6];
  AME_CFG(mcfg4, AME_I32);
  AME_CFG(mcfg5, AME_I32);
  AME_CFG(mcfg6, AME_I32);
  AME_CSR_WRITE(0x80a, 0);
  ame_shape(2, 3, 1);
  AME_LOAD(mlc, acc0, ia, 12);
  AME_LOAD(mlc, acc1, ib, 12);
  AME_LOAD(vl.row, lv31, ir, 4);
  AME_LOAD(vl.col, lv17, ic, 4);
  INTEGER_FORMS(mmul, EW_MUL);
  INTEGER_FORMS(madd, EW_ADD);
  INTEGER_FORMS(msub, EW_SUB);
  INTEGER_FORMS(mmax, EW_MAX);
  INTEGER_FORMS(mmin, EW_MIN);

  {
    float a[6], b[6], row[3], col[2], out[6];
    for (unsigned i = 0; i < 6; i++) { a[i] = (float)ia[i]; b[i] = (float)ib[i]; }
    for (unsigned i = 0; i < 3; i++) row[i] = (float)ir[i];
    for (unsigned i = 0; i < 2; i++) col[i] = (float)ic[i];
    AME_CFG(mcfg4, AME_F32);
    AME_CFG(mcfg5, AME_F32);
    AME_CFG(mcfg6, AME_F32);
    AME_CSR_WRITE(0x809, 0);
    ame_flags_clear();
    AME_LOAD(mlc, acc0, a, 12);
    AME_LOAD(mlc, acc1, b, 12);
    AME_LOAD(vl.row, lv31, row, 4);
    AME_LOAD(vl.col, lv17, col, 4);
    FLOATING_FORMS(mmul, EW_MUL);
    FLOATING_FORMS(madd, EW_ADD);
    FLOATING_FORMS(msub, EW_SUB);
    FLOATING_FORMS(mmax, EW_MAX);
    FLOATING_FORMS(mmin, EW_MIN);
    CHECK_U32("exact floating examples have no matrix exceptions", AME_CSR_READ(0x808), 0);
  }
}

static void mixed_formats_alias_tail(void) {
  int32_t out[6];
  uintptr_t cfg;
  const int8_t narrow[2] = {100, -100};
  const int32_t addend[2] = {1000, -1000};
  const uint16_t half[2] = {0x3c00, 0xbc00}; /* +/-1 */
  const uint16_t bfloat[2] = {0x3980, 0xb980}; /* +/-2^-12 */
  uint32_t bits[2];
  case_begin("EW mixed nominal INT8/INT32 keeps wide result and destination config");
  AME_CFG(mcfg4, AME_I8);
  AME_CFG(mcfg5, AME_I32);
  AME_CFG(mcfg6, AME_I8);
  ame_shape(1, 2, 1);
  AME_LOAD(mlc, acc0, narrow, 2);
  AME_LOAD(mlc, acc1, addend, 8);
  AME_EW(madd, acc2, acc1, acc0);
  AME_GETCFG(cfg, mcfg6);
  CHECK_U32("nominal INT8 config retained", cfg, AME_I8);
  AME_CFG(mcfg6, AME_I32); /* Observe the retained wide slots without narrowing. */
  AME_STORE(msc, acc2, out, 8);
  CHECK_U32("wide positive INT8 destination", out[0], 1100);
  CHECK_U32("wide negative INT8 destination", out[1], -1100);

  case_begin("EW FP16/BF16 uses FP32, including an in-place A destination");
  AME_CFG(mcfg4, AME_F16);
  AME_CFG(mcfg5, AME_BF16);
  AME_CFG(mcfg6, AME_BF16);
  AME_LOAD(mlc, acc0, half, 4);
  AME_LOAD(mlc, acc1, bfloat, 4);
  AME_EW(madd, acc2, acc1, acc0);
  AME_GETCFG(cfg, mcfg6);
  CHECK_U32("nominal BF16 config retained", cfg, AME_BF16);
  AME_CFG(mcfg6, AME_F32);
  AME_STORE(msc, acc2, bits, 8);
  CHECK_U32("no intermediate BF16 rounding", bits[0], 0x3f800800);
  CHECK_U32("negative FP32 retained", bits[1], 0xbf800800);
  AME_EW(madd, acc0, acc1, acc0);
  AME_CFG(mcfg4, AME_F32);
  AME_STORE(msc, acc0, bits, 8);
  CHECK_U32("in-place floating A", bits[0], 0x3f800800);
  CHECK_U32("in-place floating A negative", bits[1], 0xbf800800);

  case_begin("EW reads original operands with destination=A, B, or both");
  AME_CFG(mcfg4, AME_I32);
  AME_CFG(mcfg5, AME_I32);
  AME_CFG(mcfg6, AME_I32);
  ame_shape(2, 3, 1);
  AME_LOAD(mlc, acc0, ia, 12);
  AME_LOAD(mlc, acc1, ib, 12);
  AME_EW(msub, acc0, acc1, acc0);
  AME_STORE(msc, acc0, out, 12);
  check_integer_form(EW_SUB, 0, out);
  AME_LOAD(mlc, acc0, ia, 12);
  AME_EW(msub, acc1, acc1, acc0);
  AME_STORE(msc, acc1, out, 12);
  check_integer_form(EW_SUB, 0, out);
  AME_EW(mmul, acc0, acc0, acc0);
  AME_STORE(msc, acc0, out, 12);
  for (unsigned i = 0; i < 6; i++) CHECK_U32("all operands alias", out[i], ia[i] * ia[i]);

  case_begin("EW rectangular active region ignores K and preserves every tail slot");
  for (unsigned i = 0; i < AME_SLOTS; i++) wide_image[i] = 0x92340000u + i;
  AME_LOAD_WHOLE(mlc.whole, acc2, wide_image);
  AME_LOAD(mlc, acc0, ia, 12);
  AME_LOAD(mlc, acc1, ib, 12);
  ame_shape(2, 3, 0);
  AME_EW(madd, acc2, acc1, acc0);
  AME_STORE_WHOLE(msc.whole, acc2, wide_image);
  for (unsigned i = 0; i < AME_DIM; i++)
    for (unsigned j = 0; j < AME_DIM; j++) {
      unsigned k = i * AME_DIM + j;
      uint32_t expected = i < 2 && j < 3 ? (uint32_t)(ia[i * 3 + j] + ib[i * 3 + j])
                                          : 0x92340000u + k;
      CHECK_U32("active rectangle / tail", wide_image[k], expected);
    }
}

static void integer_boundaries(void) {
  const uint32_t a[4] = {0x7fffffff, 0x80000000, 0x40000000, 0xfffffff9};
  const int32_t b[4] = {2, -1, 4, 3};
  const uint32_t add[2][4] = {{0x80000001, 0x7fffffff, 0x40000004, 0xfffffffc},
                             {0x7fffffff, 0x80000000, 0x40000004, 0xfffffffc}};
  const uint32_t sub[2][4] = {{0x80000000, 0x80000001, 0x40000001, 0xfffffffa},
                             {0x7fffffff, 0x80000001, 0x40000001, 0xfffffffa}};
  const uint32_t mul[2][4] = {{0xfffffffe, 0x80000000, 0, 0xffffffeb},
                             {0x7fffffff, 0x7fffffff, 0x7fffffff, 0xffffffeb}};
  const uint32_t absolute[2][4] = {{0x7fffffff, 0x80000000, 0x40000000, 7},
                                  {0x7fffffff, 0x7fffffff, 0x40000000, 7}};
  uint32_t out[4];
  AME_CFG(mcfg4, AME_I32);
  AME_CFG(mcfg5, AME_I32);
  AME_CFG(mcfg6, AME_I32);
  ame_shape(1, 4, 1);
  AME_LOAD(mlc, acc0, a, 16);
  AME_LOAD(mlc, acc1, b, 16);
  for (unsigned sat = 0; sat < 2; sat++) {
    case_begin(sat ? "INT32 final saturation and sticky msat" : "INT32 low32 wrap and ABS(INT_MIN)");
    AME_CSR_WRITE(0x80a, sat);
    ame_flags_clear();
    AME_CSR_WRITE(0x808, AME_DZ);
    AME_EW(madd, acc2, acc1, acc0);
    AME_STORE(msc, acc2, out, 16);
    for (unsigned i = 0; i < 4; i++) CHECK_U32("ADD boundary", out[i], add[sat][i]);
    CHECK_U32("ADD saturation flag", AME_CSR_READ(0x807), sat);
    AME_CSR_WRITE(0x807, 0);
    AME_X(msub.x, acc2, 0xfffffffful, acc0);
    AME_STORE(msc, acc2, out, 16);
    for (unsigned i = 0; i < 4; i++) CHECK_U32("SUB A-minus-B boundary", out[i], sub[sat][i]);
    CHECK_U32("SUB saturation flag", AME_CSR_READ(0x807), sat);
    AME_CSR_WRITE(0x807, 0);
    AME_EW(mmul, acc2, acc1, acc0);
    AME_STORE(msc, acc2, out, 16);
    for (unsigned i = 0; i < 4; i++) CHECK_U32("MUL boundary", out[i], mul[sat][i]);
    CHECK_U32("MUL saturation flag", AME_CSR_READ(0x807), sat);
    AME_CSR_WRITE(0x807, 0);
    AME_UNARY(mabs, acc2, acc0);
    AME_STORE(msc, acc2, out, 16);
    for (unsigned i = 0; i < 4; i++) CHECK_U32("ABS boundary", out[i], absolute[sat][i]);
    CHECK_U32("ABS saturation flag", AME_CSR_READ(0x807), sat);
    AME_X(mmax.x, acc2, 0, acc2);
    CHECK_U32("ordinary operation retains sticky msat", AME_CSR_READ(0x807), sat);
    CHECK_U32("integer EW preserves matrix FP flags", AME_CSR_READ(0x808), AME_DZ);
  }
  AME_CSR_WRITE(0x80a, 0);
  ame_flags_clear();
}

static void floating_specials(void) {
  const uint32_t a[8] = {0x7fc12345, 0x3f800000, 0x7fc12345, 0x7fa12345,
                          0x80000000, 0, 0xffc54321, 0xff812345};
  const uint32_t b[8] = {0x40000000, 0x7fc12345, 0xffc54321, 0x40400000,
                          0, 0x80000000, 0x7fc12345, 0xbf800000};
  const uint32_t max_expected[8] = {0x40000000, 0x3f800000, 0x7fc00000, 0x40400000,
                                     0, 0, 0x7fc00000, 0xbf800000};
  const uint32_t min_expected[8] = {0x40000000, 0x3f800000, 0x7fc00000, 0x40400000,
                                     0x80000000, 0x80000000, 0x7fc00000, 0xbf800000};
  uint32_t out[8];
  case_begin("FP32 MAX/MIN NaNs, signed zero, and ABS raw sign clearing");
  AME_CFG(mcfg4, AME_F32);
  AME_CFG(mcfg5, AME_F32);
  AME_CFG(mcfg6, AME_F32);
  AME_CSR_WRITE(0x809, 0);
  ame_shape(1, 8, 1);
  AME_LOAD(mlc, acc0, a, 32);
  AME_LOAD(mlc, acc1, b, 32);
  ame_flags_clear();
  AME_EW(mmax, acc2, acc1, acc0);
  AME_STORE(msc, acc2, out, 32);
  for (unsigned i = 0; i < 8; i++) CHECK_U32("RISC-V FMAX.S", out[i], max_expected[i]);
  CHECK_U32("MAX signaling NaN NV", AME_CSR_READ(0x808), AME_NV);
  ame_flags_clear();
  AME_EW(mmin, acc2, acc1, acc0);
  AME_STORE(msc, acc2, out, 32);
  for (unsigned i = 0; i < 8; i++) CHECK_U32("RISC-V FMIN.S", out[i], min_expected[i]);
  CHECK_U32("MIN signaling NaN NV", AME_CSR_READ(0x808), AME_NV);
  AME_CSR_WRITE(0x808, AME_DZ);
  AME_UNARY(mabs, acc0, acc0);
  AME_STORE(msc, acc0, out, 32);
  for (unsigned i = 0; i < 8; i++) CHECK_U32("ABS preserves NaN payload and signaling bit", out[i], a[i] & 0x7fffffff);
  CHECK_U32("ABS does not raise FP exceptions", AME_CSR_READ(0x808), AME_DZ);
}

static void rounding_and_flags(void) {
  const uint32_t a[2] = {0x3f800000, 0xbf800000};
  const uint32_t b[2] = {0x33800000, 0xb3800000}; /* +/-half an ulp at 1 */
  const uint32_t expected[5][2] = {{0x3f800000, 0xbf800000}, {0x3f800000, 0xbf800000},
                                   {0x3f800000, 0xbf800001}, {0x3f800001, 0xbf800000},
                                   {0x3f800001, 0xbf800001}};
  const uint32_t exceptional_a[4] = {0x7f7fffff, 1, 0x7f800001, 0x3f800000};
  const uint32_t exceptional_b[4] = {0x40000000, 0x3f000000, 0x3f800000, 0x33800000};
  const uint32_t product[4] = {0x7f800000, 0, 0x7fc00000, 0x33800000};
  uint32_t out[4];
  case_begin("FP32 EW all five matrix rounding modes independent of scalar fcsr");
  AME_CFG(mcfg4, AME_F32);
  AME_CFG(mcfg5, AME_F32);
  AME_CFG(mcfg6, AME_F32);
  AME_CSR_WRITE(0x003, 0x28); /* scalar RTZ with preexisting DZ */
  ame_shape(1, 2, 1);
  AME_LOAD(mlc, acc0, a, 8);
  AME_LOAD(mlc, acc1, b, 8);
  for (unsigned rm = 0; rm < 5; rm++) {
    AME_CSR_WRITE(0x809, rm);
    ame_flags_clear();
    AME_EW(madd, acc2, acc1, acc0);
    AME_STORE(msc, acc2, out, 8);
    for (unsigned i = 0; i < 2; i++) CHECK_U32("matrix rounded sum", out[i], expected[rm][i]);
    CHECK_U32("matrix NX", AME_CSR_READ(0x808), AME_NX);
    CHECK_U32("scalar frm/fflags isolated", AME_CSR_READ(0x003), 0x28);
  }

  case_begin("valid FP32 arithmetic accrues NV/OF/UF/NX and preserves sticky DZ");
  AME_CSR_WRITE(0x809, 0);
  AME_CSR_WRITE(0x808, AME_DZ);
  ame_shape(1, 4, 1);
  AME_LOAD(mlc, acc0, exceptional_a, 16);
  AME_LOAD(mlc, acc1, exceptional_b, 16);
  AME_EW(mmul, acc2, acc1, acc0);
  AME_STORE(msc, acc2, out, 16);
  for (unsigned i = 0; i < 4; i++) CHECK_U32("IEEE special product", out[i], product[i]);
  CHECK_U32("all sticky matrix exception bits", AME_CSR_READ(0x808), 31);
  AME_X(madd.x, acc2, 0, acc2);
  CHECK_U32("exact EW keeps sticky flags", AME_CSR_READ(0x808), 31);
  CHECK_U32("IEEE matrix flags do not change scalar fcsr", AME_CSR_READ(0x003), 0x28);
  AME_CSR_WRITE(0x003, 0);
  ame_flags_clear();
}

void test_ew(void) {
  binary_forms();
  mixed_formats_alias_tail();
  integer_boundaries();
  floating_specials();
  rounding_and_flags();
}

void test_sfu(void) {
  const uint32_t exp_input[8] = {0xc0400000, 0xc0000000, 0xbf800000, 0x80000000,
                                  0, 0x3f800000, 0x40000000, 0x40400000};
  const uint32_t exp_expected[8] = {0x3e000000, 0x3e800000, 0x3f000000, 0x3f800000,
                                     0x3f800000, 0x40000000, 0x40800000, 0x41000000};
  const uint32_t rec_input[8] = {0x3f800000, 0x40000000, 0x3f000000, 0x40800000,
                                  0xbf800000, 0xc0000000, 0xbf000000, 0xc0800000};
  /* REC7's first table interval gives these coarse estimates, not exact 1/x. */
  const uint32_t rec_expected[8] = {0x3f7f0000, 0x3eff0000, 0x3fff0000, 0x3e7f0000,
                                     0xbf7f0000, 0xbeff0000, 0xbfff0000, 0xbe7f0000};
  const uint32_t special[8] = {0, 0x80000000, 0x7f800000, 0xff800000,
                                0x7fc12345, 0xff812345, 1, 0x80000001};
  const uint32_t exp_special[8] = {0x3f800000, 0x3f800000, 0x7f800000, 0,
                                    0x7fc00000, 0x7fc00000, 0x3f800000, 0x3f800000};
  const uint32_t rec_special[8] = {0x7f800000, 0xff800000, 0, 0x80000000,
                                    0x7fc00000, 0x7fc00000, 0x7f800000, 0xff800000};
  const uint32_t tiny[2] = {1, 0x80000001};
  const uint32_t overflow[5][2] = {{0x7f800000, 0xff800000}, {0x7f7fffff, 0xff7fffff},
                                    {0x7f7fffff, 0xff800000}, {0x7f800000, 0xff7fffff},
                                    {0x7f800000, 0xff800000}};
  const float fractional[2] = {0.5f, -0.5f};
  uint32_t out[8];
  AME_CFG(mcfg4, AME_F32);
  AME_CFG(mcfg5, AME_F32);
  AME_CSR_WRITE(0x809, 0);
  ame_shape(1, 8, 1);
  case_begin("EXP2 exact integer powers and both zero signs");
  ame_flags_clear();
  AME_LOAD(mlc, acc0, exp_input, 32);
  AME_UNARY(mexp2, acc1, acc0);
  AME_STORE(msc, acc1, out, 32);
  for (unsigned i = 0; i < 8; i++) CHECK_U32("integer EXP2", out[i], exp_expected[i]);
  CHECK_U32("exact EXP2 flags", AME_CSR_READ(0x808), 0);

  case_begin("REC7 is a signed 7-bit estimate without implicit Newton refinement");
  ame_flags_clear();
  AME_LOAD(mlc, acc0, rec_input, 32);
  AME_UNARY(mrec, acc0, acc0);
  AME_STORE(msc, acc0, out, 32);
  for (unsigned i = 0; i < 8; i++) CHECK_U32("REC7 power-of-two interval", out[i], rec_expected[i]);
  CHECK_U32("estimate error alone does not set NX", AME_CSR_READ(0x808), 0);

  case_begin("EXP2 NaNs, infinities, subnormal inputs, and scalar state isolation");
  AME_CSR_WRITE(0x003, 0x28);
  ame_flags_clear();
  AME_LOAD(mlc, acc0, special, 32);
  AME_UNARY(mexp2, acc1, acc0);
  AME_STORE(msc, acc1, out, 32);
  for (unsigned i = 0; i < 8; i++) CHECK_U32("EXP2 special value", out[i], exp_special[i]);
  CHECK_U32("EXP2 signaling NaN and subnormal flags", AME_CSR_READ(0x808), AME_NV | AME_NX);
  CHECK_U32("EXP2 scalar fcsr unchanged", AME_CSR_READ(0x003), 0x28);

  case_begin("REC7 signed zero, infinity, NaN, tiny-input overflow, sticky flags");
  AME_CSR_WRITE(0x808, AME_UF);
  AME_UNARY(mrec, acc1, acc0);
  AME_STORE(msc, acc1, out, 32);
  for (unsigned i = 0; i < 8; i++) CHECK_U32("REC7 special value", out[i], rec_special[i]);
  CHECK_U32("REC7 accrues NV/DZ/OF/NX, keeps UF", AME_CSR_READ(0x808), 31);
  CHECK_U32("REC7 scalar fcsr unchanged", AME_CSR_READ(0x003), 0x28);

  case_begin("REC7 tiny nonzero overflow follows all five matrix rounding modes");
  ame_shape(1, 2, 1);
  AME_LOAD(mlc, acc0, tiny, 8);
  for (unsigned rm = 0; rm < 5; rm++) {
    AME_CSR_WRITE(0x809, rm);
    ame_flags_clear();
    AME_UNARY(mrec, acc1, acc0);
    AME_STORE(msc, acc1, out, 8);
    for (unsigned i = 0; i < 2; i++) CHECK_U32("rounded reciprocal overflow", out[i], overflow[rm][i]);
    CHECK_U32("nonzero tiny reciprocal OF/NX", AME_CSR_READ(0x808), AME_OF | AME_NX);
    CHECK_U32("REC7 rounding remains matrix-local", AME_CSR_READ(0x003), 0x28);
  }

  case_begin("EXP2 finite fractional example at +/-one half");
  AME_CSR_WRITE(0x809, 0);
  AME_CSR_WRITE(0x003, 0);
  ame_flags_clear();
  AME_LOAD(mlc, acc0, fractional, 8);
  AME_UNARY(mexp2, acc0, acc0);
  AME_STORE(msc, acc0, out, 8);
  CHECK_U32("fractional EXP2 NX", AME_CSR_READ(0x808), AME_NX);
  /* Example-specific checks against sqrt(2), 1/sqrt(2); no ISA error bound. */
  CHECK_CLOSE("EXP2(0.5)", f32_from_bits(out[0]), 1.41421356237f, 0.00001f);
  CHECK_CLOSE("EXP2(-0.5)", f32_from_bits(out[1]), 0.70710678118f, 0.00001f);
  AME_CSR_WRITE(0x003, 0);
  ame_flags_clear();
}
