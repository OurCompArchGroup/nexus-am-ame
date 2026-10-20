#include "ame-ew.h"

/* These are finite, hand-derived binary32/INT32 conversion tables and standard
 * RISC-V canonical-NaN results. No NEMU helper or host floating-point library
 * supplies the oracle. RM rows are RNE, RTZ, RDN, RUP, RMM in that order.
 */
static uint32_t cvt_input[AME_SLOTS] __attribute__((aligned(64)));
static uint32_t cvt_output[AME_SLOTS] __attribute__((aligned(64)));
#define CFG_PAYLOAD 0x5a123456789abc00ul
#define SCALAR_SENTINEL ((3u << 5) | AME_DZ | AME_NX)

static void check_cfg(uintptr_t got, uintptr_t expected) {
  CHECK_U32("cfg low32", got, expected);
  CHECK_U32("cfg high32 payload", got >> 32, expected >> 32);
}
static void input_pattern(const uint32_t *pattern, unsigned count) {
  for (unsigned k = 0; k < AME_SLOTS; k++) cvt_input[k] = pattern[k % count];
}
static void expect_pattern(const uint32_t *pattern, unsigned count) {
  for (unsigned k = 0; k < AME_SLOTS; k++)
    CHECK_U32("whole conversion slot", cvt_output[k], pattern[k % count]);
}
static void scalar_is_unchanged(void) {
  CHECK_U32("scalar frm/fflags independent", AME_CSR_READ(0x003), SCALAR_SENTINEL);
}

static void raw_configuration(void) {
  static const uint32_t raw[] = {0, 0x80000000, 0x3f800001, 0x7f800000,
                                  0x7fc01234, 0x7f801234, 1, 0xffffffff};
  uintptr_t cfg;
  case_begin("msetcfg cvt=0: full wide bits survive a category reinterpretation");
  input_pattern(raw, 8);
  ame_shape(1, 1, 1);
  AME_CSR_WRITE(0x809, 4);
  AME_CSR_WRITE(0x003, SCALAR_SENTINEL);
  AME_CSR_WRITE(0x808, AME_DZ | AME_UF);
  AME_CFG(mcfg4, CFG_PAYLOAD | AME_I32);
  AME_LOAD_WHOLE(mlc.whole, acc0, cvt_input);
  AME_CFG(mcfg4, CFG_PAYLOAD | AME_F16);
  AME_GETCFG(cfg, mcfg4);
  check_cfg(cfg, CFG_PAYLOAD | AME_F16);
  AME_STORE_WHOLE(msc.whole, acc0, cvt_output);
  expect_pattern(raw, 8);
  CHECK_U32("configuration-only flags unchanged", AME_CSR_READ(0x808), AME_DZ | AME_UF);
  scalar_is_unchanged();

  case_begin("msetcfg.cvt: INT32 / INT8 changes retain every INT32 slot");
  AME_CFG(mcfg5, CFG_PAYLOAD | AME_I32);
  AME_LOAD_WHOLE(mlc.whole, acc1, cvt_input);
  AME_CSR_WRITE(0x80a, 1);
  AME_CSR_WRITE(0x807, 1);
  AME_CVT(mcfg5, acc1, CFG_PAYLOAD | AME_I8);
  AME_GETCFG(cfg, mcfg5);
  check_cfg(cfg, CFG_PAYLOAD | AME_I8);
  AME_STORE_WHOLE(msc.whole, acc1, cvt_output);
  expect_pattern(raw, 8);
  AME_CVT(mcfg5, acc1, (CFG_PAYLOAD ^ 0x1200000000ul) | AME_I32);
  AME_GETCFG(cfg, mcfg5);
  check_cfg(cfg, (CFG_PAYLOAD ^ 0x1200000000ul) | AME_I32);
  AME_STORE_WHOLE(msc.whole, acc1, cvt_output);
  expect_pattern(raw, 8);
  CHECK_U32("same integer category keeps msat", AME_CSR_READ(0x807), 1);
  CHECK_U32("same integer category keeps mfflags", AME_CSR_READ(0x808), AME_DZ | AME_UF);
  scalar_is_unchanged();

  case_begin("msetcfg.cvt: FP32 / FP16 / BF16 nominal changes retain raw FP32");
  AME_CFG(mcfg6, CFG_PAYLOAD | AME_F32);
  AME_LOAD_WHOLE(mlc.whole, acc2, cvt_input);
  AME_CVT(mcfg6, acc2, CFG_PAYLOAD | AME_F16);
  AME_GETCFG(cfg, mcfg6);
  check_cfg(cfg, CFG_PAYLOAD | AME_F16);
  AME_STORE_WHOLE(msc.whole, acc2, cvt_output);
  expect_pattern(raw, 8);
  AME_CVT(mcfg6, acc2, CFG_PAYLOAD | AME_BF16);
  AME_GETCFG(cfg, mcfg6);
  check_cfg(cfg, CFG_PAYLOAD | AME_BF16);
  AME_STORE_WHOLE(msc.whole, acc2, cvt_output);
  expect_pattern(raw, 8);
  AME_CVT(mcfg6, acc2, CFG_PAYLOAD | AME_F32);
  AME_GETCFG(cfg, mcfg6);
  check_cfg(cfg, CFG_PAYLOAD | AME_F32);
  AME_STORE_WHOLE(msc.whole, acc2, cvt_output);
  expect_pattern(raw, 8);
  CHECK_U32("same FP category never rounds sNaN or subnormal", AME_CSR_READ(0x808), AME_DZ | AME_UF);
  CHECK_U32("same FP category keeps msat", AME_CSR_READ(0x807), 1);
  scalar_is_unchanged();
  AME_CSR_WRITE(0x80a, 0);
}

static void whole_cross_category(void) {
  static const uint32_t integer[] = {0xfffffc00, 0xfffffffd, 0xffffffff, 0, 1, 3, 1024, 16777216};
  static const uint32_t single[] = {0xc4800000, 0xc0400000, 0xbf800000, 0, 0x3f800000,
                                     0x40400000, 0x44800000, 0x4b800000};
  uintptr_t cfg;
  case_begin("msetcfg.cvt: all 16384 slots convert with M=N=K=0");
  input_pattern(integer, 8);
  AME_CFG(mcfg7, CFG_PAYLOAD | AME_I32);
  AME_LOAD_WHOLE(mlc.whole, acc3, cvt_input);
  ame_shape(0, 0, 0); /* Legal: conversion is a whole-register operation. */
  AME_CSR_WRITE(0x809, 0);
  AME_CSR_WRITE(0x003, SCALAR_SENTINEL);
  AME_CSR_WRITE(0x80a, 1);
  ame_flags_clear();
  AME_CVT(mcfg7, acc3, CFG_PAYLOAD | AME_F32);
  AME_GETCFG(cfg, mcfg7);
  check_cfg(cfg, CFG_PAYLOAD | AME_F32);
  AME_STORE_WHOLE(msc.whole, acc3, cvt_output);
  expect_pattern(single, 8);
  AME_CVT(mcfg7, acc3, CFG_PAYLOAD | AME_I8);
  AME_GETCFG(cfg, mcfg7);
  check_cfg(cfg, CFG_PAYLOAD | AME_I8);
  AME_STORE_WHOLE(msc.whole, acc3, cvt_output);
  expect_pattern(integer, 8);
  CHECK_U32("exact cross conversion flags", AME_CSR_READ(0x808), 0);
  CHECK_U32("INT8 nominal conversion does not clip INT32 slots", AME_CSR_READ(0x807), 0);
  scalar_is_unchanged();
  AME_CSR_WRITE(0x80a, 0);
}

static void signed_float_to_integer(void) {
  static const uint32_t source[] = {0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000,
                                     0x40200000, 0xc0200000, 0x3fa00000, 0xbfa00000,
                                     0x3fe00000, 0xbfe00000};
  static const int32_t expected[5][10] = {
    {0, 0, 2, -2, 2, -2, 1, -1, 2, -2},
    {0, 0, 1, -1, 2, -2, 1, -1, 1, -1},
    {0, -1, 1, -2, 2, -3, 1, -2, 1, -2},
    {1, 0, 2, -1, 3, -2, 2, -1, 2, -1},
    {1, -1, 2, -2, 3, -3, 1, -1, 2, -2}
  };
  static const char *names[] = {"signed FCVT.W.S: RNE finite ties", "signed FCVT.W.S: RTZ finite ties",
    "signed FCVT.W.S: RDN finite ties", "signed FCVT.W.S: RUP finite ties", "signed FCVT.W.S: RMM finite ties"};
  input_pattern(source, 10);
  for (unsigned rm = 0; rm < 5; rm++) {
    case_begin(names[rm]);
    ame_shape(1, 1, 1);
    AME_CFG(mcfg4, AME_F16); /* Contents remain FP32 despite nominal FP16. */
    AME_LOAD_WHOLE(mlc.whole, acc0, cvt_input);
    AME_CSR_WRITE(0x809, rm);
    AME_CSR_WRITE(0x003, SCALAR_SENTINEL);
    ame_flags_clear();
    AME_CVT(mcfg4, acc0, AME_I8);
    AME_STORE_WHOLE(msc.whole, acc0, cvt_output);
    for (unsigned k = 0; k < AME_SLOTS; k++)
      CHECK_U32("signed FCVT.W.S result", cvt_output[k], expected[rm][k % 10]);
    CHECK_U32("signed FCVT.W.S inexact", AME_CSR_READ(0x808), AME_NX);
    CHECK_U32("FP to INT is not integer saturation", AME_CSR_READ(0x807), 0);
    CHECK_U32("matrix rounding retained", AME_CSR_READ(0x809), rm);
    scalar_is_unchanged();
  }
}

static void signed_integer_to_float(void) {
  static const uint32_t source[] = {16777217, 0xfeffffff, 16777219, 0xfefffffd,
                                     0x7fffffff, 0x80000000, 0, 1, 0xffffffff};
  static const uint32_t expected[5][9] = {
    {0x4b800000, 0xcb800000, 0x4b800002, 0xcb800002, 0x4f000000, 0xcf000000, 0, 0x3f800000, 0xbf800000},
    {0x4b800000, 0xcb800000, 0x4b800001, 0xcb800001, 0x4effffff, 0xcf000000, 0, 0x3f800000, 0xbf800000},
    {0x4b800000, 0xcb800001, 0x4b800001, 0xcb800002, 0x4effffff, 0xcf000000, 0, 0x3f800000, 0xbf800000},
    {0x4b800001, 0xcb800000, 0x4b800002, 0xcb800001, 0x4f000000, 0xcf000000, 0, 0x3f800000, 0xbf800000},
    {0x4b800001, 0xcb800001, 0x4b800002, 0xcb800002, 0x4f000000, 0xcf000000, 0, 0x3f800000, 0xbf800000}
  };
  static const char *names[] = {"signed FCVT.S.W: RNE near 2^24", "signed FCVT.S.W: RTZ near 2^24",
    "signed FCVT.S.W: RDN near 2^24", "signed FCVT.S.W: RUP near 2^24", "signed FCVT.S.W: RMM near 2^24"};
  input_pattern(source, 9);
  for (unsigned rm = 0; rm < 5; rm++) {
    case_begin(names[rm]);
    AME_CFG(mcfg5, AME_I32);
    AME_LOAD_WHOLE(mlc.whole, acc1, cvt_input);
    AME_CSR_WRITE(0x809, rm);
    AME_CSR_WRITE(0x003, SCALAR_SENTINEL);
    ame_flags_clear();
    AME_CVT(mcfg5, acc1, AME_F16);
    AME_STORE_WHOLE(msc.whole, acc1, cvt_output);
    expect_pattern(expected[rm], 9);
    CHECK_U32("signed FCVT.S.W NX only, no nominal FP16 overflow", AME_CSR_READ(0x808), AME_NX);
    CHECK_U32("signed FCVT.S.W no msat", AME_CSR_READ(0x807), 0);
    scalar_is_unchanged();
  }
}

static void signed_invalid_results(void) {
  static const uint32_t source[] = {0x7fc12345, 0x7f812345, 0x7f800000, 0xff800000,
                                     0x4f000000, 0xcf000000, 0xcf000001};
  static const uint32_t expected[] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x80000000,
                                       0x7fffffff, 0x80000000, 0x80000000};
  input_pattern(source, 7);
  for (unsigned sat = 0; sat < 2; sat++) {
    case_begin(sat ? "signed FCVT.W.S: numerical NV results with msaten=1" :
                     "signed FCVT.W.S: numerical NV results with msaten=0");
    AME_CFG(mcfg4, AME_F32);
    AME_LOAD_WHOLE(mlc.whole, acc0, cvt_input);
    AME_CSR_WRITE(0x809, 0);
    AME_CSR_WRITE(0x003, SCALAR_SENTINEL);
    AME_CSR_WRITE(0x80a, sat);
    ame_flags_clear();
    AME_CSR_WRITE(0x808, AME_NX); /* Verify sticky accumulation, not overwrite. */
    AME_CVT(mcfg4, acc0, AME_I32);
    AME_STORE_WHOLE(msc.whole, acc0, cvt_output);
    expect_pattern(expected, 7);
    CHECK_U32("FCVT invalid adds NV to existing NX", AME_CSR_READ(0x808), AME_NV | AME_NX);
    CHECK_U32("FCVT numeric clipping independent of msaten", AME_CSR_READ(0x807), 0);
    scalar_is_unchanged();
  }
  AME_CSR_WRITE(0x80a, 0);
}

static void narrowing_ties(unsigned bf) {
  static const uint32_t half_source[] = {0x3f801000, 0xbf801000, 0x3f803000, 0xbf803000};
  static const uint32_t bf_source[] = {0x3f808000, 0xbf808000, 0x3f818000, 0xbf818000};
  static const uint16_t half_expected[5][4] = {
    {0x3c00, 0xbc00, 0x3c02, 0xbc02}, {0x3c00, 0xbc00, 0x3c01, 0xbc01},
    {0x3c00, 0xbc01, 0x3c01, 0xbc02}, {0x3c01, 0xbc00, 0x3c02, 0xbc01},
    {0x3c01, 0xbc01, 0x3c02, 0xbc02}
  };
  static const uint16_t bf_expected[5][4] = {
    {0x3f80, 0xbf80, 0x3f82, 0xbf82}, {0x3f80, 0xbf80, 0x3f81, 0xbf81},
    {0x3f80, 0xbf81, 0x3f81, 0xbf82}, {0x3f81, 0xbf80, 0x3f82, 0xbf81},
    {0x3f81, 0xbf81, 0x3f82, 0xbf82}
  };
  static const char *half_names[] = {"FP16 store: RNE midpoint", "FP16 store: RTZ midpoint",
    "FP16 store: RDN midpoint", "FP16 store: RUP midpoint", "FP16 store: RMM midpoint"};
  static const char *bf_names[] = {"BF16 store: RNE midpoint", "BF16 store: RTZ midpoint",
    "BF16 store: RDN midpoint", "BF16 store: RUP midpoint", "BF16 store: RMM midpoint"};
  const uint32_t *source = bf ? bf_source : half_source;
  uint16_t result[4];
  for (unsigned k = 0; k < AME_SLOTS; k++) cvt_input[k] = k < 4 ? source[k] : 0;
  for (unsigned rm = 0; rm < 5; rm++) {
    case_begin(bf ? bf_names[rm] : half_names[rm]);
    ame_shape(1, 4, 1);
    AME_CFG(mcfg6, AME_F32);
    AME_LOAD_WHOLE(mlc.whole, acc2, cvt_input);
    AME_CSR_WRITE(0x809, rm);
    AME_CSR_WRITE(0x003, SCALAR_SENTINEL);
    ame_flags_clear();
    AME_CVT(mcfg6, acc2, bf ? AME_BF16 : AME_F16);
    CHECK_U32("nominal FP conversion delays rounding", AME_CSR_READ(0x808), 0);
    AME_STORE(msc, acc2, result, sizeof(result));
    AME_STORE_WHOLE(msc.whole, acc2, cvt_output);
    for (unsigned k = 0; k < 4; k++) {
      CHECK_U32("narrow midpoint", result[k], bf ? bf_expected[rm][k] : half_expected[rm][k]);
      CHECK_U32("narrow store preserves wide value", cvt_output[k], source[k]);
    }
    CHECK_U32("midpoint conversion NX", AME_CSR_READ(0x808), AME_NX);
    scalar_is_unchanged();
  }
}

static void narrowing_range(unsigned bf, unsigned tiny) {
  static const uint16_t half_over[5][2] = {
    {0x7c00, 0xfc00}, {0x7bff, 0xfbff}, {0x7bff, 0xfc00}, {0x7c00, 0xfbff}, {0x7c00, 0xfc00}
  };
  static const uint16_t bf_over[5][2] = {
    {0x7f80, 0xff80}, {0x7f7f, 0xff7f}, {0x7f7f, 0xff80}, {0x7f80, 0xff7f}, {0x7f80, 0xff80}
  };
  static const uint16_t under[5][2] = {
    {0, 0x8000}, {0, 0x8000}, {0, 0x8001}, {1, 0x8000}, {1, 0x8001}
  };
  /* 2^16 is already outside FP16's exponent range for every RM. BF16
   * shares FP32's exponent range: max-finite FP32 rounds to max-finite BF16
   * under RTZ with NX only. With both signs, each other RM raises OF|NX.
   */
  uint32_t magnitude = tiny ? (bf ? 0x00008000u : 0x33000000u) :
                              (bf ? 0x7f7fffffu : 0x47800000u);
  uint16_t result[2];
  for (unsigned k = 0; k < AME_SLOTS; k++) cvt_input[k] = 0;
  cvt_input[0] = magnitude;
  cvt_input[1] = magnitude | 0x80000000u;
  for (unsigned rm = 0; rm < 5; rm++) {
    case_begin(tiny ? (bf ? "BF16 store: signed underflow, RM0..4" : "FP16 store: signed underflow, RM0..4") :
                      (bf ? "BF16 store: signed finite range boundary, RM0..4" : "FP16 store: signed overflow, RM0..4"));
    ame_shape(1, 2, 1);
    AME_CFG(mcfg6, AME_F32);
    AME_LOAD_WHOLE(mlc.whole, acc2, cvt_input);
    AME_CSR_WRITE(0x809, rm);
    AME_CSR_WRITE(0x003, SCALAR_SENTINEL);
    ame_flags_clear();
    AME_CVT(mcfg6, acc2, bf ? AME_BF16 : AME_F16);
    CHECK_U32("format change does not narrow range", AME_CSR_READ(0x808), 0);
    AME_STORE(msc, acc2, result, sizeof(result));
    for (unsigned k = 0; k < 2; k++)
      CHECK_U32("narrow signed range", result[k], tiny ? under[rm][k] : bf ? bf_over[rm][k] : half_over[rm][k]);
    uint32_t flags = AME_NX | (tiny ? AME_UF : bf && rm == 1 ? 0 : AME_OF);
    CHECK_U32("narrow range exception flags", AME_CSR_READ(0x808), flags);
    scalar_is_unchanged();
  }
}

static void widening_specials(unsigned bf) {
  static const uint16_t half[] = {0, 0x8000, 0x7c00, 0xfc00, 0x7e35, 1, 0x8001, 0x7bff, 0xfbff};
  static const uint16_t bfloat[] = {0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 1, 0x8001, 0x7f7f, 0xff7f};
  static const uint32_t half_wide[] = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000,
                                        0x33800000, 0xb3800000, 0x477fe000, 0xc77fe000};
  static const uint32_t bf_wide[] = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000,
                                      0x00010000, 0x80010000, 0x7f7f0000, 0xff7f0000};
  uint16_t source[9];
  for (unsigned k = 0; k < 9; k++) source[k] = bf ? bfloat[k] : half[k];
  for (unsigned signaling = 0; signaling < 2; signaling++) {
    case_begin(signaling ? (bf ? "BF16 load: sNaN widens to canonical FP32 and NV" : "FP16 load: sNaN widens to canonical FP32 and NV") :
                          (bf ? "BF16 load: exact subnormals/zeros/infinities and quiet NaN" : "FP16 load: exact subnormals/zeros/infinities and quiet NaN"));
    if (signaling) source[4] = bf ? 0x7f81 : 0x7c35;
    ame_shape(1, 9, 1);
    AME_CFG(mcfg6, bf ? AME_BF16 : AME_F16);
    AME_CSR_WRITE(0x809, 0);
    AME_CSR_WRITE(0x003, SCALAR_SENTINEL);
    ame_flags_clear();
    AME_LOAD(mlc, acc2, source, sizeof(source));
    AME_STORE_WHOLE(msc.whole, acc2, cvt_output);
    for (unsigned k = 0; k < 9; k++)
      CHECK_U32("exact widen/canonical NaN", cvt_output[k], bf ? bf_wide[k] : half_wide[k]);
    CHECK_U32("exact widening has no UF/NX", AME_CSR_READ(0x808), signaling ? AME_NV : 0);
    scalar_is_unchanged();
  }
}

static void narrowing_specials(unsigned bf) {
  static const uint32_t half[] = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234,
                                   0x33800000, 0xb3800000, 0x477fe000, 0xc77fe000};
  static const uint32_t bfloat[] = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234,
                                     0x00010000, 0x80010000, 0x7f7f0000, 0xff7f0000};
  static const uint16_t half_expected[] = {0, 0x8000, 0x7c00, 0xfc00, 0x7e00, 1, 0x8001, 0x7bff, 0xfbff};
  static const uint16_t bf_expected[] = {0, 0x8000, 0x7f80, 0xff80, 0x7fc0, 1, 0x8001, 0x7f7f, 0xff7f};
  uint16_t result[9];
  for (unsigned k = 0; k < AME_SLOTS; k++) cvt_input[k] = k < 9 ? (bf ? bfloat[k] : half[k]) : 0;
  for (unsigned signaling = 0; signaling < 2; signaling++) {
    case_begin(signaling ? (bf ? "BF16 store: sNaN becomes canonical BF16 and NV" : "FP16 store: sNaN becomes canonical FP16 and NV") :
                          (bf ? "BF16 store: exact subnormals/zeros/infinities and quiet NaN" : "FP16 store: exact subnormals/zeros/infinities and quiet NaN"));
    if (signaling) cvt_input[4] = 0x7f801234;
    ame_shape(1, 9, 1);
    AME_CFG(mcfg6, AME_F32);
    AME_LOAD_WHOLE(mlc.whole, acc2, cvt_input);
    AME_CSR_WRITE(0x809, 0);
    AME_CSR_WRITE(0x003, SCALAR_SENTINEL);
    ame_flags_clear();
    AME_CVT(mcfg6, acc2, bf ? AME_BF16 : AME_F16);
    CHECK_U32("same category retains even signaling NaN", AME_CSR_READ(0x808), 0);
    AME_STORE(msc, acc2, result, sizeof(result));
    AME_STORE_WHOLE(msc.whole, acc2, cvt_output);
    for (unsigned k = 0; k < 9; k++) {
      CHECK_U32("exact narrow/canonical NaN", result[k], bf ? bf_expected[k] : half_expected[k]);
      CHECK_U32("special narrow preserves source FP32", cvt_output[k], cvt_input[k]);
    }
    CHECK_U32("exact narrowing has no UF/NX", AME_CSR_READ(0x808), signaling ? AME_NV : 0);
    scalar_is_unchanged();
  }
}

void test_convert(void) {
  raw_configuration();
  whole_cross_category();
  signed_float_to_integer();
  signed_integer_to_float();
  signed_invalid_results();
  narrowing_ties(0);
  narrowing_ties(1);
  narrowing_range(0, 0);
  narrowing_range(0, 1);
  narrowing_range(1, 0);
  narrowing_range(1, 1);
  widening_specials(0);
  widening_specials(1);
  narrowing_specials(0);
  narrowing_specials(1);
  AME_CSR_WRITE(0x809, 0);
  AME_CSR_WRITE(0x80a, 0);
  AME_CSR_WRITE(0x003, 0);
  ame_flags_clear();
}
