#if (defined(MXFP_RMS_ONLY) && MXFP_RMS_ONLY) || \
    (defined(MXFP_MODEL_ACTIVATION) && MXFP_MODEL_ACTIVATION) || \
    (defined(MXFP_MODEL_WEIGHT) && MXFP_MODEL_WEIGHT) || \
    (defined(MXFP_MODEL_LMHEAD) && MXFP_MODEL_LMHEAD) || \
    (defined(F16_MODEL_ATTENTION) && F16_MODEL_ATTENTION)
// Read real libc/GGML declarations BEFORE AM's printf macro. Do not combine
// klib's time()/stdio aliases with GGML's public headers or invent a stub API.
#include "ggml-cpu/vec.h"
#include "ggml-cpu/arch/riscv/ame/ame.h"
#if (defined(MXFP_MODEL_ACTIVATION) && MXFP_MODEL_ACTIVATION) || \
    (defined(MXFP_MODEL_LMHEAD) && MXFP_MODEL_LMHEAD)
#include "ggml-cpu/quants.h"
#endif
#if (defined(F16_MODEL_ATTENTION) && F16_MODEL_ATTENTION) || \
    (defined(MXFP_MODEL_LMHEAD_OUTER) && MXFP_MODEL_LMHEAD_OUTER)
#include "ggml-cpu/ggml-cpu-impl.h"
#endif
#if defined(F16_MODEL_ATTENTION) && F16_MODEL_ATTENTION
#include "ggml-cpu/llamafile/sgemm.h"
#if F16_MODEL_ATTENTION_SOFTMAX || F16_MODEL_ATTENTION_ROPE
#include "ggml-cpu/ops.h"
#endif
#endif
#if defined(MXFP_MODEL_LMHEAD_OUTER) && MXFP_MODEL_LMHEAD_OUTER
#include "ggml-cpu/ops.h"
#endif
#include <am.h>
#include <printf.h>
#else
#include <klib.h>
#endif
#include <stdint.h>

#if defined(MXFP_CPU_QUANT) && MXFP_CPU_QUANT
#if !defined(__riscv_v)
#error "MXFP_CPU_QUANT requires the canonical RVV quantize_row_mxfp4 path"
#endif
#if defined(NDEBUG) || defined(__FAST_MATH__)
#error "MXFP_CPU_QUANT requires assertions and strict floating-point semantics"
#endif
// Old modes retain klib; RMS-only already read the real public declarations.
#undef assert
#include <assert.h>
#include <errno.h>
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#if !defined(MXFP_MODEL_ACTIVATION) || !MXFP_MODEL_ACTIVATION
extern void quantize_row_mxfp4(const float *x, void *y, int64_t k);
#endif
extern void quantize_row_mxfp4_ref(const float *x, void *y, int64_t k);
extern _Bool ggml_ame_pack_mxfp4_blocks(
    uint8_t *payload, uint8_t *scales, const void *src, int64_t nblocks);
#if (defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD) || \
    (defined(MXFP_QUANT_ONLY) && MXFP_QUANT_ONLY)
extern _Bool ggml_mxfp4_prepare_supplied_scales(
    const float *x, uint8_t *scales, int64_t k);
#endif
extern float log2f(float x);
extern float floorf(float x);
extern double frexp(double x, int *exponent);
extern double ldexp(double x, int exponent);
// This is an alias of libc's real TLS definition, not a replacement errno.
extern _Thread_local int mxruntime_libc_errno __asm__("errno");
static _Thread_local volatile uint32_t mxruntime_initialized
    __attribute__((aligned(4))) = 0x13579bdf;
static _Thread_local volatile uint32_t mxruntime_zero[33]
    __attribute__((aligned(128)));
#endif

void matrix_init_128(void);
int  matrix_gemm_128(void);
void matrix_init_64(void);
int  matrix_gemm_64(void);
int  matrix_gemm_mxfp8_e4m3(void);
int  matrix_gemm_mxfp8_e5m2(void);
int  matrix_gemm_mxfp4(void);
uint64_t matrix_gemm_mxcase(uint64_t mcfg, uint64_t k, const uint8_t *sa,
                            const uint8_t *sb, const uint8_t *a,
                            const uint8_t *b, uint32_t c, uint64_t stride);
uint64_t matrix_test_xmfrm(uint64_t rm);
void matrix_test_mxlife(uint64_t *observed);
typedef struct {
  uint64_t rows, tile_k, k_tiles, payload_stride, scale_stride, out_stride;
  uint64_t convert_f32; // appended opt-in; the original six offsets are unchanged
} MxPanelConfig;
uint64_t matrix_test_mxpanel(const uint8_t *a, const uint8_t *b,
                           const uint8_t *sa, const uint8_t *sb, uint32_t *c,
                           const MxPanelConfig *config);
#if defined(MXFP_RMS_DYNAMIC) && MXFP_RMS_DYNAMIC
// The original panel ABI/initializers stay unchanged. Only the new entry
// reads this appended control, then reuses that owner's MMACC/MSC sequence.
typedef struct {
  MxPanelConfig panel;
  uint64_t stage, side, dynamic; // offsets56/64/72: load=1, consume=2; A=0/B=1
} MxRmsPanelConfig;
uint64_t matrix_test_mxrms_panel(const uint8_t *a, const uint8_t *b,
                               const uint8_t *sa, const uint8_t *sb, uint32_t *c,
                               const MxRmsPanelConfig *config);
extern void dequantize_row_mxfp4(const block_mxfp4 *x, float *y, int64_t k);
#endif
extern uint32_t mat_dist[];

typedef struct {
  uint64_t type;
  uint64_t lanes;
  uint64_t sa;
  uint64_t sb;
  uint64_t c;
  uint64_t result;
  uint64_t flags;
  uint8_t a[3];
  uint8_t b[3];
  uint64_t a_rest;
  uint64_t b_rest;
  uint64_t a32;
  uint64_t b32;
  uint64_t sa1;
  uint64_t sb1;
} MxCase;

static const MxCase mx_cases[] = {
  // 35 frozen exact cases shared with NEMU's mxfp_test exact-block fixture.
  {0x17, 3, 127, 127, 0, 0x3f800000, 0, {0x7b, 0x3c, 0xfb}, {0x7b, 0x3c, 0x7b}},
  {0x17, 3, 127, 127, 0, 0x2f800000, 0, {0x7b, 0x01, 0xfb}, {0x7b, 0x01, 0x7b}},
  {0x18, 1, 254, 0, 0, 0x43e00000, 0, {0x7e}, {0x38}},
  {0x17, 2, 254, 254, 0x3f800000, 0x3f800000, 0, {0x7b, 0xfb}, {0x7b, 0x7b}},
  {0x17, 2, 127, 127, 0xbf800000, 0x33800000, 0, {0x3c, 0x0c}, {0x3c, 0x0c}},
  {0x18, 1, 254, 128, 0xff7fffff, 0x73800000, 0, {0x38}, {0x38}},
  {0x18, 1, 0, 105, 0, 0x00000001, 0, {0x38}, {0x38}},
  {0x18, 1, 0, 104, 0, 0x00000000, 3, {0x38}, {0x38}},
  {0x18, 1, 0, 105, 0, 0x00000002, 3, {0x3c}, {0x38}},
  {0x18, 1, 0, 104, 0, 0x80000000, 3, {0xb8}, {0x38}},
  {0x17, 2, 0, 128, 0, 0x00800000, 3, {0x3c, 0x0c}, {0x3c, 0x8c}},
  {0x17, 1, 0, 0, 0, 0, 3, {0x01}, {0x01}},
  {0x18, 1, 254, 128, 0, 0x7f800000, 5, {0x38}, {0x38}},
  {0x18, 1, 254, 254, 0, 0xff800000, 5, {0xb8}, {0x38}},
  {0x18, 1, 103, 127, 0x3f800000, 0x3f800000, 1, {0x38}, {0x38}},
  {0x18, 1, 103, 127, 0x3f800001, 0x3f800002, 1, {0x38}, {0x38}},
  {0x18, 1, 103, 127, 0xbf800001, 0xbf800002, 1, {0xb8}, {0x38}},
  {0x17, 2, 127, 127, 0x00000001, 0x3f800001, 1, {0x3c, 0x0c}, {0x3c, 0x0c}},
  {0x17, 2, 127, 127, 0x80000001, 0x3f800000, 1, {0x3c, 0x0c}, {0x3c, 0x0c}},
  {0x17, 2, 127, 127, 0x80000001, 0x3f800001, 1, {0x3c, 0x0e}, {0x3c, 0x10}},
  {0x16, 1, 254, 254, 0x80000000, 0x80000000, 0, {0x08}, {0x02}},
  {0x16, 1, 254, 254, 0x00000000, 0x00000000, 0, {0x08}, {0x02}},
  {0x18, 1, 0, 0, 0x80000000, 0x00000000, 0, {0x80}, {0xb8}},
  {0x16, 2, 254, 254, 0x80000000, 0x00000000, 0, {0x02, 0x0a}, {0x02, 0x02}},
  {0x16, 1, 255, 127, 0, 0x7fc00000, 0, {0x00}, {0x02}},
  {0x18, 1, 127, 255, 0, 0x7fc00000, 0, {0x38}, {0x38}},
  {0x18, 1, 127, 127, 0, 0x7fc00000, 0, {0x7f}, {0x38}},
  {0x17, 1, 127, 127, 0, 0x7fc00000, 0, {0x7e}, {0x3c}},
  {0x17, 1, 127, 127, 0, 0x7fc00000, 16, {0x7d}, {0x3c}},
  {0x17, 1, 0, 0, 0, 0x7f800000, 0, {0x7c}, {0x3c}},
  {0x17, 1, 127, 127, 0, 0x7fc00000, 16, {0x00}, {0x7c}},
  {0x17, 2, 127, 127, 0, 0x7fc00000, 16, {0x7c, 0xfc}, {0x3c, 0x3c}},
  {0x17, 1, 127, 127, 0xff800000, 0x7fc00000, 16, {0x7c}, {0x3c}},
  {0x18, 1, 127, 127, 0x7f800001, 0x7fc00000, 16, {0x38}, {0x38}},
  {0x18, 1, 127, 127, 0x7fc00001, 0x7fc00000, 0, {0x38}, {0x38}},
  // 9 K33/K64 ordering/overflow cases.
  {0x16, 33, 103, 103, 0x3f800000, 0x3f800000, 1,
   {0x02}, {0x02}, 0, 0x02, 0x02, 0x02, 103, 103},
  {0x16, 33, 152, 127, 0x4c000000, 0x3f800000, 0,
   {0x0a}, {0x02}, 0, 0x02, 0x02, 0x02, 127, 127},
  {0x16, 64, 254, 128, 0, 0x7f800000, 5,
   {0x02}, {0x02}, 0, 0x02, 0x0a, 0x02, 254, 128},
  {0x17, 33, 103, 103, 0x3f800000, 0x3f800000, 1,
   {0x3c}, {0x3c}, 0, 0x3c, 0x3c, 0x3c, 103, 103},
  {0x17, 33, 152, 127, 0x4c000000, 0x3f800000, 0,
   {0xbc}, {0x3c}, 0, 0x3c, 0x3c, 0x3c, 127, 127},
  {0x17, 64, 254, 128, 0, 0x7f800000, 5,
   {0x3c}, {0x3c}, 0, 0x3c, 0xbc, 0x3c, 254, 128},
  {0x18, 33, 103, 103, 0x3f800000, 0x3f800000, 1,
   {0x38}, {0x38}, 0, 0x38, 0x38, 0x38, 103, 103},
  {0x18, 33, 152, 127, 0x4c000000, 0x3f800000, 0,
   {0xb8}, {0x38}, 0, 0x38, 0x38, 0x38, 127, 127},
  {0x18, 64, 254, 128, 0, 0x7f800000, 5,
   {0x38}, {0x38}, 0, 0x38, 0xb8, 0x38, 254, 128},
};

static const char *const mx_case_names[] = {
  "cancellation",
  "full-product-lattice-cancellation",
  "one-side-overflow",
  "large-products-cancel",
  "no-block-prerounding",
  "wide-block-C-cancel",
  "min-subnormal",
  "subnormal-half-even",
  "subnormal-half-odd",
  "negative-underflow",
  "normal-result-underflow",
  "extreme-small-scale",
  "overflow",
  "negative-overflow",
  "tie-even",
  "tie-odd",
  "negative-tie-odd",
  "double-round-above",
  "double-round-below",
  "double-round-odd-below",
  "negative-zero",
  "mixed-zero",
  "positive-zero",
  "exact-zero",
  "scale-A-NaN",
  "scale-B-NaN",
  "E4M3-NaN",
  "E5M2-qNaN",
  "E5M2-sNaN",
  "infinity",
  "zero-times-infinity",
  "opposite-infinities",
  "C-opposite-infinity",
  "C-sNaN",
  "C-qNaN",
  "FP4 K33 ordered two half-ulp blocks",
  "FP4 K33 ascending-K cancellation",
  "FP4 K64 ordered overflowing opposite blocks",
  "E5M2 K33 ordered two half-ulp blocks",
  "E5M2 K33 ascending-K cancellation",
  "E5M2 K64 ordered overflowing opposite blocks",
  "E4M3 K33 ordered two half-ulp blocks",
  "E4M3 K33 ascending-K cancellation",
  "E4M3 K64 ordered overflowing opposite blocks",
};

static int run_mx_vector(const char *group, const char *name,
                         unsigned index, unsigned count,
                         uint64_t type, unsigned lanes,
                         const uint8_t *sa, const uint8_t *sb,
                         const uint8_t *raw_a, const uint8_t *raw_b,
                         uint32_t c, uint32_t expected, uint8_t raised_flags) {
  uint8_t a[64] = {0};
  uint8_t b[64] = {0};
  uint64_t stride = type == 0x16 ?
    (lanes > 32 ? 32 : 16) : (lanes > 32 ? 64 : 32);

  if (type == 0x16) {
    for (unsigned k = 0; k < lanes; k++) {
      uint8_t av = raw_a[k] & 0xf;
      uint8_t bv = raw_b[k] & 0xf;
      if (!(k & 1)) {
        a[k >> 1] = av;
        b[k >> 1] = bv;
      } else {
        a[k >> 1] |= av << 4;
        b[k >> 1] |= bv << 4;
      }
    }
  } else {
    for (unsigned k = 0; k < lanes; k++) {
      a[k] = raw_a[k];
      b[k] = raw_b[k];
    }
  }

  uint64_t blocks = (lanes + 31) / 32;
  printf("MXinput=%02u name=%s a=", index, name);
  for (unsigned k = 0; k < lanes; k++) {
    uint8_t a_bits = type == 0x16 ?
      ((a[k >> 1] >> ((k & 1) ? 4 : 0)) & 0xf) : a[k];
    printf("%02x", a_bits);
    if (k + 1 == lanes) {
      printf(" b=");
    }
  }
  for (unsigned k = 0; k < lanes; k++) {
    uint8_t b_bits = type == 0x16 ?
      ((b[k >> 1] >> ((k & 1) ? 4 : 0)) & 0xf) : b[k];
    printf("%02x", b_bits);
  }
  printf(" sa=");
  for (unsigned k = 0; k < blocks; k++) {
    printf("%02x", sa[k]);
  }
  printf(" sb=");
  for (unsigned k = 0; k < blocks; k++) {
    printf("%02x", sb[k]);
  }
  printf("\n");

  uint64_t packed = matrix_gemm_mxcase(type, lanes, sa, sb, a, b, c, stride);
  uint32_t result = (uint32_t)packed;
  uint32_t actual_flags = (uint32_t)(packed >> 32);

  uint32_t expected_flags = (uint32_t)raised_flags | 8u;
  int pass = result == expected && actual_flags == expected_flags;
  printf("MXcase=%02u/%u group=%s name=%s type=0x%02x K=%u "
         "c=0x%08x result=0x%08x flags=0x%02x "
         "expected=0x%08x/0x%02x %s\n",
         index, count, group, name, (unsigned)type, lanes, c, result,
         actual_flags, expected, expected_flags, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

static int run_mx_cases(const char *mode) {
  const unsigned case_count = sizeof(mx_cases) / sizeof(mx_cases[0]);
  const unsigned name_count = sizeof(mx_case_names) / sizeof(mx_case_names[0]);
  if (case_count != 44 || name_count != case_count) {
    printf("MX vector metadata error: cases=%u names=%u\n", case_count,
           name_count);
    return 1;
  }

  int failed = 0;
  for (unsigned i = 0; i < case_count; i++) {
    uint8_t sa[2] = {mx_cases[i].sa, mx_cases[i].sa1};
    uint8_t sb[2] = {mx_cases[i].sb, mx_cases[i].sb1};
    uint8_t a[64] = {0};
    uint8_t b[64] = {0};
    uint64_t lanes = mx_cases[i].lanes;
    for (unsigned k = 0; k < lanes; k++) {
      a[k] = k < 3 ? mx_cases[i].a[k] :
        (k == 32 ? mx_cases[i].a32 : mx_cases[i].a_rest);
      b[k] = k < 3 ? mx_cases[i].b[k] :
        (k == 32 ? mx_cases[i].b32 : mx_cases[i].b_rest);
    }
    failed |= run_mx_vector(i < 35 ? "exact" : "K33/K64", mx_case_names[i],
                            i, case_count, mx_cases[i].type, lanes, sa, sb,
                            a, b, (uint32_t)mx_cases[i].c,
                            (uint32_t)mx_cases[i].result,
                            (uint8_t)mx_cases[i].flags);
  }
  printf("MX vectors: %s (35 frozen exact cases + 9 K33/K64 ordering/overflow cases)\n",
         failed ? "FAIL" : "PASS");
  return failed;
}

static int run_scale_ff_cases(const char *mode) {
  static const uint64_t types[] = {0x17, 0x18, 0x16};
  static const char *const formats[] = {"E5M2", "E4M3", "FP4"};
  static const uint8_t one[] = {0x3c, 0x38, 0x02};
  static const uint8_t poison[] = {0x7d, 0x7f, 0x07};
  static const uint8_t poison_second[] = {0x7c, 0x7e, 0x0f};
  static const char *const sides[] = {"A", "B", "AB"};
  static const struct {
    const char *name;
    uint32_t bits;
    uint8_t flags;
  } c_values[] = {
    {"finite-C", 0x3f800000, 0},
    {"positive-Inf-C", 0x7f800000, 0},
    {"negative-Inf-C", 0xff800000, 0},
    {"qNaN-C", 0x7fc00001, 0},
    {"sNaN-C", 0x7f800001, 16},
  };
  static const struct {
    const char *name;
    uint8_t sa;
    uint8_t sb;
    uint8_t flags;
    uint32_t c;
  } prior[] = {
    {"OF-NX", 254, 128, 5, 0},
    {"UF-NX", 0, 104, 3, 0},
    {"NX", 103, 127, 1, 0x3f800000},
  };
  static const char *const prior_names[3][3][2] = {
    {
      {"E5M2 OF-NX then scale-A-FF", "E5M2 OF-NX then scale-B-FF"},
      {"E5M2 UF-NX then scale-A-FF", "E5M2 UF-NX then scale-B-FF"},
      {"E5M2 NX then scale-A-FF", "E5M2 NX then scale-B-FF"},
    },
    {
      {"E4M3 OF-NX then scale-A-FF", "E4M3 OF-NX then scale-B-FF"},
      {"E4M3 UF-NX then scale-A-FF", "E4M3 UF-NX then scale-B-FF"},
      {"E4M3 NX then scale-A-FF", "E4M3 NX then scale-B-FF"},
    },
    {
      {"FP4 OF-NX then scale-A-FF", "FP4 OF-NX then scale-B-FF"},
      {"FP4 UF-NX then scale-A-FF", "FP4 UF-NX then scale-B-FF"},
      {"FP4 NX then scale-A-FF", "FP4 NX then scale-B-FF"},
    },
  };
  static const char *const later_names[] = {
    "E5M2 scale-A-FF then finite-scale poison",
    "E4M3 scale-A-FF then finite-scale poison",
    "FP4 scale-A-FF then finite-scale poison",
  };

  unsigned index = 0;
  int failed = 0;
  char name[128];
  for (unsigned t = 0; t < 3; t++) {
    uint8_t a[64] = {0};
    uint8_t b[64] = {0};
    for (unsigned side = 0; side < 3; side++) {
      uint8_t sa[2] = {side == 1 ? 127 : 255, 0};
      uint8_t sb[2] = {side == 0 ? 127 : 255, 0};
      for (unsigned k = 0; k < 32; k++) {
        a[k] = k == 0 ? poison[t] : (k == 1 ? poison_second[t] : 0);
        b[k] = k == 2 ? poison[t] : (k == 3 ? poison_second[t] : 0);
      }
      for (unsigned ci = 0; ci < 5; ci++) {
        snprintf(name, sizeof(name), "%s scale-%s-FF %s no-dot",
                 formats[t], sides[side], c_values[ci].name);
        failed |= run_mx_vector("scaleFF", name, index, 66, types[t], 32,
                                sa, sb, a, b, c_values[ci].bits,
                                0x7fc00000, c_values[ci].flags);
        index++;
      }
    }

    for (unsigned pi = 0; pi < 3; pi++) {
      for (unsigned side = 0; side < 2; side++) {
        uint8_t sa[2] = {prior[pi].sa,
                         side == 0 ? 255 : 127};
        uint8_t sb[2] = {prior[pi].sb,
                         side == 1 ? 255 : 127};
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        a[0] = one[t];
        b[0] = one[t];
        a[32] = poison[t];
        a[33] = poison_second[t];
        b[34] = poison[t];
        b[35] = poison_second[t];
        failed |= run_mx_vector("scaleFF", prior_names[t][pi][side],
                                index, 66, types[t], 64,
                                sa, sb, a, b, prior[pi].c, 0x7fc00000,
                                prior[pi].flags);
        index++;
      }
    }

    uint8_t sa[2] = {255, 127};
    uint8_t sb[2] = {127, 127};
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    for (unsigned k = 0; k < 64; k++) {
      a[k] = (k == 0 || k == 32) ? poison[t] :
        ((k == 1 || k == 33) ? poison_second[t] : 0);
      b[k] = (k == 2 || k == 34) ? poison[t] :
        ((k == 3 || k == 35) ? poison_second[t] : 0);
    }
    failed |= run_mx_vector("scaleFF", later_names[t], index, 66,
                            types[t], 64,
                            sa, sb, a, b, 0, 0x7fc00000,
                            types[t] == 0x17 ? 16 : 0);
    index++;
  }

  printf("MX scale-FF directed vectors: %s (66/66)\n",
         failed || index != 66 ? "FAIL" : "PASS");
  return failed || index != 66;
}

static int run_xmfrm_traps(const char *mode) {
  int failed = 0;
  for (uint64_t rm = 1; rm < 8; rm++) {
    uint64_t packed = matrix_test_xmfrm(rm);
    uint32_t result = (uint32_t)packed;
    uint64_t actual_rm = (packed >> 32) & 0xff;
    uint64_t cause = (packed >> 40) & 0xff;
    uint64_t traps = (packed >> 48) & 0xff;
    int pass = result == 0x12345678 && actual_rm == rm && cause == 2 &&
               traps == 1;
    printf("MXFRM rm=%u result=0x%08x xmfrm=%u cause=%u traps=%u %s\n",
           (unsigned)rm, result, (unsigned)actual_rm, (unsigned)cause,
           (unsigned)traps, pass ? "PASS" : "FAIL");
    if (!pass) {
      failed = 1;
    }
  }
  printf("MX xmfrm non-RNE traps: %s (7/7)\n", failed ? "FAIL" : "PASS");
  return failed;
}

static int run_mx_lifecycle(void) {
  static const struct {
    const char *name;
    uint64_t expected;
  } checks[] = {
    {"invalid-result", 0x7fc00000},
    {"acquired-NX-NV", 17},
    {"scalar-flags", 2},
    {"idle-minit-MS", 1},
    {"idle-minit-sticky", 17},
    {"idle-minit-scalar", 2},
    {"reloaded-scale-result", 0x40000000},
    {"reloaded-scale-sticky", 17},
    {"reused-scale-result", 0x40000000},
    {"reused-scale-sticky", 17},
    {"load-minit-MS", 1},
    {"load-minit-sticky", 17},
    {"second-reload-result", 0x40800000},
    {"second-reload-sticky", 17},
    {"final-scalar-flags", 2},
  };
  const unsigned count = sizeof(checks) / sizeof(checks[0]);
  uint64_t observed[sizeof(checks) / sizeof(checks[0])];
  for (unsigned i = 0; i < count; i++) {
    observed[i] = UINT64_MAX;
  }
  matrix_test_mxlife(observed);
  int failed = 0;
  for (unsigned i = 0; i < count; i++) {
    int pass = observed[i] == checks[i].expected;
    printf("MXlife=%02u/%u name=%s actual=0x%08x expected=0x%08x %s\n",
           i, count, checks[i].name, (uint32_t)observed[i],
           (uint32_t)checks[i].expected, pass ? "PASS" : "FAIL");
    failed |= !pass;
  }
  printf("MX lifecycle: %s (MINIT busy-drain coverage requires trace)\n",
         failed ? "FAIL" : "PASS");
  return failed;
}

static int run_mx_panel(void) {
  static uint8_t a[128 * 16] __attribute__((aligned(64)));
  static uint8_t b[128 * 16] __attribute__((aligned(64)));
  static uint8_t sa[128] __attribute__((aligned(64)));
  static uint8_t sb[128] __attribute__((aligned(64)));
  static uint32_t c[128 * 128] __attribute__((aligned(64)));
  static const int q[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  // Exactly the first native128 FP4 K32 inputs, with consecutive-K nibbles.
  for (unsigned row = 0; row < 128; ++row) {
    sa[row] = 125 + row % 5;
    sb[row] = 124 + row % 3;
    for (unsigned lane = 0; lane < 32; ++lane) {
      unsigned acode = (row * 3 + lane) & 15;
      int source = 2 * ((int)((row * 5 + lane) % 15) - 7);
      unsigned bcode = 0;
      int error = source < 0 ? -source : source;
      for (unsigned code = 1; code < 16; ++code) {
        int diff = q[code] - source;
        if (diff < 0) diff = -diff;
        if (diff < error) { error = diff; bcode = code; }
      }
      unsigned shift = (lane & 1) * 4;
      if (!(lane & 1)) { a[row * 16 + lane / 2] = 0; b[row * 16 + lane / 2] = 0; }
      a[row * 16 + lane / 2] |= acode << shift;
      b[row * 16 + lane / 2] |= bcode << shift;
    }
  }
  for (unsigned i = 0; i < 128 * 128; ++i) c[i] = 0x7fc56789;
  const MxPanelConfig config = {128, 32, 1, 16, 1, 128 * sizeof(float), 0};
  uint64_t flags = matrix_test_mxpanel(a, b, sa, sb, c, &config);
  unsigned mismatches = 0;
  for (unsigned row = 0; row < 128; ++row) {
    for (unsigned col = 0; col < 128; ++col) {
      int sum = 0;
      for (unsigned lane = 0; lane < 32; ++lane) {
        unsigned shift = (lane & 1) * 4;
        unsigned acode = (a[row * 16 + lane / 2] >> shift) & 15;
        unsigned bcode = (b[col * 16 + lane / 2] >> shift) & 15;
        sum += q[acode] * q[bcode];
      }
      // Finite bounded dyadic sums are exact; one K32 block, no dot preround.
      union { float f; uint32_t u; } expected;
      expected.f = (float)sum / (float)(1u << (256 - sa[row] - sb[col]));
      if (c[row * 128 + col] != expected.u) {
        if (mismatches < 8)
          printf("MXpanel mismatch row=%u col=%u expected=0x%08x actual=0x%08x\n",
                 row, col, expected.u, c[row * 128 + col]);
        ++mismatches;
      }
    }
  }
  printf("MXpanel M=128 N=128 K=32 MLA+MLB exact_outputs=16384 mismatches=%u flags=0x%02x %s\n",
         mismatches, (unsigned)flags, !mismatches && !flags ? "PASS" : "FAIL");
  return mismatches || flags;
}

#ifndef MXFP_PERF_MN
#define MXFP_PERF_MN 512
#endif
#if MXFP_PERF_MN != 128 && MXFP_PERF_MN != 512
#error "MXFP_PERF_MN must be 128 or 512; performance K remains 512"
#endif

#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
static int run_mx_perf_mode(int supplied_convert) {
#else
static int run_mx_perf(void) {
#endif
  enum { rows = MXFP_PERF_MN, k = 512, tile = 64, blocks = k / 32 };
  const char *tag = rows == 512 ? "MXperf512" : "MXperf128x128x512";
  static uint8_t a[rows * k / 2] __attribute__((aligned(64)));
  static uint8_t b[rows * k / 2] __attribute__((aligned(64)));
  static uint8_t sa[rows * blocks] __attribute__((aligned(64)));
  static uint8_t sb[rows * blocks] __attribute__((aligned(64)));
#if defined(MXFP_CPU_QUANT) && MXFP_CPU_QUANT
  static float f32_a[rows * k] __attribute__((aligned(64)));
  static float f32_b[rows * k] __attribute__((aligned(64)));
  block_mxfp4 quantized_a[blocks], quantized_b[blocks];
#endif
  static struct {
    uint32_t before[16], values[rows * rows], after[16];
  } output __attribute__((aligned(64)));
  static const int q[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  uint64_t tlenb, trlenb, alenb;
  // Matrix geometry CSRs themselves require an enabled matrix context.
  // Reuse the existing bare-metal initializer before probing them.
  matrix_init_64();
  __asm__ volatile("csrr %0, 0xcc1" : "=r"(tlenb));
  __asm__ volatile("csrr %0, 0xcc2" : "=r"(trlenb));
  __asm__ volatile("csrr %0, 0xcc3" : "=r"(alenb));
  // Reject the existing 128-SCP NEMU/R14 configuration, not silently label
  // a 64-subtile on that different geometry as the requested 2T64 target.
  if (tlenb != 4096 || trlenb != 64 || alenb != 16384) {
    printf("%s REJECT geometry tlenb=%lu trlenb=%lu alenb=%lu "
           "expected=4096/64/16384\n", tag, tlenb, trlenb, alenb);
    return 1;
  }
#if defined(MXFP_CPU_QUANT) && MXFP_CPU_QUANT
  // Generate every F32 input outside the timer, not quantized pattern copies.
  // All original blocks contain max |q|=12, so canonical amax/log2f/floorf
  // recovers the original e. Negative-zero code8 may normalize to code0.
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned block = 0; block < blocks; ++block) {
      const unsigned ea = 125 + ((row % 8 + block) % 3) + row / tile;
      const unsigned eb = 125 + ((row % 8 + 2 * block) % 3) + row / tile;
      union { uint32_t u; float f; } da, db;
      da.u = (ea - 1) << 23; // exact fixture scale d=2^(e-128)
      db.u = (eb - 1) << 23;
      for (unsigned lane = 0; lane < 32; ++lane) {
        f32_a[row * k + block * 32 + lane] =
            (float) q[(3 * (row % 8) + 5 * block + lane) & 15] * da.f;
        f32_b[row * k + block * 32 + lane] =
            (float) q[((row % 8) + 3 * block + 3 * lane + 1) & 15] * db.f;
      }
    }
  }
#else
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned block = 0; block < blocks; ++block) {
      // MLS consumes a compact [64 rows][2 K32 scales] K64 panel.
      // Keep SF separate from payload; every 128-byte panel is aligned.
      const unsigned scale_index = ((row / tile) * (k / 64) + block / 2) *
          (tile * 2) + (row % tile) * 2 + block % 2;
      sa[scale_index] = 125 + ((row % 8 + block) % 3) + row / tile;
      sb[scale_index] = 125 + ((row % 8 + 2 * block) % 3) + row / tile;
      for (unsigned lane = 0; lane < 32; lane += 2) {
        const unsigned acode0 = (3 * (row % 8) + 5 * block + lane) & 15;
        const unsigned bcode0 = ((row % 8) + 3 * block + 3 * lane + 1) & 15;
        a[row * (k / 2) + block * 16 + lane / 2] =
            acode0 | (((acode0 + 1) & 15) << 4);
        b[row * (k / 2) + block * 16 + lane / 2] =
            bcode0 | (((bcode0 + 3) & 15) << 4);
      }
    }
  }
#endif
  for (unsigned i = 0; i < 16; ++i) {
    output.before[i] = 0x13579bdf;
    output.after[i] = 0x2468ace0;
  }
  for (unsigned i = 0; i < rows * rows; ++i) output.values[i] = 0x7fc56789;
  MxPanelConfig config = {tile, 64, k / 64, k / 2, 2,
                         rows * sizeof(float), 0};
#if defined(MXFP_CPU_QUANT) && MXFP_CPU_QUANT
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
  int use_conversion = supplied_convert;
  unsigned preflight_rows = 0, conversion_panels = 0, fallback_count = 0;
  uint64_t rejected_flags = 0;
  if (supplied_convert) {
    printf("%s experimental=convert_load supplied_scale=CPU "
           "scalar_flags=read_only preflight=canonical "
           "reject=CPU_quant_ordinary_load_restart\n", tag);
  }
#endif
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
  if (supplied_convert) {
    printf("%s begin M=%u N=%u K=%u native_tile=64x64x64 "
           "warmup=0 runs=1 scope=f32_preflight_scale_convert_load_mma_sync_store "
           "input_init=outside cache_flush=no quantization=supplied_scale_RTL "
           "preparation_timed=yes\n", tag, rows, rows, k);
  } else
#endif
  printf("%s begin M=%u N=%u K=%u native_tile=64x64x64 "
         "warmup=0 runs=1 scope=f32_quant_pack_scale_load_mma_sync_store "
         "input_init=outside cache_flush=no quantization=RVV quant_timed=yes\n",
         tag, rows, rows, k);
  uint64_t prepared_end;
  unsigned quantized_rows = 0, packed_blocks = 0;
#else
  printf("%s begin M=%u N=%u K=%u native_tile=64x64x64 "
         "warmup=0 runs=1 scope=prepared_load_scale_mma_sync_store "
         "input_init=outside cache_flush=no quantization=not_timed\n",
         tag, rows, rows, k);
#endif
  uint64_t begin, end, flags = 0;
  __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(begin) : : "memory");
#if defined(MXFP_CPU_QUANT) && MXFP_CPU_QUANT
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
  if (use_conversion) {
    uint8_t row_sa[blocks], row_sb[blocks];
    for (unsigned row = 0; row < rows; ++row) {
      ++preflight_rows;
      const int a_ok = ggml_mxfp4_prepare_supplied_scales(
          f32_a + row * k, row_sa, k);
      ++preflight_rows;
      const int b_ok = ggml_mxfp4_prepare_supplied_scales(
          f32_b + row * k, row_sb, k);
      if (!a_ok || !b_ok) {
        use_conversion = 0;
        ++fallback_count;
        break;
      }
      for (unsigned block = 0; block < blocks; block += 2) {
        const unsigned scale_index = ((row / tile) * (k / 64) + block / 2) *
            (tile * 2) + (row % tile) * 2;
        sa[scale_index] = row_sa[block];
        sa[scale_index + 1] = row_sa[block + 1];
        sb[scale_index] = row_sb[block];
        sb[scale_index + 1] = row_sb[block + 1];
      }
    }
    if (use_conversion && supplied_convert == 2) {
      // Explicit negative fixture, never a performance variant: one illegal
      // K32 SF must cause owned NV and a real whole-operator CPU restart.
      sa[0] = 255;
      printf("%s negative_fixture=one_K32_illegal_scale "
             "expected=NV_then_CPU_restart performance_sample=no\n", tag);
    }
  }
mxperf512_cpu_prepare:
  if (!use_conversion) {
#endif
  // Quantize both complete F32 tensors: 2*rows calls and 2*rows*blocks K32s.
  // Never preset NX or clear scalar flags to qualify the canonical fast path.
  for (unsigned row = 0; row < rows; ++row) {
    quantize_row_mxfp4(f32_a + row * k, quantized_a, k);
    quantize_row_mxfp4(f32_b + row * k, quantized_b, k);
    quantized_rows += 2;
    for (unsigned block = 0; block < blocks; block += 2) {
      const unsigned scale_index = ((row / tile) * (k / 64) + block / 2) *
          (tile * 2) + (row % tile) * 2;
      if (!ggml_ame_pack_mxfp4_blocks(
              a + row * (k / 2) + block * 16, sa + scale_index,
              quantized_a + block, 2) ||
          !ggml_ame_pack_mxfp4_blocks(
              b + row * (k / 2) + block * 16, sb + scale_index,
              quantized_b + block, 2)) {
        printf("%s canonical pack FAIL row=%u block=%u\n", tag, row, block);
        return 1;
      }
      packed_blocks += 4;
    }
  }
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
  }
  config.convert_f32 = use_conversion;
  config.payload_stride = use_conversion ? k * sizeof(float) : k / 2;
#endif
  __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(prepared_end) : : "memory");
#endif
  for (unsigned m = 0; m < rows; m += tile) {
    for (unsigned n = 0; n < rows; n += tile) {
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
      const uint8_t *panel_a = use_conversion ?
          (const uint8_t *)(f32_a + m * k) : a + m * (k / 2);
      const uint8_t *panel_b = use_conversion ?
          (const uint8_t *)(f32_b + n * k) : b + n * (k / 2);
      const uint64_t panel_flags = matrix_test_mxpanel(panel_a, panel_b,
          sa + m * blocks, sb + n * blocks, output.values + m * rows + n, &config);
      if (use_conversion && (panel_flags & 16u)) {
        // Every conversion is acquired and checked before MMACC in the
        // existing panel helper. Restart all outputs, counting all work.
        rejected_flags |= panel_flags;
        ++fallback_count;
        use_conversion = 0;
        flags = 0;
        goto mxperf512_cpu_prepare;
      }
      conversion_panels += use_conversion != 0;
      flags |= panel_flags;
#else
      flags |= matrix_test_mxpanel(a + m * (k / 2), b + n * (k / 2),
          sa + m * blocks, sb + n * blocks, output.values + m * rows + n, &config);
#endif
    }
  }
  __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(end) : : "memory");
#if defined(MXFP_CPU_QUANT) && MXFP_CPU_QUANT
  uint32_t scalar_flags;
  __asm__ volatile("frflags %0" : "=r"(scalar_flags) : : "memory");
#endif

  // Compute only 8x8 independent logical patterns, then check all outputs.
  // Tile-row scale adds a common power of two. In 1/64 units every K32
  // partial sum is an integer of magnitude <2^21, so per-K32 FP32 rounding
  // is exact. No giant O(M*N*K) CPU golden runs in the RTL simulator.
  int expected_units[8][8];
  for (unsigned r = 0; r < 8; ++r) for (unsigned c = 0; c < 8; ++c) {
    int total = 0;
    for (unsigned block = 0; block < blocks; ++block) {
      int dot = 0;
      for (unsigned lane = 0; lane < 32; ++lane) {
        dot += q[(3 * r + 5 * block + lane) & 15] *
               q[(c + 3 * block + 3 * lane + 1) & 15];
      }
      total += dot * (1 << ((r + block) % 3 + (c + 2 * block) % 3));
    }
    expected_units[r][c] = total;
  }
  unsigned mismatches = 0, canary_errors = 0;
  for (unsigned row = 0; row < rows; ++row) for (unsigned col = 0; col < rows; ++col) {
    union { float f; uint32_t u; } expected;
    expected.f = (float) expected_units[row % 8][col % 8] *
        (float) (1u << (row / tile + col / tile)) / 64.0f;
    if (output.values[row * rows + col] != expected.u) {
      if (mismatches < 8) {
        printf("%s mismatch row=%u col=%u expected=%08x actual=%08x\n",
            tag, row, col, expected.u, output.values[row * rows + col]);
      }
      ++mismatches;
    }
  }
  for (unsigned i = 0; i < 16; ++i) {
    canary_errors += output.before[i] != 0x13579bdf;
    canary_errors += output.after[i] != 0x2468ace0;
  }
#if defined(MXFP_CPU_QUANT) && MXFP_CPU_QUANT
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
  const int prepared_counts_bad = use_conversion ?
      preflight_rows != 2 * rows || quantized_rows || packed_blocks ||
          conversion_panels != (rows / tile) * (rows / tile) :
      quantized_rows != 2 * rows || packed_blocks != 2 * rows * blocks;
#else
  const int prepared_counts_bad =
      quantized_rows != 2 * rows || packed_blocks != 2 * rows * blocks;
#endif
  const int failed = mismatches || canary_errors || flags || end <= begin ||
      prepared_counts_bad ||
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
      (supplied_convert == 2 &&
       (use_conversion || fallback_count != 1 || rejected_flags != 16)) ||
#endif
      prepared_end <= begin || end <= prepared_end;
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
  printf("%s canonical_cpu_quant=RVV preparation_timed=yes values=%u "
#else
  printf("%s cpu_quant=RVV quant_timed=yes values=%u "
#endif
         "quantized_rows=%u packed_K32_blocks=%u nblocks_per_pack=2 "
         "total_cycles=%lu prepare_cycles=%lu matrix_cycles=%lu scalar_fflags=%u\n",
      tag, 2 * rows * k, quantized_rows, packed_blocks, end - begin, prepared_end - begin,
      end - prepared_end, scalar_flags);
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
  printf("%s requested_conversion=%u used_conversion=%u "
         "preflight_rows=%u conversion_panels=%u fallback_count=%u "
         "rejected_matrix_flags=%lu completed_panel_conversion_K32_blocks=%u "
         "completed_panel_f32_conversion_bytes=%u "
         "prepare_includes_preflight_and_any_retry=yes\n",
      tag, supplied_convert != 0, use_conversion != 0, preflight_rows,
      conversion_panels, fallback_count, rejected_flags,
      conversion_panels * (k / 64) * tile * 4,
      conversion_panels * (k / 64) * tile * 64 * 2 * 4);
#endif
#else
  const int failed = mismatches || canary_errors || flags || end <= begin;
#endif
  printf("%s cycles=%lu useful_ops=%u output_tiles=%u "
         "expected_mmacc=%u exact_outputs=%u mismatches=%u canary_errors=%u "
         "matrix_flags=%lu warmup=0 runs=1 %s\n",
      tag, end - begin, 2 * rows * rows * k, (rows / tile) * (rows / tile),
      (rows / tile) * (rows / tile) * (k / 64), rows * rows,
      mismatches, canary_errors, flags, failed ? "FAIL" : "PASS");
#if defined(MXFP_CPU_QUANT) && MXFP_CPU_QUANT
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
  printf("%s timing_requires=cycle_accurate_RTL_or_hardware "
         "functional_simulator_cycles=not_performance "
         "full_fp32_quantized_operator=timed "
         "canonical_preflight=ggml_mxfp4_prepare_supplied_scales "
         "canonical_fallback=quantize_row_mxfp4+ggml_ame_pack_mxfp4_blocks\n", tag);
#else
  printf("%s timing_requires=cycle_accurate_RTL_or_hardware "
         "functional_simulator_cycles=not_performance "
         "full_fp32_quantized_operator=timed canonical_quant=quantize_row_mxfp4 "
         "canonical_pack=ggml_ame_pack_mxfp4_blocks\n", tag);
#endif
#else
  printf("%s timing_requires=cycle_accurate_RTL_or_hardware "
         "functional_simulator_cycles=not_performance "
         "full_fp32_quantized_operator=not_tested\n", tag);
#endif
  return failed;
}

#if defined(MXFP_CPU_QUANT) && MXFP_CPU_QUANT
static int run_mx_runtime(void) {
  matrix_init_64();
  uintptr_t tp;
  __asm__ volatile("mv %0, tp" : "=r"(tp));
  unsigned errors = 0;
  errors += !tp || (uintptr_t)&mxruntime_initialized % 4 ||
      (uintptr_t)mxruntime_zero % 128;
  errors += mxruntime_initialized != 0x13579bdf;
  for (unsigned i = 0; i < 33; ++i) errors += mxruntime_zero[i] != 0;
  int *libc_errno = __errno_location();
  errors += libc_errno != &mxruntime_libc_errno;
  errors += mxruntime_libc_errno != 0;

  volatile float argument = -1.0f;
  union { float f; uint32_t u; } negative, zero, floored;
  errno = 0;
  negative.f = log2f(argument);
  const int negative_errno = errno;
  errors += negative_errno != EDOM || (negative.u & 0x7fffffff) <= 0x7f800000;
  argument = 0.0f;
  errno = 0;
  zero.f = log2f(argument);
  const int zero_errno = errno;
  errors += zero_errno != ERANGE || zero.u != 0xff800000;
  argument = -1.25f;
  floored.f = floorf(argument);
  errors += floored.u != 0xc0000000;
  volatile double number = 12.0;
  int exponent = 0;
  const double fraction = frexp(number, &exponent);
  const double restored = ldexp(fraction, exponent);
  errors += fraction != 0.75 || exponent != 4 || restored != 12.0;
  mxruntime_initialized = 0x2468ace0;
  mxruntime_zero[0] = 0x11223344;
  mxruntime_zero[32] = 0xa5a55a5a;
  errors += mxruntime_initialized != 0x2468ace0 ||
      mxruntime_zero[0] != 0x11223344 || mxruntime_zero[32] != 0xa5a55a5a;
  uint32_t scalar_flags;
  __asm__ volatile("frflags %0" : "=r"(scalar_flags) : : "memory");
  printf("MXruntime tp=%lx tls_init=%lx align=4 tls_zero=%lx align=128 "
         "errno_location=%lx errno_TLS=%lx log2_neg=%08x errno_neg=%d "
         "log2_zero=%08x errno_zero=%d floor=%08x frexp_exp=%d "
         "scalar_fflags=%u errors=%u %s\n",
      (unsigned long)tp, (unsigned long)(uintptr_t)&mxruntime_initialized,
      (unsigned long)(uintptr_t)mxruntime_zero, (unsigned long)(uintptr_t)libc_errno,
      (unsigned long)(uintptr_t)&mxruntime_libc_errno, negative.u, negative_errno,
      zero.u, zero_errno, floored.u, exponent, scalar_flags, errors,
      errors ? "FAIL" : "PASS");
  return errors != 0;
}

static int run_mx_assert(void) {
  matrix_init_64();
  printf("MXassert begin canonical_quant_k=31 sysroot_assert_ABI "
         "expected=BADTRAP/nonzero must_not_continue\n");
  float values[32] = {0};
  block_mxfp4 quantized;
  quantize_row_mxfp4(values, &quantized, 31);
  printf("MXassert FAIL unexpected continuation\n");
  return 1;
}

typedef struct {
  uint64_t rows, columns, blocks_per_row, total_blocks;
  uint64_t total_f32, payload_bytes, scale_bytes;
} MxQuant128Shape;

enum {
  mxquant128_rows = 128,
  mxquant128_columns = 128,
  mxquant128_blocks_per_row = 4,
  mxquant128_total_blocks = 512,
  mxquant128_total_f32 = 16384,
  mxquant128_payload_bytes = 8192,
  mxquant128_scale_bytes = 512,
};

static const MxQuant128Shape mxquant128_shape = {
  .rows = mxquant128_rows,
  .columns = mxquant128_columns,
  .blocks_per_row = mxquant128_blocks_per_row,
  .total_blocks = mxquant128_total_blocks,
  .total_f32 = mxquant128_total_f32,
  .payload_bytes = mxquant128_payload_bytes,
  .scale_bytes = mxquant128_scale_bytes,
};

enum {
  mxquant128_natural_nx_blocks = 512,
};

typedef struct {
  uint32_t before[16];
  float source[mxquant128_total_f32];
  uint32_t after[16];
} MxQuant128SourceGuard;

typedef struct {
  uint32_t before[16];
  block_mxfp4 blocks[mxquant128_total_blocks];
  uint32_t after[16];
} MxQuant128BlockGuard;

typedef struct {
  uint32_t before[16];
  uint8_t bytes[mxquant128_payload_bytes];
  uint32_t after[16];
} MxQuant128PayloadGuard;

typedef struct {
  uint32_t before[16];
  uint8_t bytes[mxquant128_scale_bytes];
  uint32_t after[16];
} MxQuant128ScaleGuard;

static uint64_t mxfp_quant_hash(const uint8_t *data, size_t size) {
  uint64_t hash = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 0x100000001b3ull;
  }
  return hash;
}

static void mxfp_quant_make_input(float *values) {
  for (unsigned row = 0; row < mxquant128_shape.rows; ++row) {
    for (unsigned block = 0; block < mxquant128_shape.blocks_per_row; ++block) {
      // Piecewise powers keep d exact while row/block variation exercises
      // natural 3*d, 6*d and 12*d K32 maxima without pre-setting MX flags.
      const unsigned target = (row + block) % 3 == 0 ? 3 :
          (row + block) % 3 == 1 ? 6 : 12;
      const unsigned e = 124 + target;
      union { uint32_t u; float f; } d;
      d.u = (e - 1) << 23;  // canonical E8M0 half-scale d=2^(e-128)
      for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned index = row * mxquant128_shape.columns + block * 32 + lane;
        if (lane == 0) {
          values[index] = (row + block) & 1 ? -0.0f : 0.0f;
        } else if (lane == 1) {
          // Non-lattice but strictly inside [target*d, target*d + d/512].
          // Canonical scale selection naturally raises NX; this block amax
          // remains bounded by 12.001953125*d < 16*d.
          values[index] = (target + 0x1p-9f) * d.f;
        } else if (lane == 2) {
          values[index] = ((row + block) & 1 ? -1.0f : 1.0f) * target * d.f;
        } else {
          const unsigned magnitude = 1 + ((5 * row + 3 * block + lane) % (target - 1));
          values[index] = ((row + block + lane) & 1 ? -1.0f : 1.0f) *
              magnitude * d.f;
        }
      }
    }
  }
  values[0] = -0.0f;
  values[127] = 0.0f;
}

static unsigned mxfp_quant_check_input(const float *values,
                                      const uint32_t *preimage,
                                      const uint32_t *before,
                                      const uint32_t *after) {
  unsigned errors = 0;
  for (unsigned i = 0; i < mxquant128_shape.total_f32; ++i) {
    union { float f; uint32_t u; } current;
    current.f = values[i];
    errors += current.u != preimage[i];
  }
  for (unsigned i = 0; i < 16; ++i) {
    errors += before[i] != 0x13579bdf || after[i] != 0x2468ace0;
  }
  return errors;
}

static unsigned mxfp_quant_compare_blocks(const block_mxfp4 *actual,
                                          const block_mxfp4 *reference) {
  unsigned mismatches = 0;
  for (unsigned i = 0; i < mxquant128_shape.total_blocks; ++i) {
    mismatches += actual[i].e != reference[i].e;
    for (unsigned j = 0; j < sizeof(actual[i].qs); ++j)
      mismatches += actual[i].qs[j] != reference[i].qs[j];
  }
  return mismatches;
}

static unsigned mxfp_quant_compare_pack(const uint8_t *payload,
                                        const uint8_t *ref_payload,
                                        const uint8_t *scales,
                                        const uint8_t *ref_scales) {
  unsigned mismatches = 0;
  for (unsigned i = 0; i < mxquant128_payload_bytes; ++i)
    mismatches += payload[i] != ref_payload[i];
  for (unsigned i = 0; i < mxquant128_scale_bytes; ++i)
    mismatches += scales[i] != ref_scales[i];
  return mismatches;
}

static MxQuant128SourceGuard *mxfp_quant_guarded;
static MxQuant128SourceGuard mxfp_quant_source_storage __attribute__((aligned(64)));
static uint32_t mxfp_quant_preimage[mxquant128_total_f32] __attribute__((aligned(64)));
static MxQuant128BlockGuard mxfp_quant_quantized_guard __attribute__((aligned(64)));
static MxQuant128BlockGuard mxfp_quant_reference_guard __attribute__((aligned(64)));
static MxQuant128PayloadGuard mxfp_quant_payload_guard __attribute__((aligned(64)));
static MxQuant128PayloadGuard mxfp_quant_ref_payload_guard __attribute__((aligned(64)));
static MxQuant128ScaleGuard mxfp_quant_scales_guard __attribute__((aligned(64)));
static MxQuant128ScaleGuard mxfp_quant_ref_scales_guard __attribute__((aligned(64)));

static int run_mxquant128(void) {
  mxfp_quant_guarded = &mxfp_quant_source_storage;
  MxQuant128SourceGuard *guarded = mxfp_quant_guarded;
  float *source = guarded->source;
  for (unsigned i = 0; i < 16; ++i) {
    guarded->before[i] = 0x13579bdf;
    guarded->after[i] = 0x2468ace0;
    mxfp_quant_quantized_guard.before[i] = 0x13579bdf;
    mxfp_quant_quantized_guard.after[i] = 0x2468ace0;
    mxfp_quant_reference_guard.before[i] = 0x13579bdf;
    mxfp_quant_reference_guard.after[i] = 0x2468ace0;
    mxfp_quant_payload_guard.before[i] = 0x13579bdf;
    mxfp_quant_payload_guard.after[i] = 0x2468ace0;
    mxfp_quant_ref_payload_guard.before[i] = 0x13579bdf;
    mxfp_quant_ref_payload_guard.after[i] = 0x2468ace0;
    mxfp_quant_scales_guard.before[i] = 0x13579bdf;
    mxfp_quant_scales_guard.after[i] = 0x2468ace0;
    mxfp_quant_ref_scales_guard.before[i] = 0x13579bdf;
    mxfp_quant_ref_scales_guard.after[i] = 0x2468ace0;
  }
  mxfp_quant_make_input(source);
  const unsigned natural_nx = mxquant128_natural_nx_blocks;
  for (unsigned i = 0; i < mxquant128_total_f32; ++i) {
    union { float f; uint32_t u; } bits = { .f = source[i] };
    mxfp_quant_preimage[i] = bits.u;
  }
  for (unsigned i = 0; i < sizeof(mxfp_quant_payload_guard.bytes); ++i)
    mxfp_quant_payload_guard.bytes[i] = 0xa5;
  for (unsigned i = 0; i < sizeof(mxfp_quant_scales_guard.bytes); ++i)
    mxfp_quant_scales_guard.bytes[i] = 0x5a;

  block_mxfp4 *quantized = mxfp_quant_quantized_guard.blocks;
  block_mxfp4 *reference = mxfp_quant_reference_guard.blocks;
  uint8_t *payload = mxfp_quant_payload_guard.bytes;
  uint8_t *ref_payload = mxfp_quant_ref_payload_guard.bytes;
  uint8_t *scales = mxfp_quant_scales_guard.bytes;
  uint8_t *ref_scales = mxfp_quant_ref_scales_guard.bytes;

  uint32_t flags_before, flags_quant, flags_pack, flags_after;
  uint32_t frm_before, frm_after;
  __asm__ volatile("frrm %0" : "=r"(frm_before));
  __asm__ volatile("frflags %0" : "=r"(flags_before) : : "memory");
  uint64_t quant_begin, quant_end, pack_begin, pack_end;
  __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(quant_begin) : : "memory");
  for (unsigned row = 0; row < mxquant128_shape.rows; ++row) {
    quantize_row_mxfp4(source + row * mxquant128_shape.columns,
                       quantized + row * mxquant128_shape.blocks_per_row,
                       mxquant128_shape.columns);
  }
  __asm__ volatile("fence rw, rw\n\trdcycle %0\n\tfrflags %1"
                   : "=r"(quant_end), "=r"(flags_quant) : : "memory");

  __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(pack_begin) : : "memory");
  if (!ggml_ame_pack_mxfp4_blocks(payload, scales, quantized,
                                  mxquant128_shape.total_blocks)) {
    printf("MXquant128 canonical pack FAIL\n");
    return 1;
  }
  __asm__ volatile("fence rw, rw\n\trdcycle %0\n\tfrflags %1"
                   : "=r"(pack_end), "=r"(flags_pack) : : "memory");
  __asm__ volatile("frrm %0\n\tfrflags %1"
                   : "=r"(frm_after), "=r"(flags_after) : : "memory");

  for (unsigned row = 0; row < mxquant128_shape.rows; ++row) {
    quantize_row_mxfp4_ref(source + row * mxquant128_shape.columns,
                           reference + row * mxquant128_shape.blocks_per_row,
                           mxquant128_shape.columns);
  }
  if (!ggml_ame_pack_mxfp4_blocks(ref_payload, ref_scales, reference,
                                  mxquant128_shape.total_blocks)) {
    printf("MXquant128 reference pack FAIL\n");
    return 1;
  }
  unsigned canary_errors =
      mxfp_quant_check_input(source, mxfp_quant_preimage,
                             guarded->before, guarded->after);
  for (unsigned i = 0; i < 16; ++i) {
    canary_errors += mxfp_quant_quantized_guard.before[i] != 0x13579bdf ||
        mxfp_quant_quantized_guard.after[i] != 0x2468ace0;
    canary_errors += mxfp_quant_reference_guard.before[i] != 0x13579bdf ||
        mxfp_quant_reference_guard.after[i] != 0x2468ace0;
    canary_errors += mxfp_quant_payload_guard.before[i] != 0x13579bdf ||
        mxfp_quant_payload_guard.after[i] != 0x2468ace0;
    canary_errors += mxfp_quant_ref_payload_guard.before[i] != 0x13579bdf ||
        mxfp_quant_ref_payload_guard.after[i] != 0x2468ace0;
    canary_errors += mxfp_quant_scales_guard.before[i] != 0x13579bdf ||
        mxfp_quant_scales_guard.after[i] != 0x2468ace0;
    canary_errors += mxfp_quant_ref_scales_guard.before[i] != 0x13579bdf ||
        mxfp_quant_ref_scales_guard.after[i] != 0x2468ace0;
  }
  const unsigned compare_errors =
      mxfp_quant_compare_blocks(quantized, reference) +
      mxfp_quant_compare_pack(payload, ref_payload, scales, ref_scales);
  const uint64_t payload_hash = mxfp_quant_hash(payload, mxquant128_payload_bytes);
  const uint64_t scale_hash = mxfp_quant_hash(scales, mxquant128_scale_bytes);
  const uint64_t ref_payload_hash =
      mxfp_quant_hash(ref_payload, mxquant128_payload_bytes);
  const uint64_t ref_scale_hash =
      mxfp_quant_hash(ref_scales, mxquant128_scale_bytes);
  const int timing_bad = quant_end <= quant_begin || pack_end <= pack_begin;
  const int flags_bad = flags_before != 0 || flags_quant != 1 ||
      flags_pack != 1 || flags_after != 1;
  const int failed = canary_errors || compare_errors || timing_bad || flags_bad ||
      frm_before != 0 || frm_after != 0;

  printf("MXquant128 begin rows=128 cols=128 shape=FP32_no_GEMM "
         "blocks=512 K32=512 temp_block_bytes=8704 payload_bytes=8192 scale_bytes=512 "
         "input_init=outside warmup=0 runs=1 scope=quantize_row_mxfp4+"
         "ggml_ame_pack_mxfp4_blocks no_MMA no_matrix_CSR\n");
  printf("MXquant128 quant_cycles=%lu pack_cycles=%lu total_cycles=%lu "
         "quant_f32_bytes=%u packed_bytes=%u scale_count=%u "
         "payload_hash=%016lx scale_hash=%016lx ref_payload_hash=%016lx "
         "ref_scale_hash=%016lx "
         "exact=%u canary_errors=%u natural_nx=%u "
         "flags_before=%u flags_after_quant=%u flags_after_pack=%u flags_final=%u "
         "frm_before=%u frm_after=%u %s\n",
         quant_end - quant_begin, pack_end - pack_begin,
         pack_end - quant_begin,
         4 * mxquant128_shape.total_f32, mxquant128_shape.payload_bytes,
         mxquant128_shape.scale_bytes, payload_hash, scale_hash,
         ref_payload_hash, ref_scale_hash, compare_errors == 0 && !canary_errors,
         canary_errors, natural_nx, flags_before, flags_quant, flags_pack,
         flags_after, frm_before, frm_after, failed ? "FAIL" : "PASS");
  printf("MXquant128 timing_requires=cycle_accurate_RTL_or_hardware "
         "canonical_quant=quantize_row_mxfp4 canonical_ref=quantize_row_mxfp4_ref "
         "canonical_pack=ggml_ame_pack_mxfp4_blocks reference_timed=no\n");
  return failed;
}

static int run_mxpreflight128(void) {
  mxfp_quant_guarded = &mxfp_quant_source_storage;
  MxQuant128SourceGuard *guarded = mxfp_quant_guarded;
  float *source = guarded->source;
  for (unsigned i = 0; i < 16; ++i) {
    guarded->before[i] = 0x13579bdf;
    guarded->after[i] = 0x2468ace0;
    mxfp_quant_quantized_guard.before[i] = 0x13579bdf;
    mxfp_quant_quantized_guard.after[i] = 0x2468ace0;
    mxfp_quant_reference_guard.before[i] = 0x13579bdf;
    mxfp_quant_reference_guard.after[i] = 0x2468ace0;
    mxfp_quant_payload_guard.before[i] = 0x13579bdf;
    mxfp_quant_payload_guard.after[i] = 0x2468ace0;
    mxfp_quant_ref_payload_guard.before[i] = 0x13579bdf;
    mxfp_quant_ref_payload_guard.after[i] = 0x2468ace0;
    mxfp_quant_scales_guard.before[i] = 0x13579bdf;
    mxfp_quant_scales_guard.after[i] = 0x2468ace0;
    mxfp_quant_ref_scales_guard.before[i] = 0x13579bdf;
    mxfp_quant_ref_scales_guard.after[i] = 0x2468ace0;
  }
  uint8_t *scales = mxfp_quant_scales_guard.bytes;
  uint8_t *reference_scales = mxfp_quant_ref_scales_guard.bytes;
  block_mxfp4 *pack_blocks = mxfp_quant_reference_guard.blocks;
  uint8_t *unused_payload = mxfp_quant_ref_payload_guard.bytes;

  mxfp_quant_make_input(source);
  const unsigned natural_nx = mxquant128_natural_nx_blocks;
  for (unsigned i = 0; i < mxquant128_shape.total_f32; ++i) {
    union { float f; uint32_t u; } bits = { .f = source[i] };
    mxfp_quant_preimage[i] = bits.u;
  }
  for (unsigned i = 0; i < mxquant128_scale_bytes; ++i) scales[i] = 0x5a;
  uint32_t flags_before, flags_after, frm_before, frm_after;
  __asm__ volatile("frrm %0" : "=r"(frm_before));
  __asm__ volatile("frflags %0" : "=r"(flags_before) : : "memory");
  uint64_t begin, end;
  __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(begin) : : "memory");
  for (unsigned row = 0; row < mxquant128_shape.rows; ++row) {
    if (!ggml_mxfp4_prepare_supplied_scales(
            source + row * mxquant128_shape.columns,
            scales + row * mxquant128_shape.blocks_per_row,
            mxquant128_shape.columns)) {
      printf("MXpreflight128 canonical preflight FAIL row=%u\n", row);
      return 1;
    }
  }
  __asm__ volatile("fence rw, rw\n\trdcycle %0\n\tfrflags %1"
                   : "=r"(end), "=r"(flags_after) : : "memory");
  __asm__ volatile("frrm %0" : "=r"(frm_after));

  for (unsigned row = 0; row < mxquant128_shape.rows; ++row) {
    quantize_row_mxfp4_ref(source + row * mxquant128_shape.columns,
                           pack_blocks + row * mxquant128_shape.blocks_per_row,
                           mxquant128_shape.columns);
  }
  if (!ggml_ame_pack_mxfp4_blocks(unused_payload, reference_scales, pack_blocks,
                                  mxquant128_shape.total_blocks)) {
    printf("MXpreflight128 reference pack FAIL\n");
    return 1;
  }
  unsigned scale_mismatches = 0;
  for (unsigned i = 0; i < mxquant128_scale_bytes; ++i)
    scale_mismatches += scales[i] != reference_scales[i];
  unsigned canary_errors =
      mxfp_quant_check_input(source, mxfp_quant_preimage,
                             guarded->before, guarded->after);
  for (unsigned i = 0; i < 16; ++i) {
    canary_errors += mxfp_quant_quantized_guard.before[i] != 0x13579bdf ||
        mxfp_quant_quantized_guard.after[i] != 0x2468ace0;
    canary_errors += mxfp_quant_reference_guard.before[i] != 0x13579bdf ||
        mxfp_quant_reference_guard.after[i] != 0x2468ace0;
    canary_errors += mxfp_quant_payload_guard.before[i] != 0x13579bdf ||
        mxfp_quant_payload_guard.after[i] != 0x2468ace0;
    canary_errors += mxfp_quant_ref_payload_guard.before[i] != 0x13579bdf ||
        mxfp_quant_ref_payload_guard.after[i] != 0x2468ace0;
    canary_errors += mxfp_quant_scales_guard.before[i] != 0x13579bdf ||
        mxfp_quant_scales_guard.after[i] != 0x2468ace0;
    canary_errors += mxfp_quant_ref_scales_guard.before[i] != 0x13579bdf ||
        mxfp_quant_ref_scales_guard.after[i] != 0x2468ace0;
  }
  const uint64_t scale_hash = mxfp_quant_hash(scales, mxquant128_scale_bytes);
  const uint64_t ref_scale_hash =
      mxfp_quant_hash(reference_scales, mxquant128_scale_bytes);
  const int flags_bad = flags_before != 0 || flags_after != 1;
  const int failed = canary_errors || scale_mismatches || end <= begin ||
      flags_bad || frm_before != 0 || frm_after != 0;
  printf("MXpreflight128 begin rows=128 cols=128 shape=FP32_no_GEMM "
         "blocks=512 K32=512 scale_only=yes payload_quant=not_performed "
         "input_init=outside warmup=0 runs=1 "
         "scope=ggml_mxfp4_prepare_supplied_scales no_MMA no_matrix_CSR\n");
  printf("MXpreflight128 preflight_cycles=%lu total_cycles=%lu "
         "input_f32_bytes=%u scanned_K32_blocks=%u scale_bytes=%u "
         "scale_hash=%016lx ref_scale_hash=%016lx exact_scales=%u "
         "scale_mismatches=%u canary_errors=%u natural_nx=%u "
         "flags_before=%u flags_after=%u frm_before=%u frm_after=%u %s\n",
         end - begin, end - begin, 4 * mxquant128_shape.total_f32,
         mxquant128_shape.total_blocks, mxquant128_shape.scale_bytes,
         scale_hash, ref_scale_hash, scale_mismatches == 0, scale_mismatches,
         canary_errors, natural_nx, flags_before, flags_after, frm_before,
         frm_after, failed ? "FAIL" : "PASS");
  printf("MXpreflight128 timing_requires=cycle_accurate_RTL_or_hardware "
         "canonical_preflight=ggml_mxfp4_prepare_supplied_scales "
         "reference_timed=no full_quantization=no\n");
  return failed;
}
#endif

#if defined(MXFP_RMS_ONLY) && MXFP_RMS_ONLY
enum {
  mxrms512_k = 512, mxrms512_nb = 16, mxrms512_mn = 128, mxrms512_tk = 64,
  mxrms512_panel_blocks = mxrms512_mn * mxrms512_nb,
  mxrms512_warmup = 5, mxrms512_repeats = 10,
};

typedef struct {
  float norm[mxrms512_k + 32];
  float weighted[mxrms512_k + 32];
  block_mxfp4 blocks[mxrms512_nb + 2];
  uint8_t payload[mxrms512_panel_blocks * 16 + 128];
  uint8_t scales[mxrms512_panel_blocks + 128];
} MxRms512Output;

static float mxrms512_raw[mxrms512_k + 32] __attribute__((aligned(64)));
static float mxrms512_gamma[mxrms512_k + 32] __attribute__((aligned(64)));
static uint8_t mxrms512_raw_before[sizeof(mxrms512_raw)];
static uint8_t mxrms512_gamma_before[sizeof(mxrms512_gamma)];
static MxRms512Output mxrms512_actual __attribute__((aligned(64)));
static MxRms512Output mxrms512_reference __attribute__((aligned(64)));

// These are the SAME ordered RMS, F32 scale and gamma primitives used by the
// canonical helper/ordinary CPU path. The full helper's GEMM geometry rejects
// rows=1; this control does not fabricate 64/128 input rows to bypass it.
static __attribute__((noinline)) void mxfp_rms512_normalize(
    MxRms512Output *out, float eps) {
  const float r = ggml_vec_rms_norm_scale_f32(
      mxrms512_k, mxrms512_raw + 16, eps);
  ggml_vec_cpy_f32(mxrms512_k, out->norm + 16, mxrms512_raw + 16);
  ggml_vec_scale_f32(mxrms512_k, out->norm + 16, r);
  ggml_vec_mul_f32(mxrms512_k, out->weighted + 16, out->norm + 16,
                    mxrms512_gamma + 16);
}

// Memory-prepared endpoint only: real quantizer batch8 x2, then unchanged AME
// reorder into canonical row0/MN128/K64 slots. No matrix CSR, TR/SF load or MMA.
static __attribute__((noinline)) int mxfp_rms512_chain(
    MxRms512Output *out, float eps) {
  mxfp_rms512_normalize(out, eps);
  for (int kb = 0; kb < mxrms512_nb; kb += 8)
    quantize_row_mxfp4(out->weighted + 16 + kb * 32, out->blocks + 1 + kb, 256);
  for (int kb = 0; kb < mxrms512_nb; kb += 2) {
    const size_t index = ggml_ame_mx_packed_block_index(
        0, kb, mxrms512_nb, mxrms512_tk, mxrms512_mn);
    if (!ggml_ame_pack_mxfp4_blocks(out->payload + 64 + index * 16,
                                    out->scales + 64 + index,
                                    out->blocks + 1 + kb, 2)) return 1;
  }
  return 0;
}

static uint32_t mxfp_rms512_fcsr(void) {
  uint32_t fcsr;
  __asm__ volatile("frcsr %0" : "=r"(fcsr) : : "memory");
  return fcsr;
}

static void mxfp_rms512_seed(uint32_t fcsr) {
  __asm__ volatile("fscsr %0" : : "r"(fcsr) : "memory");
}

static float mxfp_rms512_input(unsigned pattern) {
  memset(mxrms512_raw, 0x93, sizeof(mxrms512_raw));
  memset(mxrms512_gamma, 0x93, sizeof(mxrms512_gamma));
  for (int k = 0; k < mxrms512_k; ++k) {
    mxrms512_raw[16 + k] = pattern ? 1.0f :
        (float)((k * 13) % 127 - 63) * 0x1p-5f;
    mxrms512_gamma[16 + k] = pattern ? 1.0f :
        (float)(32 + (k * 7) % 65) * 0x1p-6f;
  }
  // Original row512 natural UF/NX and two-rounding counterexample fixtures.
  if (pattern) mxrms512_gamma[16 + 7 * 32 + 3] = 0x1p-149f;
  if (pattern == 2) mxrms512_raw[16 + 7 * 32 + 3] = 2.0f;
  memcpy(mxrms512_raw_before, mxrms512_raw, sizeof(mxrms512_raw));
  memcpy(mxrms512_gamma_before, mxrms512_gamma, sizeof(mxrms512_gamma));
  return pattern == 0 ? 0x1p-10f : pattern == 2 ? 0x1.7f4p+1f : 3.0f;
}

// Independent scalar quant reference, full-row rather than two batch8 calls,
// and literal row0 slot oracle. RMS math and the packer remain canonical.
static int mxfp_rms512_reference(float eps, uint32_t seed, uint32_t *final) {
  MxRms512Output *out = &mxrms512_reference;
  memset(out, 0xa5, sizeof(*out));
  mxfp_rms512_seed(seed);
  mxfp_rms512_normalize(out, eps);
  quantize_row_mxfp4_ref(out->weighted + 16, out->blocks + 1, mxrms512_k);
  for (int kb = 0; kb < mxrms512_nb; ++kb) {
    const size_t index = (size_t)(kb / 2) * mxrms512_mn * 2 + kb % 2;
    if (!ggml_ame_pack_mxfp4_blocks(out->payload + 64 + index * 16,
                                    out->scales + 64 + index,
                                    out->blocks + 1 + kb, 1)) return 1;
  }
  *final = mxfp_rms512_fcsr();
  return 0;
}

static unsigned mxfp_rms512_verify(uint32_t actual_fcsr, uint32_t expected_fcsr) {
  // Whole-object compare includes all output canaries and untouched panel
  // rows, not only logical row0. References retain the same poison bytes.
  unsigned errors = 0;
  for (unsigned i = 0; i < 16; ++i) {
    uint32_t bits[4];
    memcpy(bits + 0, mxrms512_actual.norm + i, 4);
    memcpy(bits + 1, mxrms512_actual.norm + 16 + mxrms512_k + i, 4);
    memcpy(bits + 2, mxrms512_actual.weighted + i, 4);
    memcpy(bits + 3, mxrms512_actual.weighted + 16 + mxrms512_k + i, 4);
    for (unsigned j = 0; j < 4; ++j) errors += bits[j] != 0xa5a5a5a5u;
  }
  const uint8_t *block_bytes = (const uint8_t *)mxrms512_actual.blocks;
  for (unsigned i = 0; i < sizeof(block_mxfp4); ++i)
    errors += block_bytes[i] != 0xa5 ||
        block_bytes[(mxrms512_nb + 1) * sizeof(block_mxfp4) + i] != 0xa5;
  for (unsigned i = 0; i < 64; ++i) {
    errors += mxrms512_actual.payload[i] != 0xa5 ||
        mxrms512_actual.payload[64 + mxrms512_panel_blocks * 16 + i] != 0xa5;
    errors += mxrms512_actual.scales[i] != 0xa5 ||
        mxrms512_actual.scales[64 + mxrms512_panel_blocks + i] != 0xa5;
  }
  return errors + (memcmp(&mxrms512_actual, &mxrms512_reference,
                  sizeof(mxrms512_actual)) != 0) +
      (memcmp(mxrms512_raw, mxrms512_raw_before, sizeof(mxrms512_raw)) != 0) +
      (memcmp(mxrms512_gamma, mxrms512_gamma_before, sizeof(mxrms512_gamma)) != 0) +
      (actual_fcsr != expected_fcsr);
}

// Observation only, not a new quantizer or an allocated instruction contract.
// E/F are the raw amax FP32 exponent/fraction. Exact powers have F==0.
static int mxfp_rms512_candidate_guard(uint32_t amax) {
  const unsigned E = (amax >> 23) & 255, F = amax & 0x7fffff;
  return amax == 0 || (E >= 2 && E <= 253 &&
      (F == 0 || (F >= 0x400 && F <= 0x7ffc00)));
}

static unsigned mxfp_rms512_diagnostics(float eps) {
  const uint32_t saved = mxfp_rms512_fcsr();
  MxRms512Output *out = &mxrms512_reference;
  // Replay only original RMS/gamma outside every timer to expose their real
  // prefix flags. Never seed NX or change pattern0 to increase acceptance.
  mxfp_rms512_seed(0);
  const float r = ggml_vec_rms_norm_scale_f32(
      mxrms512_k, mxrms512_raw + 16, eps);
  ggml_vec_cpy_f32(mxrms512_k, out->norm + 16, mxrms512_raw + 16);
  ggml_vec_scale_f32(mxrms512_k, out->norm + 16, r);
  const uint32_t norm_fcsr = mxfp_rms512_fcsr();
  ggml_vec_mul_f32(mxrms512_k, out->weighted + 16, out->norm + 16,
                    mxrms512_gamma + 16);
  const uint32_t gamma_fcsr = mxfp_rms512_fcsr();
  unsigned errors = memcmp(out->weighted, mxrms512_actual.weighted,
                           sizeof(out->weighted)) != 0;
  unsigned accepted = 0;
  for (unsigned kb = 0; kb < mxrms512_nb; ++kb) {
    uint32_t amax = 0;
    int finite = 1;
    for (unsigned lane = 0; lane < 32; ++lane) {
      uint32_t bits;
      memcpy(&bits, mxrms512_actual.weighted + 16 + kb * 32 + lane, 4);
      bits &= 0x7fffffffu;
      finite &= bits < 0x7f800000u;
      if (bits > amax) amax = bits;
    }
    const int admit = finite && mxfp_rms512_candidate_guard(amax);
    accepted += admit;
    const unsigned candidate_e = amax == 0 ? 0 : ((amax >> 23) & 255) - 2;
    errors += admit && mxrms512_actual.blocks[1 + kb].e != candidate_e;
    printf("MXrms512 pattern0 block=%u amax_rawbits=%08x E=%u F=%06x "
           "canonical_e=%u candidate_e=%u finite32=%u candidate_guard=%u\n",
           kb, amax, (amax >> 23) & 255, amax & 0x7fffff,
           mxrms512_actual.blocks[1 + kb].e, candidate_e, finite, admit);
  }
  printf("MXrms512 pattern0 candidate_accepted=%u/16 "
         "guard=finite32_zero_or_E2..253_Fzero_or_000400..7ffc00 "
         "rms_prefix_fflags=%u gamma_prefix_fflags=%u "
         "row_dynamic_candidate=%u NX0=whole_row_CPU_fallback "
         "frm=%u diagnostic=outside_timer dynamic_instruction=not_issued\n",
         accepted, norm_fcsr & 31, gamma_fcsr & 31,
         accepted == 16 && (gamma_fcsr & 1), gamma_fcsr >> 5);
  mxfp_rms512_seed(saved);
  return errors;
}

static int run_mxrms512(void) {
  const uint32_t saved_fcsr = mxfp_rms512_fcsr();
  mxfp_rms512_seed(0);
  const float eps = mxfp_rms512_input(0);
  memset(&mxrms512_actual, 0xa5, sizeof(mxrms512_actual));
  mxfp_rms512_seed(0);
  uint64_t begin, end;
  uint32_t actual_fcsr, expected_fcsr;
  // This is the first FULL chain invocation, before any oracle or warmup.
  __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(begin) : : "memory");
  const int first_status = mxfp_rms512_chain(&mxrms512_actual, eps);
  __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(end) : : "memory");
  actual_fcsr = mxfp_rms512_fcsr();
  unsigned errors = first_status + (end <= begin);
  errors += mxfp_rms512_reference(eps, 0, &expected_fcsr);
  errors += mxfp_rms512_verify(actual_fcsr, expected_fcsr);
  const uint64_t payload_hash = mxfp_quant_hash(
      mxrms512_actual.payload + 64, mxrms512_panel_blocks * 16);
  const uint64_t scale_hash = mxfp_quant_hash(
      mxrms512_actual.scales + 64, mxrms512_panel_blocks);
  printf("MXrms512 one_shot cycles=%lu warmup=0 calls=1 frm=0 incoming=0 "
         "final_fcsr=%u ref_fcsr=%u errors=%u first_chain=yes\n",
         end - begin, actual_fcsr, expected_fcsr, errors);
  errors += mxfp_rms512_diagnostics(eps);

  uint64_t samples[mxrms512_repeats];
  for (unsigned i = 0; i < mxrms512_warmup + mxrms512_repeats; ++i) {
    memset(&mxrms512_actual, 0xa5, sizeof(mxrms512_actual));
    mxfp_rms512_seed(0);
    __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(begin) : : "memory");
    const int status = mxfp_rms512_chain(&mxrms512_actual, eps);
    __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(end) : : "memory");
    actual_fcsr = mxfp_rms512_fcsr();
    errors += status + (end <= begin) +
        mxfp_rms512_verify(actual_fcsr, expected_fcsr);
    if (i >= mxrms512_warmup) samples[i - mxrms512_warmup] = end - begin;
  }
  // Print only after all timed repeats. Post-oracle/W5 state is explicitly
  // distinct from one-shot and from historical no-warmup measurements.
  for (unsigned i = 0; i < mxrms512_repeats; ++i)
    printf("MXrms512 repeat=%u cycles=%lu warmup=5 repeats=10 "
           "frm=0 incoming=0 cache=post_oracle_not_cold_DRAM\n", i, samples[i]);

  unsigned cases = 0, flag_errors = 0;
  for (unsigned pattern = 0; pattern < 3; ++pattern) {
    mxfp_rms512_seed(0);
    const float test_eps = mxfp_rms512_input(pattern);
    for (unsigned frm = 0; frm < 4; ++frm) {
      for (unsigned flags = 0; flags < 32; ++flags) {
        const uint32_t initial = (frm << 5) | flags;
        errors += mxfp_rms512_reference(test_eps, initial, &expected_fcsr);
        // Check real natural UF/NX after the complete reference chain.
        flag_errors += pattern == 1 && (expected_fcsr & 3) != 3;
        memset(&mxrms512_actual, 0xa5, sizeof(mxrms512_actual));
        mxfp_rms512_seed(initial);
        const uint32_t seeded = mxfp_rms512_fcsr();
        errors += mxfp_rms512_chain(&mxrms512_actual, test_eps);
        actual_fcsr = mxfp_rms512_fcsr();
        flag_errors += seeded != initial || (actual_fcsr >> 5) != frm ||
            (actual_fcsr & flags) != flags;
        errors += mxfp_rms512_verify(actual_fcsr, expected_fcsr);
        ++cases;
      }
    }
  }
  errors += flag_errors;
  mxfp_rms512_seed(saved_fcsr);
  printf("MXrms512 control rows=1 K=512 logical_payload=256 logical_sf=16 "
         "tile_mn=128 tile_k=64 padded_payload=32768 padded_sf=2048 "
         "quant_calls=2 batch_blocks=8 payload_hash=%016lx sf_hash=%016lx "
         "gate_cases=%u patterns=3 frm=4 sticky=32 flag_errors=%u errors=%u %s\n",
         payload_hash, scale_hash, cases, flag_errors, errors,
         errors ? "FAIL" : "PASS");
  printf("MXrms512 endpoint=memory_prepared TR_SF_load=not_included MMA=none "
         "matrix_CSR=none quant=canonical_RVV ref=canonical_scalar "
         "RMS=canonical_vec full_GEMM_helper=not_called_single_row "
         "oracle_poison_seed_print=outside_timer timing_requires=RTL_or_hardware "
         "historical_R35_speedup=not_comparable\n");
  return errors != 0;
}
#endif

#if defined(MXFP_RMS_DYNAMIC) && MXFP_RMS_DYNAMIC
#ifndef MXFP_RMS_DYNAMIC_ATTEST
#define MXFP_RMS_DYNAMIC_ATTEST "none"
#endif
enum { mxrmsdyn_rows = 8, mxrmsdyn_payload = 256, mxrmsdyn_sf_backing = 64 };
typedef struct {
  uint8_t payload[mxrmsdyn_payload + 128];
  uint8_t scales[mxrmsdyn_sf_backing + 128];
  block_mxfp4 blocks[mxrms512_nb + 2];
} MxRmsDynPacked;
typedef struct {
  uint32_t prefix_fcsr, final_fcsr, status, rejected, old_flags, retry_flags;
  unsigned dynamic_calls, cpu_rows, ready;
} MxRmsDynResult;
static MxRmsDynPacked mxrmsdyn_packed __attribute__((aligned(64)));
static int mxrmsdyn_attested;

static void mxfp_rmsdyn_poison(void) {
  memset(&mxrms512_actual, 0xa5, sizeof(mxrms512_actual));
  memset(&mxrmsdyn_packed, 0xa5, sizeof(mxrmsdyn_packed));
  // SF loads may fetch the containing 64B beat. All unused bytes are real
  // readable backing, initialized to the existing neutral scale padding.
  memset(mxrmsdyn_packed.scales + 64, 127, mxrmsdyn_sf_backing);
}

static int mxfp_rmsdyn_pack(const float *weighted, MxRmsDynPacked *out) {
  for (int kb = 0; kb < mxrms512_nb; kb += 8)
    quantize_row_mxfp4(weighted + kb * 32, out->blocks + 1 + kb, 256);
  return !ggml_ame_pack_mxfp4_blocks(out->payload + 64, out->scales + 64,
                                    out->blocks + 1, mxrms512_nb);
}

// Full-row transaction. This caller takes already-normalized input as well
// as the genuine RMS-chain input. NX is read, never manufactured here.
static MxRmsDynResult mxfp_rmsdyn_ready(const float *weighted,
                                     unsigned side, int request_dynamic) {
  MxRmsDynResult r = {0};
  r.prefix_fcsr = mxfp_rms512_fcsr();
  if (side > 1 || (request_dynamic && !mxrmsdyn_attested)) {
    r.status = 32; // caller refusal, not a matrix/scalar exception
    r.final_fcsr = r.prefix_fcsr;
    return r;
  }
  MxRmsPanelConfig config = {
    {mxrmsdyn_rows, 64, 1, 32, 2, 0, 0}, 1, side, 0
  };
  if (request_dynamic && (r.prefix_fcsr & 1)) {
    config.dynamic = 1;
    config.panel.payload_stride = 256;
    const uint64_t status = matrix_test_mxrms_panel(
        (const uint8_t *)weighted, NULL, NULL, NULL, NULL, &config);
    r.old_flags |= status >> 32;
    r.status = status & 31;
    ++r.dynamic_calls;
    if (!r.status) {
      r.ready = 1;
      r.final_fcsr = mxfp_rms512_fcsr();
      return r;
    }
    r.rejected = r.status;
    // Only isolated unsupported NV is the retry contract. Unexpected status
    // is terminal; no consumer may wait on a failed generation.
    if (r.status != 16) {
      r.final_fcsr = mxfp_rms512_fcsr();
      return r;
    }
  }
  // NX0 or acquired rejection: quantize/pack/reload ALL16 blocks. No partial
  // reuse of a provisional TR/SF generation and no scalar NV bridge.
  if (mxfp_rmsdyn_pack(weighted, &mxrmsdyn_packed)) {
    r.status = 32;
    r.final_fcsr = mxfp_rms512_fcsr();
    return r;
  }
  ++r.cpu_rows;
  config.dynamic = 0;
  config.panel.payload_stride = 32;
  const uint64_t status = matrix_test_mxrms_panel(
      mxrmsdyn_packed.payload + 64, NULL, mxrmsdyn_packed.scales + 64,
      NULL, NULL, &config);
  if (r.dynamic_calls) r.retry_flags = status >> 32;
  else r.old_flags = status >> 32;
  r.status = status & 31;
  r.ready = !r.status;
  r.final_fcsr = mxfp_rms512_fcsr();
  return r;
}

static MxRmsDynResult mxfp_rmsdyn_chain(float eps, unsigned side, int dynamic) {
  mxfp_rms512_normalize(&mxrms512_actual, eps);
  return mxfp_rmsdyn_ready(mxrms512_actual.weighted + 16, side, dynamic);
}

// Validation only, outside all prepare timers. Build an actual quantized
// identity using the canonical reference and packer, not literal FP4 codes.
static float mxrmsdyn_basis_input[64 * 64] __attribute__((aligned(64)));
static block_mxfp4 mxrmsdyn_basis_blocks[128] __attribute__((aligned(64)));
static uint8_t mxrmsdyn_basis_payload[2048] __attribute__((aligned(64)));
static uint8_t mxrmsdyn_basis_sf[128] __attribute__((aligned(64)));
static MxRmsDynPacked mxrmsdyn_ref_packed __attribute__((aligned(64)));
static float mxrmsdyn_dequant[512] __attribute__((aligned(64)));
static uint32_t mxrmsdyn_result[512 + 32] __attribute__((aligned(64)));
static float mxrmsdyn_caller_input[512 + 32] __attribute__((aligned(64)));

static int mxfp_rmsdyn_basis(void) {
  const uint32_t saved = mxfp_rms512_fcsr();
  for (unsigned m = 0; m < 64; ++m)
    for (unsigned k = 0; k < 64; ++k)
      mxrmsdyn_basis_input[m * 64 + k] = m == k ? 1.0f : 0.0f;
  mxfp_rms512_seed(0);
  quantize_row_mxfp4_ref(mxrmsdyn_basis_input, mxrmsdyn_basis_blocks, 4096);
  const int error = !ggml_ame_pack_mxfp4_blocks(
      mxrmsdyn_basis_payload, mxrmsdyn_basis_sf, mxrmsdyn_basis_blocks, 128);
  mxfp_rms512_seed(saved);
  return error;
}

static unsigned mxfp_rmsdyn_verify(const MxRmsDynResult *r, unsigned side,
                                  unsigned dynamic, float eps, uint32_t seed,
                                  int normalized_caller, int rejected_block,
                                  uint32_t expected_old_flags) {
  const uint32_t actual_fcsr = mxfp_rms512_fcsr();
  unsigned errors = !r->ready || r->status || actual_fcsr != r->final_fcsr;
  memset(&mxrms512_reference, 0xa5, sizeof(mxrms512_reference));
  memset(&mxrmsdyn_ref_packed, 0xa5, sizeof(mxrmsdyn_ref_packed));
  memset(mxrmsdyn_ref_packed.scales + 64, 127, mxrmsdyn_sf_backing);
  mxfp_rms512_seed(seed);
  if (normalized_caller) {
    memcpy(mxrms512_reference.weighted, mxrmsdyn_caller_input,
           sizeof(mxrms512_reference.weighted));
  } else {
    mxfp_rms512_normalize(&mxrms512_reference, eps);
  }
  const uint32_t reference_prefix = mxfp_rms512_fcsr();
  quantize_row_mxfp4_ref(mxrms512_reference.weighted + 16,
                        mxrmsdyn_ref_packed.blocks + 1, 512);
  const uint32_t reference_final = mxfp_rms512_fcsr();
  errors += !ggml_ame_pack_mxfp4_blocks(
      mxrmsdyn_ref_packed.payload + 64, mxrmsdyn_ref_packed.scales + 64,
      mxrmsdyn_ref_packed.blocks + 1, 16);
  dequantize_row_mxfp4(mxrmsdyn_ref_packed.blocks + 1, mxrmsdyn_dequant, 512);
  const unsigned expected_dynamic = dynamic && (reference_prefix & 1);
  const unsigned expected_reject = expected_dynamic && rejected_block >= 0;
  const unsigned expected_cpu = !expected_dynamic || expected_reject;
  errors += r->prefix_fcsr != reference_prefix || r->final_fcsr != reference_final;
  errors += r->dynamic_calls != expected_dynamic || r->cpu_rows != expected_cpu;
  errors += r->rejected != (expected_reject ? 16u : 0u);
  errors += r->old_flags != expected_old_flags;
  errors += r->retry_flags != (expected_reject ? 16u : 0u);
  errors += memcmp(&mxrms512_actual, &mxrms512_reference,
                    sizeof(mxrms512_actual)) != 0;
  errors += memcmp(mxrms512_raw, mxrms512_raw_before, sizeof(mxrms512_raw)) != 0;
  errors += memcmp(mxrms512_gamma, mxrms512_gamma_before,
                    sizeof(mxrms512_gamma)) != 0;
  if (expected_cpu) {
    // Includes SF readable padding, both block sentinels and payload guards.
    errors += memcmp(&mxrmsdyn_packed, &mxrmsdyn_ref_packed,
                      sizeof(mxrmsdyn_packed)) != 0;
  } else {
    // A successful hardware producer must not have done hidden CPU quant.
    for (unsigned i = 0; i < sizeof(mxrmsdyn_packed.payload); ++i)
      errors += mxrmsdyn_packed.payload[i] != 0xa5;
    for (unsigned i = 0; i < sizeof(mxrmsdyn_packed.scales); ++i)
      errors += mxrmsdyn_packed.scales[i] !=
          (i >= 64 && i < 64 + mxrmsdyn_sf_backing ? 127 : 0xa5);
    const uint8_t *blocks = (const uint8_t *)mxrmsdyn_packed.blocks;
    for (unsigned i = 0; i < sizeof(mxrmsdyn_packed.blocks); ++i)
      errors += blocks[i] != 0xa5;
  }
  // Restore the actual post-producer state after the independent oracle.
  // Matrix status is NEVER copied into scalar fflags, even on retry NV.
  mxfp_rms512_seed(actual_fcsr);
  uint64_t matrix_frm_before, matrix_frm_after;
  __asm__ volatile("csrr %0, 0x809" : "=r"(matrix_frm_before) : : "memory");
  memset(mxrmsdyn_result, 0xa5, sizeof(mxrmsdyn_result));
  if (r->ready && !r->status) {
    const MxRmsPanelConfig consume = {
      {8, 64, 1, 32, 2, side ? 8 * 4 : 64 * 4, 0}, 2, side, 0
    };
    // For A: load ONLY B identity. For B: load ONLY A identity. The new
    // entry joins the original MMACC/MSC owner without replacing tested TR/SF.
    const uint64_t flags = matrix_test_mxrms_panel(
        mxrmsdyn_basis_payload, mxrmsdyn_basis_payload,
        mxrmsdyn_basis_sf, mxrmsdyn_basis_sf, mxrmsdyn_result + 16, &consume);
    errors += flags != 0;
    for (unsigned row = 0; row < (side ? 64u : 8u); ++row) {
      for (unsigned col = 0; col < (side ? 8u : 64u); ++col) {
        const unsigned index = row * (side ? 8 : 64) + col;
        const unsigned input_index = side ? col * 64 + row : index;
        uint32_t expected;
        memcpy(&expected, mxrmsdyn_dequant + input_index, 4);
        errors += mxrmsdyn_result[16 + index] != expected;
      }
    }
  }
  for (unsigned i = 0; i < 16; ++i)
    errors += mxrmsdyn_result[i] != 0xa5a5a5a5u ||
        mxrmsdyn_result[16 + 512 + i] != 0xa5a5a5a5u;
  __asm__ volatile("csrr %0, 0x809" : "=r"(matrix_frm_after) : : "memory");
  errors += matrix_frm_before != matrix_frm_after;
  errors += mxfp_rms512_fcsr() != actual_fcsr;
  return errors;
}

static void mxfp_rmsdyn_diagnostics(uint32_t prefix) {
  unsigned accepted = 0;
  // Called after the pattern0 one-shot oracle, never inside a sample timer.
  for (unsigned kb = 0; kb < 16; ++kb) {
    uint32_t amax = 0;
    unsigned finite = 1;
    for (unsigned lane = 0; lane < 32; ++lane) {
      uint32_t bits;
      memcpy(&bits, mxrms512_actual.weighted + 16 + kb * 32 + lane, 4);
      bits &= 0x7fffffff;
      finite &= bits < 0x7f800000;
      if (bits > amax) amax = bits;
    }
    const unsigned admit = finite && mxfp_rms512_candidate_guard(amax);
    accepted += admit;
    printf("MXrmsdyn512 pattern0 block=%u amax_rawbits=%08x E=%u F=%06x "
           "ref_e=%u candidate_e=%u finite32=%u accepted=%u\n", kb, amax,
           (amax >> 23) & 255, amax & 0x7fffff,
           mxrmsdyn_ref_packed.blocks[kb + 1].e,
           amax ? ((amax >> 23) & 255) - 2 : 0, finite, admit);
  }
  printf("MXrmsdyn512 pattern0 accepted=%u/16 prefix_fcsr=%u "
         "NX_not_manufactured=1 SF_BYTES_NOT_DIRECTLY_OBSERVED\n",
         accepted, prefix);
}

static int run_mxrmsdyn512(void) {
  // Explicit operator assertion about the actual config/model. The geometry
  // check below proves capacity only, never feature presence.
  if (strcmp(MXFP_RMS_DYNAMIC_ATTEST,
             "DefaultMatrix2TMxfpDynamicQuantConfig") != 0) {
    printf("MXrmsdyn512 REFUSED explicit_config_model_attestation_required "
           "attest=%s geometry_is_not_feature_proof\n", MXFP_RMS_DYNAMIC_ATTEST);
    return 1;
  }
  uint32_t saved_fcsr = mxfp_rms512_fcsr();
  __asm__ volatile("csrs mstatus, %0" : : "r"(0x02002200ul) : "memory");
  uint64_t tlenb, trlenb, alenb, saved_mfflags, saved_mfrm;
  __asm__ volatile("msyncregreset sync1\n\tmrelease sync1\n\t"
                   "macquire sync1, %5\n\tmfence\n\t"
                   "csrr %0, 0xcc1\n\tcsrr %1, 0xcc2\n\tcsrr %2, 0xcc3\n\t"
                   "csrr %3, 0x808\n\tcsrr %4, 0x809"
      : "=r"(tlenb), "=r"(trlenb), "=r"(alenb),
        "=r"(saved_mfflags), "=r"(saved_mfrm) : "r"(1ul) : "memory");
  if (tlenb != 8192 || trlenb != 64 || alenb != 65536) {
    printf("MXrmsdyn512 REFUSED capacity=%lu/%lu/%lu expected=8192/64/65536\n",
           tlenb, trlenb, alenb);
    return 1;
  }
  __asm__ volatile("csrw 0x808, zero" : : : "memory");
  mxrmsdyn_attested = 1;
  mxfp_rms512_seed(0);
  const float eps = mxfp_rms512_input(0);
  unsigned errors = 0, samples_checked = 0;
  uint32_t last_prefix = 0;
  // One full row512, reinterpreted as8xK64 only AFTER the same ordered RMS
  // and two F32 roundings. Both endpoints include acquired TR/SF readiness.
  for (unsigned side = 0; side < 2; ++side) {
    for (unsigned dynamic = 0; dynamic < 2; ++dynamic) {
      mxfp_rmsdyn_poison();
      mxfp_rms512_seed(0);
      uint64_t begin, end;
      __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(begin) : : "memory");
      const MxRmsDynResult r = mxfp_rmsdyn_chain(eps, side, dynamic);
      __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(end) : : "memory");
      errors += !r.ready || r.status || end <= begin;
      // First CPU path already ran before identity/oracle construction.
      // Later first-per-path samples are ordered, not equal cold-DRAM runs.
      if (side == 0 && dynamic == 0) errors += mxfp_rmsdyn_basis();
      errors += mxfp_rmsdyn_verify(&r, side, dynamic, eps, 0, 0, -1, 0);
      last_prefix = r.prefix_fcsr;
      ++samples_checked;
      printf("MXrmsdyn512 one_shot side=%u dynamic=%u cycles=%lu "
             "cold_DRAM=not_claimed order=CPU_A_dynamic_A_CPU_B_dynamic_B "
             "prefix_fcsr=%u final_fcsr=%u status=%u rejected=%u old_flags=%u "
             "retry_flags=%u "
             "dynamic_calls=%u CPU_rows=%u ready=%u\n", side, dynamic, end - begin,
             r.prefix_fcsr, r.final_fcsr, r.status, r.rejected, r.old_flags,
             r.retry_flags,
             r.dynamic_calls, r.cpu_rows, r.ready);
    }
  }
  mxfp_rmsdyn_diagnostics(last_prefix);

  uint64_t samples[2][2][mxrms512_repeats];
  MxRmsDynResult sample_results[2][2][mxrms512_repeats];
  for (unsigned round = 0; round < mxrms512_warmup + mxrms512_repeats; ++round) {
    for (unsigned side = 0; side < 2; ++side) {
      for (unsigned slot = 0; slot < 2; ++slot) {
        const unsigned dynamic = (round + slot) & 1;
        mxfp_rmsdyn_poison();
        mxfp_rms512_seed(0);
        uint64_t begin, end;
        __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(begin) : : "memory");
        const MxRmsDynResult r = mxfp_rmsdyn_chain(eps, side, dynamic);
        __asm__ volatile("fence rw, rw\n\trdcycle %0" : "=r"(end) : : "memory");
        errors += !r.ready || r.status || end <= begin;
        errors += mxfp_rmsdyn_verify(&r, side, dynamic, eps, 0, 0, -1, 0);
        if (round >= mxrms512_warmup) {
          samples[side][dynamic][round - mxrms512_warmup] = end - begin;
          sample_results[side][dynamic][round - mxrms512_warmup] = r;
        }
        ++samples_checked;
      }
    }
  }
  for (unsigned side = 0; side < 2; ++side)
    for (unsigned dynamic = 0; dynamic < 2; ++dynamic)
      for (unsigned i = 0; i < mxrms512_repeats; ++i) {
        const MxRmsDynResult *r = &sample_results[side][dynamic][i];
        printf("MXrmsdyn512 repeat=%u side=%u dynamic=%u cycles=%lu W=5 R=10 "
               "frm=0 incoming=0 order=paired_rotation endpoint=TR_SF_ready "
               "prefix_fcsr=%u final_fcsr=%u dynamic_calls=%u CPU_rows=%u "
               "status=%u rejected=%u ready=%u "
               "cache=post_oracle_not_cold_DRAM\n",
               i, side, dynamic, samples[side][dynamic][i],
               r->prefix_fcsr, r->final_fcsr, r->dynamic_calls, r->cpu_rows,
               r->status, r->rejected, r->ready);
      }

  // True RMS/gamma flags: four frm values and all32 initial sticky states.
  // No fcsr write between normalize and ready. Old matrix NV is separately
  // seeded OUTSIDE the chain to prove it does not masquerade as owned NV.
  unsigned flag_cases = 0, caller_cases = 0;
  for (unsigned frm = 0; frm < 4; ++frm) {
    for (unsigned sticky = 0; sticky < 32; ++sticky) {
      const uint32_t seed = (frm << 5) | sticky;
      for (unsigned side = 0; side < 2; ++side) {
        for (unsigned dynamic = 0; dynamic < 2; ++dynamic) {
          mxfp_rmsdyn_poison();
          __asm__ volatile("csrw 0x808, %0\n\tcsrw 0x809, %1"
              : : "r"(16ul), "r"(3ul) : "memory");
          mxfp_rms512_seed(seed);
          const MxRmsDynResult r = mxfp_rmsdyn_chain(eps, side, dynamic);
          errors += mxfp_rmsdyn_verify(&r, side, dynamic, eps, seed, 0, -1, 16);
          ++flag_cases;
        }
      }
    }
  }
  // Independent normalized-input CALLER gates, not RMS-chain performance.
  // NX0 is established in TEST SETUP, never by clearing a genuine prefix.
  const int reject_blocks[] = {-1, 0, 7, 15};
  for (unsigned fixture = 0; fixture < 4; ++fixture) {
    for (unsigned side = 0; side < 2; ++side) {
      mxfp_rmsdyn_poison();
      float *weighted = mxrms512_actual.weighted + 16;
      for (unsigned k = 0; k < 512; ++k) weighted[k] = fixture ? 0.0f : 1.0f;
      if (fixture) {
        // Safe finite normal E127 F1, sole max of one K32; others zero.
        const uint32_t bits = 0x3f800001u;
        memcpy(weighted + reject_blocks[fixture] * 32, &bits, 4);
      }
      memcpy(mxrmsdyn_caller_input, mxrms512_actual.weighted,
             sizeof(mxrmsdyn_caller_input));
      const uint32_t seed = fixture ? 1 : 0;
      __asm__ volatile("csrw 0x808, %0" : : "r"(16ul) : "memory");
      mxfp_rms512_seed(seed);
      const MxRmsDynResult r = mxfp_rmsdyn_ready(weighted, side, 1);
      const unsigned before = errors;
      errors += mxfp_rmsdyn_verify(&r, side, 1, eps, seed, 1,
                                    reject_blocks[fixture], 16);
      printf("MXrmsdyn512 caller_only normalized_input=1 RMS_chain=not_run "
             "performance_sample=0 side=%u reject_block=%d test_seed=%u "
             "prefix_fcsr=%u final_fcsr=%u status=%u rejected=%u "
             "old_flags=%u retry_flags=%u dynamic_calls=%u CPU_rows=%u "
             "ready=%u errors=%u\n", side, reject_blocks[fixture], seed,
             r.prefix_fcsr, r.final_fcsr, r.status, r.rejected, r.old_flags,
             r.retry_flags, r.dynamic_calls, r.cpu_rows, r.ready, errors - before);
      ++caller_cases;
    }
  }
  mxrmsdyn_attested = 0;
  // All work was acquired. Restore independent scalar and matrix flag/round
  // domains; TR/acc/tile contents are intentionally clobbered by this app.
  __asm__ volatile("csrw 0x808, %0\n\tcsrw 0x809, %1"
      : : "r"(saved_mfflags), "r"(saved_mfrm) : "memory");
  mxfp_rms512_seed(saved_fcsr);
  printf("MXrmsdyn512 endpoint=TR_SF_ready source=one_row512_as_8xK64 "
         "SF_BYTES_NOT_DIRECTLY_OBSERVED historical_R35_speedup=not_comparable "
         "samples_checked=%u flag_cases=%u caller_cases=%u "
         "consumer=identity_MMA_MSC_512_outputs_each input=original_pattern0 "
         "oracle_poison_seed_print_MMA=outside_timer errors=%u %s\n",
         samples_checked, flag_cases, caller_cases, errors, errors ? "FAIL" : "PASS");
  return errors != 0;
}
#endif

static void print_failure_samples(const char *mode) {
#if !defined(MXFP_QUANT_ONLY) || !MXFP_QUANT_ONLY
  printf("dist[0]      = 0x%08x\n", mat_dist[0]);
  printf("dist[63]     = 0x%08x\n", mat_dist[63]);
  if (strcmp(mode, "64") == 0) {
    printf("dist[256]    = 0x%08x\n", mat_dist[256]);
    printf("dist[16191]  = 0x%08x\n", mat_dist[16191]);
  } else {
    printf("dist[64]     = 0x%08x\n", mat_dist[64]);
    printf("dist[127]    = 0x%08x\n", mat_dist[127]);
    printf("dist[128]    = 0x%08x\n", mat_dist[128]);
    printf("dist[32768]  = 0x%08x\n", mat_dist[32768]);
    printf("dist[32896]  = 0x%08x\n", mat_dist[32896]);
  }
#else
  (void)mode;
#endif
}

#if defined(MXFP_MODEL_ACTIVATION) && MXFP_MODEL_ACTIVATION
enum { model_rows = 256, model_tile_mn = 128,
#if defined(MXFP_MODEL_QWEN) && MXFP_MODEL_QWEN
       model_max_k = 3072,
#else
       model_max_k = 768,
#endif
       model_max_blocks = model_rows * (model_max_k / QK_MXFP4), model_guard = 64 };
static float model_input[model_rows * model_max_k] __attribute__((aligned(64)));
static uint8_t model_payload[2 * model_guard + model_max_blocks * 16] __attribute__((aligned(64)));
static uint8_t model_scales[2 * model_guard + model_max_blocks] __attribute__((aligned(64)));
static uint8_t model_ref_payload[model_max_blocks * 16];
static uint8_t model_ref_scales[model_max_blocks];

void __wrap_ggml_abort(const char *file, int line, const char *format, ...) {
  printf("MXModelActivation abort %s:%d %s\n", file, line, format);
  _halt(1);
}

static unsigned mxfp_model_activation_shape(int width, int tile_k) {
  const size_t blocks_per_row = (size_t)width / QK_MXFP4;
  const size_t blocks = model_rows * blocks_per_row;
  const size_t batches = model_rows * ((blocks_per_row + 7) / 8);
  const size_t payload_bytes = blocks * 16;
  const size_t input_bytes = model_rows * (size_t)width * sizeof(float);
  const size_t blocks_per_group = tile_k / QK_MXFP4;
  uint8_t *const payload = model_payload + model_guard;
  uint8_t *const scales = model_scales + model_guard;
  unsigned errors = 0;
  unsigned saved_fcsr;
  __asm__ volatile("frcsr %0\n\tfscsr %1" : "=&r"(saved_fcsr) : "r"(17u) : "memory");
  for (int row = 0; row < model_rows; ++row) {
    float *row_input = model_input + row * width;
    for (size_t block = 0; block < blocks_per_row; ++block) {
      const float amax = (float)(1u << ((row + 3 * block) % 5));
      for (size_t lane = 0; lane < QK_MXFP4; ++lane) {
        row_input[block * QK_MXFP4 + lane] = lane == (row + block) % QK_MXFP4 ? amax :
            lane % 4 == 1 ? -0.5f * amax : lane % 4 == 2 ? 0.25f * amax : 0.0f;
      }
    }
    block_mxfp4 encoded[model_max_k / QK_MXFP4];
    quantize_row_mxfp4(row_input, encoded, width);
    for (size_t block = 0; block < blocks_per_row; block += blocks_per_group) {
      const size_t panel_base = (row / model_tile_mn * blocks_per_row + block) * model_tile_mn;
      const size_t index = panel_base + (row % model_tile_mn) * blocks_per_group;
      errors += !ggml_ame_pack_mxfp4_blocks(model_ref_payload + index * 16,
          model_ref_scales + index, encoded + block, blocks_per_group);
    }
  }
  const uint64_t input_hash = mxfp_quant_hash((const uint8_t *)model_input, input_bytes);
  for (unsigned sample = 0; sample < 5; ++sample) {
    const bool fallback = sample == 4;
    const unsigned seed = fallback ? 17u : 1u;
    uint64_t begin, end;
    unsigned final_fcsr;
    memset(model_payload, 0xa5, payload_bytes + 2 * model_guard);
    memset(model_scales, 0x5a, blocks + 2 * model_guard);
    ggml_mxfp4_packed_stats_reset();
#if defined(GGML_AME_MXFP4_PREP_SPLIT_PROFILE) && GGML_AME_MXFP4_PREP_SPLIT_PROFILE
    struct ame_mx_prep_split_counters split = {0};
    struct ame_mx_prep_split_counters *const previous_split = ggml_ame_mx_prep_split_set(&split);
#endif
    __asm__ volatile("fscsr %1\n\tfence rw, rw\n\trdcycle %0" : "=r"(begin) : "r"(seed) : "memory");
    ggml_ame_mx_prepare_activation(payload, scales, model_input, model_rows,
        width, (size_t)width * sizeof(float), AME_MCFG_MXFP4, tile_k, model_tile_mn);
    __asm__ volatile("fence rw, rw\n\trdcycle %0\n\tfrcsr %1" : "=r"(end), "=r"(final_fcsr) : : "memory");
#if defined(GGML_AME_MXFP4_PREP_SPLIT_PROFILE) && GGML_AME_MXFP4_PREP_SPLIT_PROFILE
    const bool split_restored = ggml_ame_mx_prep_split_set(previous_split) == &split;
#endif
    struct ggml_mxfp4_packed_stats stats;
    ggml_mxfp4_packed_stats_get(&stats);
    unsigned mismatch = 0, canaries = 0;
    for (size_t byte = 0; byte < payload_bytes; ++byte) mismatch += payload[byte] != model_ref_payload[byte];
    for (size_t block = 0; block < blocks; ++block) mismatch += scales[block] != model_ref_scales[block];
    for (size_t byte = 0; byte < model_guard; ++byte) {
      canaries += model_payload[byte] != 0xa5 || payload[payload_bytes + byte] != 0xa5;
      canaries += model_scales[byte] != 0x5a || scales[blocks + byte] != 0x5a;
    }
    const bool valid = end > begin && final_fcsr == seed && mismatch == 0 && canaries == 0 &&
        stats.calls == batches && stats.adjacent_calls == batches &&
        stats.packed_calls == (fallback ? 0 : batches) &&
        stats.packed_blocks == (fallback ? 0 : blocks) && stats.adjacent_blocks == stats.packed_blocks &&
        stats.qualification_fallbacks == (fallback ? batches : 0) &&
        stats.adjacent_fallbacks == stats.qualification_fallbacks &&
        stats.packed_nv_fallbacks == 0 && stats.amax_reuse_calls == 0 &&
        mxfp_quant_hash((const uint8_t *)model_input, input_bytes) == input_hash;
    errors += !valid;
#if defined(GGML_AME_MXFP4_PREP_SPLIT_PROFILE) && GGML_AME_MXFP4_PREP_SPLIT_PROFILE
    const uint64_t total_cycles = end - begin;
    const bool split_within_total = end > begin && split.quantizer_cycles <= total_cycles &&
        split.panel_layout_cycles <= total_cycles - split.quantizer_cycles;
    const bool split_valid = valid && previous_split == NULL && split_restored && split_within_total &&
        split.batches == batches && split.k32_blocks == blocks &&
        split.fallback_batches == stats.qualification_fallbacks &&
        split.quantizer_cycles > 0 && split.panel_layout_cycles > 0;
    errors += valid && !split_valid;
    printf("MXModelActivationLayout N=%u K=%d tile_k=%d sample=%u phase=%s batches=%llu k32=%llu "
           "fallback_batches=%llu quantizer=%llu panel_layout=%llu remainder=%llu total=%llu %s\n",
           model_rows, width, tile_k, sample, fallback ? "fallback" : sample == 0 ? "warmup" : "measure",
           (unsigned long long)split.batches, (unsigned long long)split.k32_blocks,
           (unsigned long long)split.fallback_batches, (unsigned long long)split.quantizer_cycles,
           (unsigned long long)split.panel_layout_cycles,
           (unsigned long long)(split_within_total ? total_cycles - split.quantizer_cycles - split.panel_layout_cycles : 0),
           (unsigned long long)total_cycles, split_valid ? "PASS" : "FAIL");
#endif
#if defined(GGML_MXFP4_PACKED_PHASE_PROFILE) && GGML_MXFP4_PACKED_PHASE_PROFILE
    const uint64_t total_cycles = end - begin;
    const uint64_t accounted = stats.qualification_cycles + stats.vector_pack_cycles + stats.split_reorder_cycles;
    const bool phase_valid = valid && accounted <= total_cycles && stats.rms_rows == 0 && stats.rms_blocks == 0 &&
        (fallback ? accounted == 0 : stats.qualification_cycles > 0 && stats.vector_pack_cycles > 0);
    errors += valid && !phase_valid;
    printf("MXModelActivationPhase K=%d sample=%u qualification=%llu vector_pack=%llu split_window=%llu "
           "unassigned=%llu total=%llu %s\n", width, sample,
           (unsigned long long)stats.qualification_cycles, (unsigned long long)stats.vector_pack_cycles,
           (unsigned long long)stats.split_reorder_cycles,
           (unsigned long long)(accounted <= total_cycles ? total_cycles - accounted : 0),
           (unsigned long long)total_cycles, phase_valid ? "PASS" : "FAIL");
#endif
    printf("MXModelActivation N=%u K=%d tile_k=%d tile_mn=%u sample=%u phase=%s cycles=%llu "
           "calls=%llu packed=%llu k32=%llu qual_fallback=%llu post_nv=%llu fcsr=%u bytes=%u canaries=%u %s\n",
           model_rows, width, tile_k, model_tile_mn, sample,
           fallback ? "fallback" : sample == 0 ? "warmup" : "measure", (unsigned long long)(end - begin),
           (unsigned long long)stats.calls, (unsigned long long)stats.packed_calls,
           (unsigned long long)stats.packed_blocks, (unsigned long long)stats.qualification_fallbacks,
           (unsigned long long)stats.packed_nv_fallbacks, final_fcsr, mismatch, canaries, valid ? "PASS" : "FAIL");
  }
  __asm__ volatile("fscsr %0" : : "r"(saved_fcsr) : "memory");
  return errors;
}

static int run_mxfp_model_activation(void) {
#if defined(MXFP_MODEL_QWEN) && MXFP_MODEL_QWEN
  unsigned errors = mxfp_model_activation_shape(1024, 64);
  errors += mxfp_model_activation_shape(2048, 64);
  errors += mxfp_model_activation_shape(3072, 64);
#else
  unsigned errors = mxfp_model_activation_shape(288, 32);
  errors += mxfp_model_activation_shape(768, 64);
#endif
  printf("MXModelActivation %s errors=%u\n", errors == 0 ? "PASS" : "FAIL", errors);
  return errors != 0;
}
#endif

#if defined(MXFP_MODEL_WEIGHT_FILE) || defined(MXFP_MODEL_LMHEAD_WEIGHT_FILE) || defined(F16_MODEL_SOFTMAX_LOGITS_FILE)
#define MXFP4_STRINGIFY_IMPL(value) #value
#define MXFP4_STRINGIFY(value) MXFP4_STRINGIFY_IMPL(value)
#define EMBED_MXFP4_FIXTURE_FILE(symbol, path, bytes)                        \
  __asm__(".pushsection .rodata." #symbol ",\"a\",@progbits\n"              \
          ".balign 64\n.local " #symbol "\n.type " #symbol ",@object\n"      \
          #symbol ":\n.incbin \"" path "\"\n"                              \
          ".size " #symbol ", . - " #symbol "\n"                            \
          ".if (. - " #symbol ") != " MXFP4_STRINGIFY(bytes) "\n"           \
          ".error \"unexpected MXFP4 fixture file size\"\n.endif\n"          \
          ".popsection\n")
#endif

#if defined(MXFP_MODEL_WEIGHT) && MXFP_MODEL_WEIGHT
extern void ggml_ame_test_weight_match_set(void *source, void *packed,
    int64_t rows, int64_t columns, int tile_k, int tile_mn);
extern bool ggml_ame_test_weight_match(void);

enum { weight_tile_mn = 128, weight_tile_k = 64, weight_guard = 64,
       weight_max_blocks = 1024 * 3072 / QK_MXFP4 };

#ifdef MXFP_MODEL_WEIGHT_FILE
enum { weight_fixture_blocks = (1024 + 2048 + 3072 + 2048 + 3072) * 1024 / QK_MXFP4 };
extern const block_mxfp4 model_weight_fixture[weight_fixture_blocks];
_Static_assert(sizeof(model_weight_fixture) == 6127616, "expected five Qwen MXFP4 shapes");
EMBED_MXFP4_FIXTURE_FILE(model_weight_fixture, MXFP_MODEL_WEIGHT_FILE, 6127616);
#endif

static block_mxfp4 model_weight_source[weight_max_blocks + 2 * weight_guard]
    __attribute__((aligned(64)));
static uint8_t model_weight_packed[17 * weight_max_blocks + 2 * weight_guard]
    __attribute__((aligned(64)));

static uint8_t model_weight_scale(size_t block) {
  return 1 + (block * 3 % 254);
}

static uint8_t model_weight_code_byte(size_t block, unsigned byte) {
  return (uint8_t)(block * 13 + (block >> 8) + byte * 73);
}

static unsigned model_weight_shape(int rows, int columns, bool negative_checks,
                                   const block_mxfp4 *fixture) {
  const size_t nb = (size_t) columns / QK_MXFP4;
  const size_t blocks = (size_t) rows * nb;
  block_mxfp4 *source = model_weight_source + weight_guard;
  uint8_t *packed = model_weight_packed + weight_guard;
  uint8_t *scales = packed + 16 * blocks;
  if (blocks == 0 || blocks > weight_max_blocks || rows % weight_tile_mn || columns % weight_tile_k)
    return 1;

  uint8_t *source_raw = (uint8_t *) source;
  uint8_t *source_end = (uint8_t *) (source + blocks);
  uint8_t *packed_end = scales + blocks;
  memset(source_raw - weight_guard, 0x5b, weight_guard);
  memset(source_end, 0x5b, weight_guard);
  memset(packed - weight_guard, 0x5b, weight_guard);
  memset(packed_end, 0x5b, weight_guard);
  if (fixture != NULL) {
    memcpy(source, fixture, blocks * sizeof(*source));
  } else {
    for (size_t block = 0; block < blocks; ++block) {
      source[block].e = model_weight_scale(block);
      for (unsigned byte = 0; byte < sizeof(source[block].qs); ++byte)
        source[block].qs[byte] = model_weight_code_byte(block, byte);
    }
  }
  memset(packed, 0xa5, 17 * blocks);
  for (int row = 0; row < rows; ++row) {
    for (size_t kb = 0; kb < nb; kb += 2) {
      const size_t index = ggml_ame_mx_packed_block_index(row, kb, nb, weight_tile_k, weight_tile_mn);
      if (!ggml_ame_pack_mxfp4_blocks(packed + 16 * index, scales + index,
                                      source + (size_t) row * nb + kb, 2)) return 1;
    }
  }
  ggml_ame_test_weight_match_set(source, packed, rows, columns, weight_tile_k, weight_tile_mn);
  unsigned errors = 0;
  for (unsigned sample = 0; sample < 4; ++sample) {
    uint64_t begin, end;
    uint32_t saved, fcsr;
    const uint32_t seed = 0x81;
    __asm__ volatile("frcsr %0\n\tfscsr %2\n\tfence rw,rw\n\trdcycle %1"
                     : "=&r"(saved), "=r"(begin) : "r"(seed) : "memory");
    const bool same = ggml_ame_test_weight_match();
    __asm__ volatile("fence rw,rw\n\trdcycle %0\n\tfrcsr %1\n\tfscsr %2"
                     : "=&r"(end), "=&r"(fcsr) : "r"(saved) : "memory");
    const bool pass = same && fcsr == seed && end > begin;
    errors += !pass;
    printf("MXModelWeight M=%d N=256 K=%d tile_k=64 tile_mn=128 stride=%u sample=%u phase=%s "
           "cycles=%llu expected_k32=%u hit=%u fcsr=%u %s\n", rows, columns,
           (unsigned) (nb * sizeof(*source)), sample, sample == 0 ? "warmup" : "measure",
           (unsigned long long) (end - begin), (unsigned) blocks, (unsigned) same, fcsr,
           pass ? "PASS" : "FAIL");
  }

  unsigned payload_rejected = 0, scale_rejected = 0;
  if (negative_checks) {
    source[blocks-1].qs[15] ^= 0x80;
    payload_rejected = !ggml_ame_test_weight_match();
    source[blocks-1].qs[15] ^= 0x80;
    source[blocks-1].e ^= 1;
    scale_rejected = !ggml_ame_test_weight_match();
    source[blocks-1].e ^= 1;
    errors += !payload_rejected + !scale_rejected;
  }
  unsigned source_bad = 0, payload_bad = 0, scale_bad = 0, guard_bad = 0;
  for (int row = 0; row < rows; ++row) {
    for (size_t kb = 0; kb < nb; ++kb) {
      const size_t block = (size_t) row * nb + kb;
      const size_t panel = ((size_t) row / 128) * ((size_t) columns / 64) + kb / 2;
      const size_t index = (panel * 128 + (size_t) row % 128) * 2 + kb % 2;
      const uint8_t scale = fixture != NULL ? fixture[block].e : model_weight_scale(block);
      source_bad += source[block].e != scale || scale == 255;
      scale_bad += scales[index] != scale;
      for (unsigned lane = 0; lane < QK_MXFP4; ++lane) {
        const uint8_t expected = fixture != NULL ? fixture[block].qs[lane % 16] :
            model_weight_code_byte(block, lane % 16);
        if (lane < 16) source_bad += source[block].qs[lane] != expected;
        const unsigned decoded = (packed[16 * index + lane / 2] >> (4 * (lane % 2))) & 15;
        payload_bad += decoded != ((expected >> (lane < 16 ? 0 : 4)) & 15u);
      }
    }
  }
  for (unsigned byte = 0; byte < weight_guard; ++byte) {
    guard_bad += (source_raw - weight_guard)[byte] != 0x5b;
    guard_bad += source_end[byte] != 0x5b;
    guard_bad += (packed - weight_guard)[byte] != 0x5b;
    guard_bad += packed_end[byte] != 0x5b;
  }
  errors += source_bad + payload_bad + scale_bad + guard_bad;
  printf("MXModelWeightOracle M=%d K=%d source=%u payload=%u scale=%u guards=%u "
         "payload_rejected=%u scale_rejected=%u %s\n", rows, columns, source_bad,
         payload_bad, scale_bad, guard_bad, payload_rejected, scale_rejected, errors ? "FAIL" : "PASS");
  return errors;
}

static int run_mxfp_model_weight(void) {
  static const struct { int rows, columns; } shapes[] = {
    { 1024, 1024 }, { 1024, 2048 }, { 1024, 3072 }, { 2048, 1024 }, { 3072, 1024 },
  };
  const block_mxfp4 *fixture = NULL;
#ifdef MXFP_MODEL_WEIGHT_FILE
  fixture = model_weight_fixture;
  printf("MXModelWeightFixture model=qwen06 source=gguf shapes=%u bytes=%u\n",
         (unsigned) (sizeof(shapes) / sizeof(shapes[0])), (unsigned) sizeof(model_weight_fixture));
#endif
  unsigned errors = 0;
  for (unsigned shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); ++shape) {
    errors += model_weight_shape(shapes[shape].rows, shapes[shape].columns, shape == 0, fixture);
    if (fixture != NULL) fixture += (size_t) shapes[shape].rows * shapes[shape].columns / QK_MXFP4;
  }
  printf("MXModelWeight %s errors=%u\n", errors ? "FAIL" : "PASS", errors);
  return errors != 0;
}
#endif

#if (defined(F16_MODEL_ATTENTION) && F16_MODEL_ATTENTION) || \
    (defined(MXFP_MODEL_LMHEAD_OUTER) && MXFP_MODEL_LMHEAD_OUTER)
struct ggml_threadpool *ggml_xsai_am_single_thread_pool(void);

#if defined(MXFP_MODEL_LMHEAD_OUTER) && MXFP_MODEL_LMHEAD_OUTER
#define MODEL_GGML_LOG_PREFIX "MXModelLMHead"
#else
#define MODEL_GGML_LOG_PREFIX "F16ModelAttention"
#endif

void __wrap_ggml_log_internal(enum ggml_log_level level, const char *format, ...) {
  va_list args;
  va_start(args, format);
  printf(MODEL_GGML_LOG_PREFIX " ggml log level=%d: ", (int)level);
  vprintf(format, args);
  va_end(args);
}

void __wrap_ggml_abort(const char *file, int line, const char *format, ...) {
  va_list args;
  va_start(args, format);
  printf(MODEL_GGML_LOG_PREFIX " abort %s:%d: ", file, line);
  vprintf(format, args);
  va_end(args);
  _halt(1);
}
#undef MODEL_GGML_LOG_PREFIX
#endif

#if defined(MXFP_MODEL_LMHEAD_OUTER) && MXFP_MODEL_LMHEAD_OUTER
int __wrap_printf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  const int result = vprintf(format, args);
  va_end(args);
  return result;
}

int __wrap_putchar(int character) {
  const unsigned char byte = (unsigned char)character;
  _putc((char)byte);
  return byte;
}
#endif

#if defined(MXFP_MODEL_LMHEAD) && MXFP_MODEL_LMHEAD
enum {
  lm_rows = MXFP_MODEL_N1_ROWS,
  lm_blocks = MXFP_MODEL_N1_BLOCKS,
  lm_width = lm_blocks * QK_MXFP4,
  lm_guard = 32,
};
#ifdef MXFP_MODEL_LMHEAD_WEIGHT_FILE
#define MXFP_MODEL_N1_WEIGHT_BYTES (MXFP_MODEL_N1_ROWS * MXFP_MODEL_N1_BLOCKS * 17)
extern const block_mxfp4 lm_weights[lm_rows * lm_blocks];
_Static_assert(sizeof(lm_weights) == MXFP_MODEL_N1_WEIGHT_BYTES, "expected packed MXFP4 projection shape");
EMBED_MXFP4_FIXTURE_FILE(lm_weights, MXFP_MODEL_LMHEAD_WEIGHT_FILE, MXFP_MODEL_N1_WEIGHT_BYTES);
#else
static block_mxfp4 lm_weights[lm_rows * lm_blocks] __attribute__((aligned(64)));
#endif
static block_q8_0 lm_rhs[lm_blocks + 2] __attribute__((aligned(64)));
static float lm_src[lm_width] __attribute__((aligned(64)));
static float lm_output[lm_rows + 2 * lm_guard] __attribute__((aligned(64)));

#ifdef MXFP_MODEL_LMHEAD_REFERENCE_FILE
#define MXFP_MODEL_N1_REFERENCE_BYTES (MXFP_MODEL_N1_ROWS * 4 + MXFP_MODEL_N1_BLOCKS * 34)
struct lm_reference_fixture {
  uint32_t output[lm_rows];
  uint8_t rhs[lm_blocks * sizeof(block_q8_0)];
};
extern const struct lm_reference_fixture lm_reference;
_Static_assert(sizeof(block_q8_0) == 34 && sizeof(lm_reference) == MXFP_MODEL_N1_REFERENCE_BYTES,
               "expected Qwen N1 reference layout");
EMBED_MXFP4_FIXTURE_FILE(lm_reference, MXFP_MODEL_LMHEAD_REFERENCE_FILE, MXFP_MODEL_N1_REFERENCE_BYTES);
#endif

static int lm_code(uint8_t nibble) {
  static const int8_t magnitude[] = { 0, 1, 2, 3, 4, 6, 8, 12 };
  const int value = magnitude[nibble & 7];
  return (nibble & 8) ? -value : value;
}

static float lm_dot_reference(const block_mxfp4 *weights, const block_q8_0 *rhs, unsigned blocks) {
  float result = 0.0f;
  for (unsigned block = 0; block < blocks; ++block) {
    int isum = 0;
    for (unsigned lane = 0; lane < QK_MXFP4; ++lane) {
      const unsigned half = QK_MXFP4 / 2;
      const uint8_t codes = weights[block].qs[lane % half];
      isum += lm_code(lane < half ? codes & 15 : codes >> 4) * rhs[block].qs[lane];
    }
    const float lhs_scale = GGML_E8M0_TO_FP32_HALF(weights[block].e);
    const float rhs_scale = GGML_COMPUTE_FP16_TO_FP32(rhs[block].d);
    const float isum_f32 = (float)isum;
    float scale;
    __asm__ volatile("fmul.s %0, %2, %3, dyn\n\tfmadd.s %1, %0, %4, %1, dyn"
                     : "=&f"(scale), "+f"(result) : "f"(lhs_scale), "f"(rhs_scale), "f"(isum_f32));
  }
  return result;
}

static unsigned lm_directed(void) {
  static const unsigned widths[] = { 1, 8, lm_blocks };
  block_mxfp4 weights[lm_blocks];
  block_q8_0 rhs[lm_blocks];
  unsigned mismatches = 0, cases = 0;
  for (unsigned fixture = 0; fixture < 3; ++fixture) {
    for (unsigned block = 0; block < lm_blocks; ++block) {
      weights[block].e = fixture == 0 ? 127 : (uint8_t)(block < 2 ? block : 122 + block);
      rhs[block].d = fixture == 0 ? 0x3c00 : (ggml_half)(0x3555 + block * 17);
      for (unsigned lane = 0; lane < QK_MXFP4 / 2; ++lane) {
        weights[block].qs[lane] = fixture == 0 ? 0xff : (uint8_t)(fixture * 71 + 19 * lane + 53 * block);
      }
      for (unsigned lane = 0; lane < QK_MXFP4; ++lane) {
        rhs[block].qs[lane] = fixture == 0 ? -128 : (int8_t)(fixture * 43 + 13 * lane + 39 * block);
      }
    }
    for (unsigned frm = 0; frm < 5; ++frm) {
      for (unsigned width = 0; width < sizeof(widths) / sizeof(widths[0]); ++width) {
        const unsigned seed = (frm << 5) | 8;
        unsigned saved, expected_fcsr, actual_fcsr;
        float actual;
        __asm__ volatile("frcsr %0\n\tfscsr %1" : "=&r"(saved) : "r"(seed) : "memory");
        const float expected = lm_dot_reference(weights, rhs, widths[width]);
        __asm__ volatile("frcsr %0\n\tfscsr %1" : "=&r"(expected_fcsr) : "r"(seed), "f"(expected) : "memory");
        ggml_vec_dot_mxfp4_q8_0(widths[width] * QK_MXFP4, &actual, 0, weights, 0, rhs, 0, 1);
        __asm__ volatile("frcsr %0\n\tfscsr %1" : "=&r"(actual_fcsr) : "r"(saved) : "memory");
        mismatches += fp32_to_bits(actual) != fp32_to_bits(expected) || actual_fcsr != expected_fcsr;
        ++cases;
      }
    }
  }
  printf("MXModelLMHeadDirected cases=%u frm=5 widths=32,256,%u mismatches=%u\n", cases, lm_width, mismatches);
  return mismatches;
}

static int run_mxfp_model_lmhead(void) {
  unsigned errors = lm_directed();
#ifdef MXFP_MODEL_LMHEAD_WEIGHT_FILE
  printf("MXModelLMHeadFixture weights=file bytes=%zu activation=synthetic\n", sizeof(lm_weights));
#else
  for (unsigned index = 0; index < lm_rows * lm_blocks; ++index) {
    lm_weights[index].e = 121 + (index * 3 + index / lm_blocks) % 13;
    for (unsigned lane = 0; lane < QK_MXFP4 / 2; ++lane) {
      lm_weights[index].qs[lane] = (uint8_t)(index * 13 + (index >> 8) + lane * 73);
    }
  }
#endif
  for (unsigned lane = 0; lane < lm_width; ++lane) {
    lm_src[lane] = ((int)((lane * 139 + lane / QK_MXFP4) % 257) - 128) * 0x1p-6f;
  }

  const float sentinel = 12345.0f;
  float *out = lm_output + lm_guard;
  block_q8_0 *rhs = lm_rhs + 1;
#if defined(MXFP_MODEL_LMHEAD_OUTER) && MXFP_MODEL_LMHEAD_OUTER
  const size_t weight_stride = lm_blocks * sizeof(block_mxfp4);
  struct ggml_tensor weights = {
    .type = GGML_TYPE_MXFP4, .data = (void *)lm_weights, .ne = { lm_width, lm_rows, 1, 1 },
    .nb = { sizeof(block_mxfp4), weight_stride, sizeof(lm_weights), sizeof(lm_weights) },
  };
  struct ggml_tensor input = {
    .type = GGML_TYPE_F32, .data = lm_src, .ne = { lm_width, 1, 1, 1 },
    .nb = { sizeof(float), sizeof(lm_src), sizeof(lm_src), sizeof(lm_src) },
  };
  struct ggml_tensor output = {
    .type = GGML_TYPE_F32, .op = GGML_OP_MUL_MAT, .src = { &weights, &input },
    .data = out, .ne = { lm_rows, 1, 1, 1 },
    .nb = { sizeof(float), lm_rows * sizeof(float), lm_rows * sizeof(float), lm_rows * sizeof(float) },
  };
  const struct ggml_compute_params params = {
    .ith = 0, .nth = 1, .wsize = lm_blocks * sizeof(*rhs), .wdata = rhs,
    .threadpool = ggml_xsai_am_single_thread_pool(),
  };
  printf("MXModelLMHeadPath entry=ggml_compute_forward_mul_mat nth=1 weight_stride=%zu\n", weight_stride);
#endif
  uint64_t previous_digest = 0;
  for (unsigned sample = 0; sample < 4; ++sample) {
    for (unsigned elem = 0; elem < lm_rows + 2 * lm_guard; ++elem) lm_output[elem] = sentinel;
    memset(lm_rhs, 0x5b, sizeof(lm_rhs));
    unsigned saved, fcsr;
    uint64_t begin, end;
    const unsigned seed = 8;
    __asm__ volatile("frcsr %0\n\tfscsr %2\n\tfence rw,rw\n\trdcycle %1"
                     : "=&r"(saved), "=r"(begin) : "r"(seed) : "memory");
#if defined(MXFP_MODEL_LMHEAD_OUTER) && MXFP_MODEL_LMHEAD_OUTER
    ggml_compute_forward_mul_mat(&params, &output);
#else
    quantize_row_q8_0(lm_src, rhs, lm_width);
    for (unsigned row = 0; row < lm_rows; ++row) {
      ggml_vec_dot_mxfp4_q8_0(lm_width, out + row, 0, lm_weights + row * lm_blocks, 0, rhs, 0, 1);
    }
#endif
    __asm__ volatile("fence rw,rw\n\trdcycle %0\n\tfrcsr %1\n\tfscsr %2"
                     : "=&r"(end), "=&r"(fcsr) : "r"(seed) : "memory");
    unsigned bad = 0, guards = 0;
    uint64_t digest = 14695981039346656037ull;
    for (unsigned row = 0; row < lm_rows; ++row) {
      const uint32_t bits = fp32_to_bits(out[row]);
      bad += out[row] == sentinel;
      if (sample == 3) {
#ifdef MXFP_MODEL_LMHEAD_REFERENCE_FILE
        bad += bits != lm_reference.output[row];
#else
        bad += bits != fp32_to_bits(lm_dot_reference(lm_weights + row * lm_blocks, rhs, lm_blocks));
#endif
      }
      digest = (digest ^ bits) * 1099511628211ull;
    }
    for (unsigned byte = 0; byte < lm_blocks * sizeof(*rhs); ++byte) {
      const uint8_t rhs_byte = ((const uint8_t *)rhs)[byte];
#ifdef MXFP_MODEL_LMHEAD_REFERENCE_FILE
      if (sample == 3) {
        bad += rhs_byte != lm_reference.rhs[byte];
      }
#endif
      digest = (digest ^ rhs_byte) * 1099511628211ull;
    }
    __asm__ volatile("fscsr %0" : : "r"(saved) : "memory");
    for (unsigned elem = 0; elem < lm_guard; ++elem) {
      guards += lm_output[elem] != sentinel || out[lm_rows + elem] != sentinel;
    }
    for (unsigned byte = 0; byte < sizeof(*rhs); ++byte) {
      guards += ((const uint8_t *)lm_rhs)[byte] != 0x5b || ((const uint8_t *)(rhs + lm_blocks))[byte] != 0x5b;
    }
    const bool pass = end > begin && bad == 0 && guards == 0 && fcsr == (seed | 1) &&
        (sample == 0 || digest == previous_digest);
    previous_digest = digest;
    errors += !pass;
    printf("MXModelLMHead M=%u N=1 K=%u sample=%u phase=%s cycles=%llu k32=%u outputs=%u "
           "fcsr=%u digest=%016llx mismatches=%u canaries=%u %s\n",
           lm_rows, lm_width, sample, sample == 0 ? "warmup" : "measure", (unsigned long long)(end - begin),
           lm_rows * lm_blocks, lm_rows, fcsr, (unsigned long long)digest, bad, guards, pass ? "PASS" : "FAIL");
  }
  printf("MXModelLMHead %s errors=%u\n", errors == 0 ? "PASS" : "FAIL", errors);
  return errors != 0;
}
#endif

#if defined(F16_MODEL_ATTENTION) && F16_MODEL_ATTENTION
enum { attn_heads = F16_MODEL_ATTENTION_QWEN ? 16 : 6,
       attn_kv_heads = F16_MODEL_ATTENTION_QWEN ? 8 : attn_heads,
       attn_tokens = 256, attn_width = F16_MODEL_ATTENTION_QWEN ? 128 : 48, attn_period = 31,
       attn_lhs_elems = attn_kv_heads * attn_tokens * attn_width,
       attn_rhs_elems = attn_heads * attn_tokens * attn_tokens, attn_guard = 32 };

struct f16_attn_shape {
  const char *name;
  int rows, inner, lhs_row, lhs_head, rhs_row, rhs_head;
  bool contiguous_rhs;
};

static ggml_fp16_t attn_lhs[attn_lhs_elems] __attribute__((aligned(64)));
static float attn_rhs[attn_rhs_elems] __attribute__((aligned(64)));
static ggml_fp16_t attn_rhs_f16[attn_rhs_elems + 2 * attn_guard] __attribute__((aligned(64)));
static ggml_fp16_t attn_expected_f16[attn_rhs_elems];
static float attn_out[attn_rhs_elems + 2 * attn_guard] __attribute__((aligned(64)));
static int attn_dot[attn_period][attn_period];

static unsigned f16_attn_cast_check(void) {
  const float input[] = { -0.0f, 0x1.002p0f, 0x1.006p0f, -0x1.006p0f, 0x1p-24f };
  const ggml_fp16_t expected[] = { 0x8000, 0x3c00, 0x3c02, 0xbc02, 0x0001 };
  ggml_fp16_t actual[sizeof(expected) / sizeof(expected[0])];
  unsigned saved, flags;
  __asm__ volatile("frcsr %0\n\tfscsr zero" : "=r"(saved) : : "memory");
  ggml_cpu_fp32_to_fp16(input, actual, sizeof(actual) / sizeof(actual[0]));
  __asm__ volatile("frcsr %0\n\tfscsr %1" : "=r"(flags) : "r"(saved) : "memory");
  return flags != 1 || memcmp(actual, expected, sizeof(expected)) != 0;
}

static int f16_attn_value(int phase, int step, int lane) {
  return (phase + step * lane) % attn_period - attn_period / 2;
}

static int f16_attn_kv_head(int query_head) {
  return query_head / (attn_heads / attn_kv_heads);
}

static unsigned f16_attn_run(const struct f16_attn_shape *shape,
                            const struct ggml_compute_params *params, unsigned sample_limit) {
  const size_t converted = (size_t)attn_heads * attn_tokens * shape->inner;
  const size_t outputs = (size_t)attn_heads * attn_tokens * shape->rows;
  ggml_fp16_t *const narrow = attn_rhs_f16 + attn_guard;
  float *const output = attn_out + attn_guard;
  const float sentinel = -12345.0f;
  unsigned errors = 0;
  for (int left_phase = 0; left_phase < attn_period; ++left_phase) {
    for (int right_phase = 0; right_phase < attn_period; ++right_phase) {
      int sum = 0;
      for (int lane = 0; lane < shape->inner; ++lane) {
        sum += f16_attn_value(left_phase, 5, lane) * f16_attn_value(right_phase, 13, lane);
      }
      attn_dot[left_phase][right_phase] = sum;
    }
  }
  for (int head = 0; head < attn_heads; ++head) {
    if (head % (attn_heads / attn_kv_heads) == 0) {
      const int kv_head = f16_attn_kv_head(head);
      for (int row = 0; row < shape->rows; ++row) {
        const int phase = (11 * kv_head + 3 * row) % attn_period;
        for (int lane = 0; lane < shape->inner; ++lane) {
          float value = f16_attn_value(phase, 5, lane) * (1 + row / attn_period) * 0x1p-7f;
          attn_lhs[kv_head * shape->lhs_head + row * shape->lhs_row + lane] = GGML_COMPUTE_FP32_TO_FP16(value);
        }
      }
    }
    for (int token = 0; token < attn_tokens; ++token) {
      const int phase = (17 * head + 7 * token) % attn_period;
      for (int lane = 0; lane < shape->inner; ++lane) {
        float value = f16_attn_value(phase, 13, lane) * (1 + token / attn_period) * 0x1p-7f;
        attn_rhs[head * shape->rhs_head + token * shape->rhs_row + lane] = value;
        attn_expected_f16[(head * attn_tokens + token) * shape->inner + lane] = GGML_COMPUTE_FP32_TO_FP16(value);
      }
    }
  }

  for (unsigned sample = 0; sample < sample_limit; ++sample) {
    for (size_t elem = 0; elem < outputs + 2 * attn_guard; ++elem) attn_out[elem] = sentinel;
    memset(attn_rhs_f16, 0xff, (converted + 2 * attn_guard) * sizeof(*narrow));
    unsigned saved, final_fcsr, hits = 0;
    uint64_t begin, end;
#if F16_MODEL_ATTENTION_SPLIT
    uint64_t cast_begin, cast_end;
#endif
    const unsigned seed = 8;
    __asm__ volatile("frcsr %0\n\tfscsr %2\n\tfence rw,rw\n\trdcycle %1"
                     : "=&r"(saved), "=r"(begin) : "r"(seed) : "memory");
    bool rejected = !shape->contiguous_rhs ||
        !llamafile_sgemm(params, shape->rows, attn_tokens, shape->inner,
          attn_lhs, shape->lhs_row, attn_rhs, shape->rhs_row, output, shape->rows,
          GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_F32);
#if F16_MODEL_ATTENTION_SPLIT
    __asm__ volatile("fence rw,rw\n\trdcycle %0" : "=r"(cast_begin) : : "memory");
#endif
    for (int head = 0; head < attn_heads; ++head) {
      for (int token = 0; token < attn_tokens; ++token) {
        ggml_cpu_fp32_to_fp16(attn_rhs + head * shape->rhs_head + token * shape->rhs_row,
          narrow + (head * attn_tokens + token) * shape->inner, shape->inner);
      }
    }
#if F16_MODEL_ATTENTION_SPLIT
    __asm__ volatile("fence rw,rw\n\trdcycle %0" : "=r"(cast_end) : : "memory");
#endif
    ggml_threadpool_chunk_set(params->threadpool, params->nth);
    ggml_barrier(params->threadpool);
    for (int head = 0; head < attn_heads; ++head) {
      hits += llamafile_sgemm(params, shape->rows, attn_tokens, shape->inner,
        attn_lhs + f16_attn_kv_head(head) * shape->lhs_head, shape->lhs_row,
        narrow + head * attn_tokens * shape->inner, shape->inner,
        output + head * attn_tokens * shape->rows, shape->rows,
        GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32);
    }
    __asm__ volatile("fence rw,rw\n\trdcycle %0\n\tfrcsr %1\n\tfscsr %2"
                     : "=&r"(end), "=&r"(final_fcsr) : "r"(saved) : "memory");
    unsigned bad_out = 0, bad_cast = 0, bad_guard = 0;
    for (int head = 0; head < attn_heads; ++head) {
      const int expected_kv_head = head * attn_kv_heads / attn_heads;
      for (int token = 0; token < attn_tokens; ++token) {
        const int right_phase = (17 * head + 7 * token) % attn_period;
        for (int row = 0; row < shape->rows; ++row) {
          const int left_phase = (11 * expected_kv_head + 3 * row) % attn_period;
          const float expected = attn_dot[left_phase][right_phase] *
              (1 + row / attn_period) * (1 + token / attn_period) * 0x1p-14f;
          bad_out += output[(head * attn_tokens + token) * shape->rows + row] != expected;
        }
      }
    }
    for (size_t elem = 0; elem < converted; ++elem) bad_cast += narrow[elem] != attn_expected_f16[elem];
    for (int elem = 0; elem < attn_guard; ++elem) {
      bad_guard += attn_out[elem] != sentinel || output[outputs + elem] != sentinel;
      bad_guard += attn_rhs_f16[elem] != 0xffff || narrow[converted + elem] != 0xffff;
    }
    bool valid = rejected && hits == attn_heads && end > begin && final_fcsr == seed &&
        bad_out == 0 && bad_cast == 0 && bad_guard == 0;
#if F16_MODEL_ATTENTION_SPLIT
    valid = valid && begin <= cast_begin && cast_begin < cast_end && cast_end < end;
    printf("F16ModelAttentionPhase name=%s sample=%u pre_cast=%llu cast=%llu handoff_sgemm=%llu total=%llu %s\n",
        shape->name, sample, (unsigned long long)(cast_begin - begin),
        (unsigned long long)(cast_end - cast_begin), (unsigned long long)(end - cast_end),
        (unsigned long long)(end - begin), valid ? "PASS" : "FAIL");
#endif
    errors += !valid;
    printf("F16ModelAttention name=%s M=%d N=%u K=%d heads=%u lda=%d rhs_stride=%d sample=%u phase=%s "
           "cycles=%llu hits=%u cast_elements=%u outputs=%u mixed_attempt=%u mixed_ok=%u "
           "bad_out=%u bad_cast=%u canaries=%u fcsr=%u %s\n",
        shape->name, shape->rows, attn_tokens, shape->inner, attn_heads, shape->lhs_row, shape->rhs_row,
        sample, sample == 0 ? "warmup" : "measure", (unsigned long long)(end - begin),
        hits, (unsigned)converted, (unsigned)outputs, shape->contiguous_rhs, rejected,
        bad_out, bad_cast, bad_guard, final_fcsr, valid ? "PASS" : "FAIL");
  }
  return errors;
}

static int run_f16_model_attention(unsigned sample_limit) {
  static const struct f16_attn_shape cases[] = {
    { "kq",  attn_tokens, attn_width, attn_kv_heads * attn_width, attn_width,
      attn_heads * attn_width, attn_width, false },
    { "kqv", attn_width, attn_tokens, attn_tokens, attn_tokens * attn_width,
      attn_tokens, attn_tokens * attn_tokens, true },
  };
  const struct ggml_compute_params params = { .ith = 0, .nth = 1,
    .threadpool = ggml_xsai_am_single_thread_pool() };
  if (F16_MODEL_ATTENTION_QWEN) {
    printf("F16ModelAttentionConfig model=qwen06 q_heads=%d kv_heads=%d tokens=%d head_width=%d\n",
        attn_heads, attn_kv_heads, attn_tokens, attn_width);
  }
  unsigned errors = f16_attn_cast_check();
  for (unsigned shape = 0; shape < sizeof(cases) / sizeof(cases[0]); ++shape) {
    errors += f16_attn_run(&cases[shape], &params, sample_limit);
  }
  printf("F16ModelAttention %s errors=%u\n", errors == 0 ? "PASS" : "FAIL", errors);
  return errors != 0;
}

#if defined(F16_MODEL_ATTENTION_SOFTMAX) && F16_MODEL_ATTENTION_SOFTMAX
static float attn_mask[attn_tokens * attn_tokens] __attribute__((aligned(64)));
static float attn_softmax_work[attn_tokens + CACHE_LINE_SIZE / sizeof(float) + 2 * attn_guard] __attribute__((aligned(64)));
#ifdef F16_MODEL_SOFTMAX_LOGITS_FILE
extern const float attn_softmax_logits_fixture[attn_rhs_elems];
_Static_assert(sizeof(attn_rhs) == 1572864, "expected Stories PP256 scores");
EMBED_MXFP4_FIXTURE_FILE(attn_softmax_logits_fixture, F16_MODEL_SOFTMAX_LOGITS_FILE, 1572864);
#endif

static struct ggml_tensor f16_attn_softmax_tensor(float *data, int heads) {
  const size_t row_bytes = sizeof(float) * attn_tokens;
  const size_t head_bytes = row_bytes * attn_tokens;
  return (struct ggml_tensor){ .type = GGML_TYPE_F32, .data = data,
    .ne = { attn_tokens, attn_tokens, heads, 1 },
    .nb = { sizeof(float), row_bytes, head_bytes, head_bytes * heads } };
}

static int run_f16_model_softmax(void) {
  float *const output = attn_out + attn_guard;
  float *const work = attn_softmax_work + attn_guard;
  const float sentinel = -12345.0f;
  const float neg_inf = -__builtin_inff();
  const float scale = 1.0f / sqrtf((float)attn_width);
  const unsigned fcsr_seed = 0x08, expected_fcsr = 0x0b;
  struct ggml_tensor input = f16_attn_softmax_tensor(attn_rhs, attn_heads);
  struct ggml_tensor mask = f16_attn_softmax_tensor(attn_mask, 1);
  struct ggml_tensor dst = f16_attn_softmax_tensor(output, attn_heads);
  dst.op = GGML_OP_SOFT_MAX;
  dst.src[0] = &input;
  dst.src[1] = &mask;
  memcpy(dst.op_params, &scale, sizeof(scale));
  const struct ggml_compute_params params = { .ith = 0, .nth = 1,
    .wdata = work, .wsize = (attn_tokens + CACHE_LINE_SIZE_F32) * sizeof(float) };

  for (unsigned token = 0; token < attn_tokens; ++token) {
    for (unsigned lane = 0; lane < attn_tokens; ++lane) {
      const unsigned row_index = token * attn_tokens + lane;
      attn_mask[row_index] = lane <= token ? 0.0f : neg_inf;
#ifndef F16_MODEL_SOFTMAX_LOGITS_FILE
      for (unsigned head = 0; head < attn_heads; ++head) {
        attn_rhs[head * attn_tokens * attn_tokens + row_index] =
            (float)((int)((103 * lane + 19 * token + 83 * head) % 263) - 131) * 0x1p-4f;
      }
#endif
    }
  }
#ifdef F16_MODEL_SOFTMAX_LOGITS_FILE
  memcpy(attn_rhs, attn_softmax_logits_fixture, sizeof(attn_rhs));
  printf("F16SoftmaxFixture logits=file bytes=%zu\n", sizeof(attn_rhs));
#endif

  unsigned errors = 0;
  uint64_t reference_digest = 0;
  for (unsigned sample = 0; sample < 4; ++sample) {
    for (size_t elem = 0; elem < sizeof(attn_out) / sizeof(*attn_out); ++elem) attn_out[elem] = sentinel;
    for (size_t elem = 0; elem < sizeof(attn_softmax_work) / sizeof(*work); ++elem) attn_softmax_work[elem] = sentinel;
    unsigned saved, fcsr;
    uint64_t begin, end;
    __asm__ volatile("frcsr %0\n\tfscsr %2\n\tfence rw,rw\n\trdcycle %1"
                     : "=&r"(saved), "=r"(begin) : "r"(fcsr_seed) : "memory");
    ggml_compute_forward_soft_max(&params, &dst);
    __asm__ volatile("fence rw,rw\n\trdcycle %0\n\tfrcsr %1\n\tfscsr %2"
                     : "=&r"(end), "=&r"(fcsr) : "r"(saved) : "memory");

    unsigned bad = 0, guards = 0;
    uint64_t digest = UINT64_C(14695981039346656037);
    for (unsigned row = 0; row < attn_heads * attn_tokens; ++row) {
      const unsigned valid = row % attn_tokens + 1;
      const unsigned base = row * attn_tokens;
      float oracle[attn_tokens];
      double denominator = 0, output_sum = 0;
      if (sample == 3) {
        for (unsigned lane = 0; lane < valid; ++lane) {
          oracle[lane] = expf(attn_rhs[base + lane] * scale);
          denominator += oracle[lane];
        }
      }
      for (unsigned lane = 0; lane < attn_tokens; ++lane) {
        const float observed = output[base + lane];
        const uint32_t bits = fp32_to_bits(observed);
        digest = (digest ^ bits) * UINT64_C(1099511628211);
        if (lane >= valid) {
          bad += bits != 0;
        } else {
          bad += !(observed > 0 && observed <= 1);
          if (sample == 3) {
            const double expected = oracle[lane] / denominator;
            bad += !(fabs(observed - expected) <= 2e-8 + 2e-6 * expected);
          }
        }
        output_sum += observed;
      }
      bad += !(fabs(output_sum - 1) < 2e-6);
    }
    for (unsigned elem = 0; elem < attn_guard; ++elem) {
      guards += attn_out[elem] != sentinel || output[attn_rhs_elems + elem] != sentinel;
      guards += attn_softmax_work[elem] != sentinel || work[attn_tokens + CACHE_LINE_SIZE_F32 + elem] != sentinel;
    }
    const bool pass = end > begin && bad == 0 && guards == 0 && fcsr == expected_fcsr &&
        (sample == 0 || digest == reference_digest);
    reference_digest = digest;
    errors += !pass;
    printf("F16ModelSoftmax keys=%u tokens=%u heads=%u sample=%u phase=%s cycles=%llu outputs=%u "
           "masked=%u fcsr=%u digest=%016llx bad=%u canaries=%u %s\n", attn_tokens, attn_tokens, attn_heads,
           sample, sample == 0 ? "warmup" : "measure", (unsigned long long)(end - begin), attn_rhs_elems,
           attn_heads * attn_tokens * (attn_tokens - 1) / 2, fcsr, (unsigned long long)digest, bad, guards,
           pass ? "PASS" : "FAIL");
  }
  printf("F16ModelSoftmax %s errors=%u\n", errors == 0 ? "PASS" : "FAIL", errors);
  return errors != 0;
}
#endif

#if defined(F16_MODEL_ATTENTION_ROPE) && F16_MODEL_ATTENTION_ROPE
enum { rope_elems = attn_width * attn_heads * attn_tokens,
       rope_work_elems = attn_width + CACHE_LINE_SIZE / sizeof(float) };
static float attn_rope_input[rope_elems] __attribute__((aligned(64)));
static float attn_rope_work[rope_work_elems + 2 * attn_guard] __attribute__((aligned(64)));
static int32_t attn_rope_positions[attn_tokens] __attribute__((aligned(64)));

static struct ggml_tensor f16_attn_rope_tensor(float *data) {
  const size_t head_bytes = attn_width * sizeof(float);
  const size_t token_bytes = attn_heads * head_bytes;
  return (struct ggml_tensor){ .type = GGML_TYPE_F32, .data = data,
    .ne = { attn_width, attn_heads, attn_tokens, 1 },
    .nb = { sizeof(float), head_bytes, token_bytes, attn_tokens * token_bytes } };
}

static float f16_attn_rope_input(unsigned elem) {
  return (float)((int)((173 * elem + elem / attn_width) % 511) - 255) * 0x1p-8f;
}

static int run_f16_model_rope(void) {
  float *const output = attn_out + attn_guard;
  float *const work = attn_rope_work + attn_guard;
  const float sentinel = -12345.0f;
  const float rope_params[] = { 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f };
  const int32_t rope_ints[] = { 0, attn_width, GGML_ROPE_TYPE_NORMAL, 0, 128 };
  const float theta_scale = powf(rope_params[0], -2.0f / attn_width);
  struct ggml_tensor input = f16_attn_rope_tensor(attn_rope_input);
  struct ggml_tensor pos = { .type = GGML_TYPE_I32, .data = attn_rope_positions,
    .ne = { attn_tokens, 1, 1, 1 }, .nb = { sizeof(int32_t), sizeof(attn_rope_positions),
      sizeof(attn_rope_positions), sizeof(attn_rope_positions) } };
  struct ggml_tensor dst = f16_attn_rope_tensor(output);
  dst.op = GGML_OP_ROPE;
  dst.src[0] = &input;
  dst.src[1] = &pos;
  memcpy(dst.op_params, rope_ints, sizeof(rope_ints));
  memcpy(dst.op_params + 5, rope_params, sizeof(rope_params));
  const struct ggml_compute_params params = { .ith = 0, .nth = 1,
    .wdata = work, .wsize = rope_work_elems * sizeof(float),
    .threadpool = ggml_xsai_am_single_thread_pool() };

  for (unsigned index = 0; index < rope_elems; ++index) attn_rope_input[index] = f16_attn_rope_input(index);
  for (unsigned token = 0; token < attn_tokens; ++token) attn_rope_positions[token] = token;

  unsigned errors = 0;
  uint64_t reference_digest = 0;
  for (unsigned sample = 0; sample < 5; ++sample) {
    const bool fallback = sample == 4;
    attn_rope_positions[0] = fallback ? -1 : 0;
    for (unsigned elem = 0; elem < rope_elems + 2 * attn_guard; ++elem) attn_out[elem] = sentinel;
    for (unsigned elem = 0; elem < rope_work_elems + 2 * attn_guard; ++elem) attn_rope_work[elem] = sentinel;
    unsigned saved, fcsr;
    uint64_t begin, end;
    __asm__ volatile("frcsr %0\n\tfscsr zero\n\tfence rw,rw\n\trdcycle %1"
                     : "=&r"(saved), "=r"(begin) : : "memory");
    ggml_compute_forward_rope(&params, &dst);
    __asm__ volatile("fence rw,rw\n\trdcycle %0\n\tfrcsr %1\n\tfscsr %2"
                     : "=&r"(end), "=&r"(fcsr) : "r"(saved) : "memory");

    unsigned bad = 0, guards = 0;
    for (unsigned token = 0; token < attn_tokens; ++token) {
      const int32_t position = (fallback && token == 0) ? -1 : (int32_t)token;
      float theta = (float)position;
      bad += attn_rope_positions[token] != position;
      for (unsigned lane = 0; lane < attn_width; lane += 2) {
        const double cosine = cos((double)theta), sine = sin((double)theta);
        for (unsigned head = 0; head < attn_heads; ++head) {
          const unsigned index = (token * attn_heads + head) * attn_width + lane;
          const float first = f16_attn_rope_input(index), second = f16_attn_rope_input(index + 1);
          bad += attn_rope_input[index] != first || attn_rope_input[index + 1] != second;
          bad += !(fabs(output[index] - (first * cosine - second * sine)) <= 2e-7);
          bad += !(fabs(output[index + 1] - (first * sine + second * cosine)) <= 2e-7);
        }
        theta *= theta_scale;
      }
    }
    uint64_t digest = UINT64_C(14695981039346656037);
    for (unsigned index = 0; index < rope_elems; ++index) {
      digest = (digest ^ fp32_to_bits(output[index])) * UINT64_C(1099511628211);
    }
    for (unsigned elem = 0; elem < attn_guard; ++elem) {
      guards += attn_out[elem] != sentinel || output[rope_elems + elem] != sentinel;
      guards += attn_rope_work[elem] != sentinel || work[rope_work_elems + elem] != sentinel;
    }
    for (unsigned elem = attn_width; elem < rope_work_elems; ++elem) guards += work[elem] != sentinel;
    const bool pass = end > begin && bad == 0 && guards == 0 && fcsr == 1 &&
        (sample == 0 || (fallback ? digest != reference_digest : digest == reference_digest));
    if (sample == 0) reference_digest = digest;
    errors += !pass;
    printf("F16ModelRope width=%u heads=%u tokens=%u sample=%u phase=%s cycles=%llu outputs=%u "
           "fcsr=%u digest=%016llx bad=%u canaries=%u %s\n", attn_width, attn_heads, attn_tokens,
           sample, fallback ? "fallback" : sample == 0 ? "warmup" : "measure", (unsigned long long)(end - begin),
           rope_elems, fcsr, (unsigned long long)digest, bad, guards, pass ? "PASS" : "FAIL");
  }
  printf("F16ModelRope %s errors=%u\n", errors == 0 ? "PASS" : "FAIL", errors);
  return errors != 0;
}
#endif
#endif

int main(const char *args) {
  const char *mode = (args && args[0]) ? args : "128";

#if defined(F16_MODEL_ATTENTION) && F16_MODEL_ATTENTION
#if defined(F16_MODEL_ATTENTION_ROPE) && F16_MODEL_ATTENTION_ROPE
  if (strcmp(mode, "f16modelrope") == 0) return run_f16_model_rope();
#endif
#if defined(F16_MODEL_ATTENTION_SOFTMAX) && F16_MODEL_ATTENTION_SOFTMAX
  if (strcmp(mode, "f16modelsoftmax") == 0) return run_f16_model_softmax();
#endif
  if (strcmp(mode, "f16modelattn") == 0) return run_f16_model_attention(4);
  if (strcmp(mode, "f16attncheck") == 0) return run_f16_model_attention(1);
  printf("Unknown F16 model mainargs \"%s\"; expected f16modelattn\n", mode);
  return 1;
#elif defined(MXFP_MODEL_LMHEAD) && MXFP_MODEL_LMHEAD
  if (strcmp(mode, "mxlmhead") == 0) return run_mxfp_model_lmhead();
  printf("Unknown MXFP model mainargs \"%s\"; expected mxlmhead\n", mode);
  return 1;
#elif defined(MXFP_MODEL_WEIGHT) && MXFP_MODEL_WEIGHT
  if (strcmp(mode, "mxmodelweight") == 0) return run_mxfp_model_weight();
  printf("Unknown MXFP model mainargs \"%s\"; expected mxmodelweight\n", mode);
  return 1;
#elif defined(MXFP_QUANT_ONLY) && MXFP_QUANT_ONLY
#if defined(MXFP_MODEL_ACTIVATION) && MXFP_MODEL_ACTIVATION
  if (strcmp(mode, "mxmodelact") == 0) return run_mxfp_model_activation();
#endif
#if defined(MXFP_RMS_DYNAMIC) && MXFP_RMS_DYNAMIC
  if (strcmp(mode, "mxrmsdyn512") == 0) return run_mxrmsdyn512();
#endif
#if defined(MXFP_RMS_ONLY) && MXFP_RMS_ONLY
  if (strcmp(mode, "mxrms512") == 0) return run_mxrms512();
#endif
  if (strcmp(mode, "mxquant128") == 0) return run_mxquant128();
  if (strcmp(mode, "mxpreflight128") == 0) return run_mxpreflight128();
  printf("Unknown MXFP_QUANT_ONLY mainargs \"%s\"; expected "
#if defined(MXFP_RMS_DYNAMIC) && MXFP_RMS_DYNAMIC
         "{mxrmsdyn512, mxrms512, mxquant128, mxpreflight128}\n", mode);
#else
#if defined(MXFP_RMS_ONLY) && MXFP_RMS_ONLY
         "{mxrms512, mxquant128, mxpreflight128}\n", mode);
#else
#if defined(MXFP_MODEL_ACTIVATION) && MXFP_MODEL_ACTIVATION
         "{mxmodelact, mxquant128, mxpreflight128}\n", mode);
#else
         "{mxquant128, mxpreflight128}\n", mode);
#endif
#endif
#endif
  return 1;
#else
  int ret = 1;
  if (strcmp(mode, "64") == 0) {
    matrix_init_64();
    printf("GEMM Test started! mode=64 (64x64 * 64x64 -> 64x64)\n");
    ret = matrix_gemm_64();
  } else if (strcmp(mode, "128") == 0) {
    matrix_init_128();
    printf("GEMM Test started! mode=128 (128x64 * 64x128 -> 128x128)\n");
    ret = matrix_gemm_128();
  } else if (strcmp(mode, "mxfp8e4") == 0) {
    printf("GEMM Test started! MXFP8 E4M3, 32x32^T -> 1x1\n");
    ret = matrix_gemm_mxfp8_e4m3();
  } else if (strcmp(mode, "mxfp8e5") == 0) {
    printf("GEMM Test started! MXFP8 E5M2, 32x32^T -> 1x1\n");
    ret = matrix_gemm_mxfp8_e5m2();
  } else if (strcmp(mode, "mxfp4") == 0) {
    printf("GEMM Test started! MXFP4, 32x32^T -> 1x1\n");
    ret = matrix_gemm_mxfp4();
  } else if (strcmp(mode, "mxall") == 0) {
    printf("GEMM Test started! MX full-chain smoke (E4M3/E5M2/MXFP4)\n");
    const int e4 = matrix_gemm_mxfp8_e4m3();
    printf("MXFP8 E4M3: %s\n", e4 == 0 ? "PASS" : "FAIL");
    const int e5 = matrix_gemm_mxfp8_e5m2();
    printf("MXFP8 E5M2: %s\n", e5 == 0 ? "PASS" : "FAIL");
    const int fp4 = matrix_gemm_mxfp4();
    printf("MXFP4: %s\n", fp4 == 0 ? "PASS" : "FAIL");
    ret = e4 | e5 | fp4;
  } else if (strcmp(mode, "mxexact") == 0) {
    printf("GEMM Test started! MX vectors: 35 frozen exact cases + 9 K33/K64 ordering/overflow cases\n");
    ret = run_mx_cases(mode);
  } else if (strcmp(mode, "mxff") == 0) {
    printf("GEMM Test started! MX scale-FF directed vectors (66)\n");
    ret = run_scale_ff_cases(mode);
  } else if (strcmp(mode, "xmfrmtrap") == 0) {
    printf("GEMM Test started! MX xmfrm non-RNE trap/no-mutation\n");
    ret = run_xmfrm_traps(mode);
  } else if (strcmp(mode, "mxlife") == 0) {
    printf("GEMM Test started! MX flags/MINIT/scale lifecycle\n");
    ret = run_mx_lifecycle();
  } else if (strcmp(mode, "mxpanel") == 0) {
    printf("GEMM Test started! native-shaped MLA/MLB FP4 panel\n");
    ret = run_mx_panel();
  } else if (strcmp(mode, MXFP_PERF_MN == 512 ?
                    "mxperf512" : "mxperf128x128x512") == 0) {
#if defined(MXFP_CONVERT_LOAD) && MXFP_CONVERT_LOAD
    return run_mx_perf_mode(0);
  } else if (strcmp(mode, MXFP_PERF_MN == 512 ?
                    "mxcvt512" : "mxcvt128x128x512") == 0) {
    return run_mx_perf_mode(1);
  } else if (strcmp(mode, MXFP_PERF_MN == 512 ?
                    "mxcvt512reject" : "mxcvt128x128x512reject") == 0) {
    return run_mx_perf_mode(2);
#else
    return run_mx_perf();
#endif
#if defined(MXFP_CPU_QUANT) && MXFP_CPU_QUANT
  } else if (strcmp(mode, "mxruntime") == 0) {
    return run_mx_runtime();
  } else if (strcmp(mode, "mxassert") == 0) {
    return run_mx_assert();
  } else if (strcmp(mode, "mxquant128") == 0) {
    return run_mxquant128();
  } else if (strcmp(mode, "mxpreflight128") == 0) {
    return run_mxpreflight128();
#endif
  } else {
    printf("Unknown mainargs \"%s\"; expected {128, 64, mxfp8e4, mxfp8e5, mxfp4, mxall, mxexact, mxff, xmfrmtrap, mxlife, mxpanel, mxperf512, mxquant128, mxpreflight128}\n", mode);
    return 1;
  }

  if (ret == 0) {
    printf("GEMM Test passed!\n");
  } else {
    printf("GEMM Test failed!\n");
    print_failure_samples(mode);
  }
  return ret;
#endif
}
