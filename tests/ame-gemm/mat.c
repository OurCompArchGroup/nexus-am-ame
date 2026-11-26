#include <klib.h>
#include <stdint.h>

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
void matrix_test_mxsemantics(uint64_t *observed);
uint32_t matrix_test_mxscale_regs(uint64_t allregs);
typedef struct {
  uint64_t rows, tile_k, k_tiles, payload_stride, scale_stride, out_stride;
} MxPanelConfig;
uint64_t matrix_test_mxpanel(const uint8_t *a, const uint8_t *b,
                           const uint8_t *sa, const uint8_t *sb, uint32_t *c,
                           const MxPanelConfig *config);
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
      uint8_t sa[2] = {side == 1 ? 127 : 255};
      uint8_t sb[2] = {side == 0 ? 127 : 255};
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

static int run_mx_semantics(void) {
  // Slots are documented beside matrix_test_mxsemantics in matop.S.
  uint64_t observed[17];
  for (unsigned i = 0; i < sizeof(observed) / sizeof(observed[0]); ++i) {
    observed[i] = UINT64_MAX;
  }
  matrix_test_mxsemantics(observed);

  const int unpub_a = observed[0] == 0x12345678u &&
      observed[1] == 2 && observed[2] == 1;
  const int unpub_b = observed[3] == 0x12345678u &&
      observed[4] == 2 && observed[5] == 1;
  const int fp4_k96 = observed[6] == 0x43040000u &&
      observed[7] == UINT64_MAX && observed[8] == 0;
  const int fp8_k96_reject = observed[9] == 2 && observed[10] == 1;
  const int unaligned_convert = observed[11] == 4 && observed[13] == 1 &&
      observed[12] == observed[14] && (observed[12] & 3) == 1 &&
      observed[16] == observed[15];
  const int failed = !unpub_a || !unpub_b || !fp4_k96 ||
      !fp8_k96_reject || !unaligned_convert;

  printf("MXSEM unpublished-A result=0x%08x cause=%llu traps=%llu %s\n",
         (uint32_t)observed[0], (unsigned long long)observed[1],
         (unsigned long long)observed[2], unpub_a ? "PASS" : "FAIL");
  printf("MXSEM unpublished-B result=0x%08x cause=%llu traps=%llu %s\n",
         (uint32_t)observed[3], (unsigned long long)observed[4],
         (unsigned long long)observed[5], unpub_b ? "PASS" : "FAIL");
  printf("MXSEM fp4-K96 result=0x%08x cause=0x%llx traps=%llu %s\n",
         (uint32_t)observed[6], (unsigned long long)observed[7],
         (unsigned long long)observed[8], fp4_k96 ? "PASS" : "FAIL");
  printf("MXSEM fp8-K96-reject cause=%llu traps=%llu %s\n",
         (unsigned long long)observed[9], (unsigned long long)observed[10],
         fp8_k96_reject ? "PASS" : "FAIL");
  printf("MXSEM unaligned-convert cause=%llu tval=0x%llx base=0x%llx "
         "mepc=0x%llx pc=0x%llx traps=%llu %s\n",
         (unsigned long long)observed[11],
         (unsigned long long)observed[12], (unsigned long long)observed[14],
         (unsigned long long)observed[16], (unsigned long long)observed[15],
         (unsigned long long)observed[13],
         unaligned_convert ? "PASS" : "FAIL");
  printf("MX semantics: %s (5/5)\n", failed ? "FAIL" : "PASS");
  return failed;
}

static int run_mxscale_regs(uint64_t allregs) {
  const uint32_t expected = allregs ? 0x42b00000u : 0x42300000u;
  printf("MXSCALE_REGS begin sequence=%s B=B1 mmacc_ts1=%u mmacc_ts2=1 "
         "rows=1 K=32 wait=macquire_sync3\n",
         allregs ? "A0,A1,A2,A3" : "A0,A1,A0", allregs ? 3u : 0u);
  const uint32_t actual = matrix_test_mxscale_regs(allregs);
  const int failed = actual != expected;
  printf("MXSCALE_%s RETURNED result=%d %s actual=0x%08x expected=0x%08x\n",
         allregs ? "ALLREGS" : "CONTROL", failed, failed ? "FAIL" : "PASS",
         actual, expected);
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
  const MxPanelConfig config = {128, 32, 1, 16, 1, 128 * sizeof(float)};
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

static int run_mx_scale_stride(unsigned rows, unsigned stride) {
  static volatile uint8_t a[4 * 16] __attribute__((aligned(64)));
  static volatile uint8_t b[4 * 16] __attribute__((aligned(64)));
  static volatile uint8_t sa[4 * 6] __attribute__((aligned(64)));
  static volatile uint8_t sb[4 * 6] __attribute__((aligned(64)));
  static volatile uint32_t c[4 * 4] __attribute__((aligned(64)));
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned byte = 0; byte < 16; ++byte) {
      a[row * 16 + byte] = 0x21;
      b[row * 16 + byte] = 0x43;
    }
    for (unsigned byte = 0; byte < stride; ++byte) {
      sa[row * stride + byte] = byte == 0 ? 127 : 0xee;
      sb[row * stride + byte] = byte == 0 ? 127 : 0xee;
    }
  }
  for (unsigned i = 0; i < rows * rows; ++i) c[i] = 0x7fc56789;
  const MxPanelConfig config = {rows, 32, 1, 16, stride, rows * sizeof(float)};
  const uint64_t flags = matrix_test_mxpanel((const uint8_t *)a,
      (const uint8_t *)b, (const uint8_t *)sa, (const uint8_t *)sb,
      (uint32_t *)c, &config);
  unsigned mismatches = 0;
  for (unsigned i = 0; i < rows * rows; ++i) {
    mismatches += c[i] != 0x42300000u;
  }
  const int failed = mismatches != 0 || flags != 0;
  printf("MXScaleStride M=%u N=%u K=32 row_bytes=1 stride=%u outputs=%u "
         "mismatches=%u flags=0x%02x %s\n", rows, rows, stride, rows * rows,
         mismatches, (unsigned)flags,
         failed ? "FAIL" : "PASS");
  return failed;
}

static void print_failure_samples(const char *mode) {
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
}

int main(const char *args) {
  const char *mode = (args && args[0]) ? args : "128";
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
  } else if (strcmp(mode, "mxsemantics") == 0) {
    printf("GEMM Test started! MX scale publication, FP4 capacity, and alignment\n");
    ret = run_mx_semantics();
  } else if (strcmp(mode, "mxscale-allregs") == 0) {
    printf("GEMM Test started! MX scale register liveness\n");
    ret = run_mxscale_regs(1);
  } else if (strcmp(mode, "mxscale-control") == 0) {
    printf("GEMM Test started! MX scale register same-ID control\n");
    ret = run_mxscale_regs(0);
  } else if (strcmp(mode, "mxlife") == 0) {
    printf("GEMM Test started! MX flags/MINIT/scale lifecycle\n");
    ret = run_mx_lifecycle();
  } else if (strcmp(mode, "mxpanel") == 0) {
    printf("GEMM Test started! native-shaped MLA/MLB FP4 panel\n");
    ret = run_mx_panel();
  } else if (strcmp(mode, "mxscalestride") == 0) {
    printf("GEMM Test started! padded noncompact scale row stride\n");
    ret = run_mx_scale_stride(2, 6);
  } else if (strcmp(mode, "mxscalestride1") == 0) {
    printf("GEMM Test started! single-row padded scale row stride\n");
    ret = run_mx_scale_stride(1, 6);
  } else {
    printf("Unknown mainargs \"%s\"; expected {128, 64, mxfp8e4, mxfp8e5, mxfp4, "
           "mxall, mxexact, mxff, xmfrmtrap, mxsemantics, mxscale-allregs, "
           "mxscale-control, mxlife, mxpanel, mxscalestride, mxscalestride1}\n", mode);
    return 1;
  }

  if (ret == 0) {
    printf("GEMM Test passed!\n");
  } else {
    printf("GEMM Test failed!\n");
    if (strcmp(mode, "128") == 0 || strcmp(mode, "64") == 0) {
      print_failure_samples(mode);
    }
  }
  return ret;
}
