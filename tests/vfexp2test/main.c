#include <klib.h>
#include <stdint.h>

#define FFLAGS_NV 0x10u
#define FFLAGS_OF 0x04u
#define FFLAGS_UF 0x02u
#define FFLAGS_NX 0x01u

#define FP16_IN_PATTERN 0x3c00, 0x0000, 0xfc00, 0x4000
#define FP16_OUT_PATTERN 0x4000, 0x3c00, 0x0000, 0x4400
#define FP32_IN_PATTERN 0x3f800000, 0x00000000, 0xff800000, 0x40000000
#define FP32_OUT_PATTERN 0x40000000, 0x3f800000, 0x00000000, 0x40800000

static const uint16_t fp16_basic_in[4] __attribute__((aligned(16))) = {
    FP16_IN_PATTERN};
static const uint16_t fp16_basic_expected[4] __attribute__((aligned(16))) = {
    FP16_OUT_PATTERN};
static uint16_t fp16_basic_out[4] __attribute__((aligned(16)));

static const uint16_t bf16_basic_in[4] __attribute__((aligned(16))) = {
    0x3f80, 0x0000, 0xff80, 0x4000};
static const uint16_t bf16_basic_expected[4] __attribute__((aligned(16))) = {
    0x4000, 0x3f80, 0x0000, 0x4080};
static uint16_t bf16_basic_out[4] __attribute__((aligned(16)));

static const uint32_t fp32_basic_in[4] __attribute__((aligned(16))) = {
    FP32_IN_PATTERN};
static const uint32_t fp32_basic_expected[4] __attribute__((aligned(16))) = {
    FP32_OUT_PATTERN};
static uint32_t fp32_basic_out[4] __attribute__((aligned(16)));

static const uint16_t fp16_flags_in[4] __attribute__((aligned(16))) = {
    0x7c01, 0x0001, 0x4c00, 0xce40};
static const uint16_t fp16_flags_expected[4] __attribute__((aligned(16))) = {
    0x7e00, 0x3c00, 0x7c00, 0x0000};
static uint16_t fp16_flags_out[4] __attribute__((aligned(16)));

static const uint16_t bf16_flags_in[4] __attribute__((aligned(16))) = {
    0x7f81, 0x0001, 0x4300, 0xc30c};
static const uint16_t bf16_flags_expected[4] __attribute__((aligned(16))) = {
    0x7fc0, 0x3f80, 0x7f80, 0x0000};
static uint16_t bf16_flags_out[4] __attribute__((aligned(16)));

static const uint32_t fp32_flags_in[4] __attribute__((aligned(16))) = {
    0x7f800001, 0x00000001, 0x43000000, 0xc3160000};
static const uint32_t fp32_flags_expected[4] __attribute__((aligned(16))) = {
    0x7fc00000, 0x3f800000, 0x7f800000, 0x00000000};
static uint32_t fp32_flags_out[4] __attribute__((aligned(16)));

static const uint16_t mask_old_vd[4] __attribute__((aligned(16))) = {
    0x1111, 0x2222, 0x3333, 0x4444};
static const uint16_t mask_expected[4] __attribute__((aligned(16))) = {
    0x7e00, 0x2222, 0x3333, 0x4444};
static const uint8_t mask_lane_zero[1] __attribute__((aligned(16))) = {0x01};
static uint16_t mask_out[4] __attribute__((aligned(16)));

static uint16_t pipe_fp16_basic_out[4] __attribute__((aligned(16)));
static uint16_t pipe_fp16_flags_out[4] __attribute__((aligned(16)));
static uint16_t pipe_bf16_basic_out[4] __attribute__((aligned(16)));
static uint16_t pipe_bf16_flags_out[4] __attribute__((aligned(16)));

static const uint16_t fp16_m8_in[64] __attribute__((aligned(128))) = {
    FP16_IN_PATTERN, FP16_IN_PATTERN, FP16_IN_PATTERN, FP16_IN_PATTERN,
    FP16_IN_PATTERN, FP16_IN_PATTERN, FP16_IN_PATTERN, FP16_IN_PATTERN,
    FP16_IN_PATTERN, FP16_IN_PATTERN, FP16_IN_PATTERN, FP16_IN_PATTERN,
    FP16_IN_PATTERN, FP16_IN_PATTERN, FP16_IN_PATTERN, FP16_IN_PATTERN};
static const uint16_t fp16_m8_expected[64] __attribute__((aligned(128))) = {
    FP16_OUT_PATTERN, FP16_OUT_PATTERN, FP16_OUT_PATTERN, FP16_OUT_PATTERN,
    FP16_OUT_PATTERN, FP16_OUT_PATTERN, FP16_OUT_PATTERN, FP16_OUT_PATTERN,
    FP16_OUT_PATTERN, FP16_OUT_PATTERN, FP16_OUT_PATTERN, FP16_OUT_PATTERN,
    FP16_OUT_PATTERN, FP16_OUT_PATTERN, FP16_OUT_PATTERN, FP16_OUT_PATTERN};
static uint16_t fp16_m8_out[64] __attribute__((aligned(128)));

static const uint32_t fp32_m8_in[32] __attribute__((aligned(128))) = {
    FP32_IN_PATTERN, FP32_IN_PATTERN, FP32_IN_PATTERN, FP32_IN_PATTERN,
    FP32_IN_PATTERN, FP32_IN_PATTERN, FP32_IN_PATTERN, FP32_IN_PATTERN};
static const uint32_t fp32_m8_expected[32] __attribute__((aligned(128))) = {
    FP32_OUT_PATTERN, FP32_OUT_PATTERN, FP32_OUT_PATTERN, FP32_OUT_PATTERN,
    FP32_OUT_PATTERN, FP32_OUT_PATTERN, FP32_OUT_PATTERN, FP32_OUT_PATTERN};
static uint32_t fp32_m8_out[32] __attribute__((aligned(128)));

static volatile uint32_t observed_fflags[10];
static volatile uint64_t observed_vl[2];
static volatile uint32_t vfexp2test_failures;

static inline void enable_vector(void) {
  asm volatile(
      "li t0, 0x200\n"
      "csrs mstatus, t0\n"
      :
      :
      : "t0", "memory");
}

static int check_u16(const uint16_t *got, const uint16_t *expected, int n) {
  int failures = 0;
  for (int i = 0; i < n; i++) {
    failures += got[i] != expected[i];
  }
  return failures;
}

static int check_u32(const uint32_t *got, const uint32_t *expected, int n) {
  int failures = 0;
  for (int i = 0; i < n; i++) {
    failures += got[i] != expected[i];
  }
  return failures;
}

static inline uint32_t run_fp16(
    const uint16_t *input, uint16_t *output, int bf16) {
  uint32_t flags;
  if (bf16) {
    asm volatile(
        "csrw fflags, x0\n"
        "li t0, 4\n"
        "vsetvli x0, t0, e16, m1, ta, ma\n"
        "vle16.v v1, (%1)\n"
        "vfexp2bf16.v v2, v1\n"
        "vse16.v v2, (%2)\n"
        "csrr %0, fflags\n"
        : "=r"(flags)
        : "r"(input), "r"(output)
        : "t0", "v1", "v2", "memory");
  } else {
    asm volatile(
        "csrw fflags, x0\n"
        "li t0, 4\n"
        "vsetvli x0, t0, e16, m1, ta, ma\n"
        "vle16.v v1, (%1)\n"
        "vfexp2.v v2, v1\n"
        "vse16.v v2, (%2)\n"
        "csrr %0, fflags\n"
        : "=r"(flags)
        : "r"(input), "r"(output)
        : "t0", "v1", "v2", "memory");
  }
  return flags & 0x1fu;
}

static inline uint32_t run_fp32(const uint32_t *input, uint32_t *output) {
  uint32_t flags;
  asm volatile(
      "csrw fflags, x0\n"
      "li t0, 4\n"
      "vsetvli x0, t0, e32, m1, ta, ma\n"
      "vle32.v v1, (%1)\n"
      "vfexp2.v v2, v1\n"
      "vse32.v v2, (%2)\n"
      "csrr %0, fflags\n"
      : "=r"(flags)
      : "r"(input), "r"(output)
      : "t0", "v1", "v2", "memory");
  return flags & 0x1fu;
}

static inline uint32_t run_mask(void) {
  uint32_t flags;
  asm volatile(
      "csrw vstart, x0\n"
      "csrw fflags, x0\n"
      "li t0, 4\n"
      "vsetvli x0, t0, e16, m1, tu, mu\n"
      "vle16.v v1, (%1)\n"
      "vle16.v v2, (%2)\n"
      "vlm.v v0, (%3)\n"
      "vfexp2.v v2, v1, v0.t\n"
      "vse16.v v2, (%4)\n"
      "csrr %0, fflags\n"
      : "=r"(flags)
      : "r"(fp16_flags_in), "r"(mask_old_vd), "r"(mask_lane_zero),
        "r"(mask_out)
      : "t0", "v0", "v1", "v2", "memory");
  return flags & 0x1fu;
}

static inline uint32_t run_pipeline_burst(void) {
  uint32_t flags;
  asm volatile(
      "csrw fflags, x0\n"
      "li t0, 4\n"
      "vsetvli x0, t0, e16, m1, ta, ma\n"
      "vle16.v v1, (%1)\n"
      "vle16.v v3, (%2)\n"
      "vle16.v v5, (%3)\n"
      "vle16.v v7, (%4)\n"
      "vfexp2.v v2, v1\n"
      "vfexp2.v v4, v3\n"
      "vfexp2bf16.v v6, v5\n"
      "vfexp2bf16.v v8, v7\n"
      "vse16.v v2, (%5)\n"
      "vse16.v v4, (%6)\n"
      "vse16.v v6, (%7)\n"
      "vse16.v v8, (%8)\n"
      "csrr %0, fflags\n"
      : "=r"(flags)
      : "r"(fp16_basic_in), "r"(fp16_flags_in), "r"(bf16_basic_in),
        "r"(bf16_flags_in), "r"(pipe_fp16_basic_out),
        "r"(pipe_fp16_flags_out), "r"(pipe_bf16_basic_out),
        "r"(pipe_bf16_flags_out)
      : "t0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
        "memory");
  return flags & 0x1fu;
}

static inline uint32_t run_fp16_m8(uint64_t *actual_vl) {
  uint32_t flags;
  asm volatile(
      "csrw fflags, x0\n"
      "li t0, 64\n"
      "vsetvli %1, t0, e16, m8, tu, mu\n"
      "vle16.v v8, (%2)\n"
      "vfexp2.v v16, v8\n"
      "vse16.v v16, (%3)\n"
      "csrr %0, fflags\n"
      : "=r"(flags), "=&r"(*actual_vl)
      : "r"(fp16_m8_in), "r"(fp16_m8_out)
      : "t0", "v8", "v16", "memory");
  return flags & 0x1fu;
}

static inline uint32_t run_fp32_m8(uint64_t *actual_vl) {
  uint32_t flags;
  asm volatile(
      "csrw fflags, x0\n"
      "li t0, 32\n"
      "vsetvli %1, t0, e32, m8, tu, mu\n"
      "vle32.v v8, (%2)\n"
      "vfexp2.v v16, v8\n"
      "vse32.v v16, (%3)\n"
      "csrr %0, fflags\n"
      : "=r"(flags), "=&r"(*actual_vl)
      : "r"(fp32_m8_in), "r"(fp32_m8_out)
      : "t0", "v8", "v16", "memory");
  return flags & 0x1fu;
}

int main(void) {
  const uint32_t all_exception_flags =
      FFLAGS_NV | FFLAGS_OF | FFLAGS_UF | FFLAGS_NX;
  int failures = 0;

  enable_vector();

  observed_fflags[0] = run_fp16(fp16_basic_in, fp16_basic_out, 0);
  observed_fflags[1] = run_fp16(bf16_basic_in, bf16_basic_out, 1);
  observed_fflags[2] = run_fp32(fp32_basic_in, fp32_basic_out);
  observed_fflags[3] = run_fp16(fp16_flags_in, fp16_flags_out, 0);
  observed_fflags[4] = run_fp16(bf16_flags_in, bf16_flags_out, 1);
  observed_fflags[5] = run_fp32(fp32_flags_in, fp32_flags_out);
  observed_fflags[6] = run_mask();
  observed_fflags[7] = run_pipeline_burst();
  observed_fflags[8] = run_fp16_m8((uint64_t *)&observed_vl[0]);
  observed_fflags[9] = run_fp32_m8((uint64_t *)&observed_vl[1]);

  failures += check_u16(fp16_basic_out, fp16_basic_expected, 4);
  failures += check_u16(bf16_basic_out, bf16_basic_expected, 4);
  failures += check_u32(fp32_basic_out, fp32_basic_expected, 4);
  failures += check_u16(fp16_flags_out, fp16_flags_expected, 4);
  failures += check_u16(bf16_flags_out, bf16_flags_expected, 4);
  failures += check_u32(fp32_flags_out, fp32_flags_expected, 4);
  failures += check_u16(mask_out, mask_expected, 4);
  failures += check_u16(pipe_fp16_basic_out, fp16_basic_expected, 4);
  failures += check_u16(pipe_fp16_flags_out, fp16_flags_expected, 4);
  failures += check_u16(pipe_bf16_basic_out, bf16_basic_expected, 4);
  failures += check_u16(pipe_bf16_flags_out, bf16_flags_expected, 4);
  failures += check_u16(fp16_m8_out, fp16_m8_expected, 64);
  failures += check_u32(fp32_m8_out, fp32_m8_expected, 32);

  failures += observed_fflags[0] != 0;
  failures += observed_fflags[1] != 0;
  failures += observed_fflags[2] != 0;
  failures += observed_fflags[3] != all_exception_flags;
  failures += observed_fflags[4] != all_exception_flags;
  failures += observed_fflags[5] != all_exception_flags;
  failures += observed_fflags[6] != FFLAGS_NV;
  failures += observed_fflags[7] != all_exception_flags;
  failures += observed_fflags[8] != 0;
  failures += observed_fflags[9] != 0;
  failures += observed_vl[0] != 64;
  failures += observed_vl[1] != 32;

  vfexp2test_failures = (uint32_t)failures;
  printf("[VFEXP2-CI] %s failures=%d\n", failures == 0 ? "PASS" : "FAIL",
         failures);
  return failures != 0;
}
