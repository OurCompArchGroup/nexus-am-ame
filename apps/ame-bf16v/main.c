#include "bf16.h"
#include <klib.h>
#include <math.h>
#include <stdint.h>

/*
** The value in the BASE field of mtvec
** must always be aligned on a 4-byte boundary
** aligned(4) means aligned on a 4-byte boundary
** not aligned on a 2^4 byte boundary
*/
__attribute__((aligned(4))) void __am_asm_trap(void) {
  asm volatile("csrr t0, mepc\n\t"
               "addi t0, t0, 4\n\t"
               "csrw mepc, t0\n\t"
               "mret");
}
#define INIT()                                                                 \
  do {                                                                         \
    asm volatile("lui a0,0x2\n"                                                \
                 "addiw a0,a0,512\n"                                           \
                 "csrs mstatus,a0\n"                                           \
                 "csrwi vcsr,0" ::);                                           \
  } while (0)

// Test data structure for BF16 conversion tests
typedef struct {
  float input;       // Input float value
  uint16_t expected; // Expected BF16 value
  const char *desc;  // Description of the test case
} test_case_t;

// Test data for BF16 conversion
static test_case_t test_cases[] = {
    // Normal numbers
    {1.0f, 0x3F80, "1.0"},
    {0.0f, 0x0000, "0.0"},
    {-1.0f, 0xBF80, "-1.0"},
    {0.5f, 0x3F00, "0.5"},
    {-0.5f, 0xBF00, "-0.5"},
    {2.0f, 0x4000, "2.0"},
    {-2.0f, 0xC000, "-2.0"},
    {100.0f, 0x42C8, "100.0"},
    {-100.0f, 0xC2C8, "-100.0"},
    {0.0001f, 0x38D2, "0.0001"},
    {-0.0001f, 0xB8D2, "-0.0001"},
    {100000.0f, 0x47C3, "100000.0"},
    {-100000.0f, 0xC7C3, "-100000.0"},

    // Special values
    {INFINITY, 0x7F80, "+inf"},
    {-INFINITY, 0xFF80, "-inf"},
    {NAN, 0x7FC0, "NaN"},

    // Denormal numbers
    {1.40129846e-45f, 0x0000, "Smallest positive denormal"},
    {1.0e-45f, 0x0000, "Rounds to 0"},
    {-1.40129846e-45f, 0x8000, "Smallest negative denormal"},

    // Edge cases
    {0.0078125f, 0x3C00, "2^-7"},
    {65504.0f, 0x4780, "Largest normal number"},
};

static int num_test_cases = sizeof(test_cases) / sizeof(test_cases[0]);

// Test vfncvtbf16.f.f.w (Vector convert FP32 to BF16)
void test_vfncvtbf16() {
  printf("Testing vfncvtbf16.f.f.w (Vector convert FP32 to BF16):\n");

  // Test 1: Basic conversion
  printf("\nTest 1: Basic conversion\n");

  // Test with vector length 8
  float test_values[4] = {};
  uint16_t results[4] = {3, 3, 3, 3};

  // int idx = 0;
  for (int base = 0; base + 4 < num_test_cases; base += 4) {
    for (int i = base; i < base + 4; i++) {
      test_values[i - base] = test_cases[i].input;
    }
    asm volatile("csrwi frm,0 \n"
                 "vsetivli zero, 4, e32, m1, ta, ma\n"
                 "vle32.v v2, (%0)\n"
                 "vsetivli zero, 4, e16, m1, ta, ma\n"
                 "vfncvtbf16.f.f.w v0, v2\n"
                 "vse16.v v0, (%1)\n"
                 "fence rw, rw\n"
                 :
                 : "r"(test_values), "r"(results)
                 : "v0", "v1", "v2");
    // Check results
    for (int i = base; i < base + 4; i++) {
      uint16_t expected = test_cases[i].expected;
      if (results[i - base] == expected) {
        printf("%sPASS%s: %g -> 0x%04x\n", COLOR_GREEN, COLOR_RESET,
               test_values[i - base], results[i - base]);
      } else {
        printf("%sFAIL%s: %g -> 0x%04x, expected 0x%04x\n", COLOR_RED,
               COLOR_RESET, test_values[i - base], results[i - base], expected);
      }
    }
  }
  // Test 2: Masked operation
  printf("\nTest 2: Masked operation\n");

  float masked_values[4] = {test_cases[0].input, test_cases[1].input,
                            test_cases[2].input, test_cases[3].input};
  uint16_t masked_results[4] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};
  uint8_t mask[4] = {1, 0, 1, 0};
  uint8_t mask_bits[] = {(mask[3] << 3) | (mask[2] << 2) | (mask[1] << 1) |
                         mask[0]};
  asm volatile("csrwi frm,0 \n"
               "vsetivli zero, 4, e16, m1, tu, mu\n"
               "vle16.v v2, (%2)\n"
               "vsetivli zero, 4, e32, m1, tu, mu\n"
               "vle32.v v4, (%0)\n"
               "vsetivli zero, 4, e8, m1, tu, mu\n"
               "vlm.v v0, (%1)\n"
               "vsetivli zero, 4, e16, m1, tu, mu\n"
               "vfncvtbf16.f.f.w v2, v4, v0.t\n"
               "vse16.v v2, (%2)\n"
               "fence rw, rw\n"
               :
               : "r"(masked_values), "r"(mask_bits), "r"(masked_results)
               : "v0", "v1", "v2", "v4", "v5");

  for (int i = 0; i < 4; i++) {
    if (mask[i]) {
      uint16_t expected = test_cases[i].expected;
      if (masked_results[i] == expected) {
        printf("%sPASS%s: [masked] %g -> 0x%04x\n", COLOR_GREEN, COLOR_RESET,
               masked_values[i], masked_results[i]);
      } else {
        printf("%sFAIL%s: [masked] %g -> 0x%04x, expected 0x%04x\n", COLOR_RED,
               COLOR_RESET, masked_values[i], masked_results[i], expected);
      }
    } else {
      if (masked_results[i] == 0xFFFF) {
        printf("%sPASS%s: [unmasked] %g -> 0x%04x (unchanged)\n", COLOR_GREEN,
               COLOR_RESET, masked_values[i], masked_results[i]);
      } else {
        printf("%sFAIL%s: [unmasked] %g -> 0x%04x, expected 0xFFFF\n",
               COLOR_RED, COLOR_RESET, masked_values[i], masked_results[i]);
      }
    }
  }

  printf("\nvfncvtbf16.f.f.w test completed\n\n");
}

// Test vfwcvtbf16.f.f.v (Vector convert BF16 to FP32)
void test_vfwcvtbf16() {
  printf("Testing vfwcvtbf16.f.f.v (Vector convert BF16 to FP32):\n");

  // Test 1: Basic conversion
  printf("\nTest 1: Basic conversion\n");

  uint16_t bf16_values[4];
  float results[4];

  for (int base = 0; base + 4 < num_test_cases; base += 4) {
    for (int i = base; i < base + 4; i++) {
      bf16_values[i - base] = test_cases[i].expected;
    }
    asm volatile("csrwi frm,0 \n"
                 "vsetivli zero, 4, e16, m1, ta, ma\n"
                 "vle16.v v2, (%0)\n"
                 "vfwcvtbf16.f.f.v v0, v2\n"
                 "vsetivli zero, 4, e32, m1, ta, ma\n"
                 "vse32.v v0, (%1)\n"
                 "fence rw, rw\n"
                 :
                 : "r"(bf16_values), "r"(results)
                 : "v0", "v1", "v2", "v3"); // v2-v3 for EMUL=2 with e32

    // Check results
    for (int i = base; i < base + 4; i++) {
      float expected = test_cases[i].input;
      float_bits result_bits;
      result_bits.f = results[i - base];
      if (float_equal(results[i - base], expected, 1e-6f, 1e-5f)) {
        printf("%sPASS%s: 0x%04x -> %g\n", COLOR_GREEN, COLOR_RESET,
               bf16_values[i - base], results[i - base]);
      } else {
        float expected2 = bf16_to_float(bf16_values[i - base]);
        if (float_equal(results[i - base], expected2, 1e-6f, 1e-5f)) {
          printf("%sPASS%s: 0x%04x -> %g~%g\n", COLOR_GREEN, COLOR_RESET,
                 bf16_values[i - base], results[i - base], expected);
        } else {
          printf("%sFAIL%s: 0x%04x -> %g/%08x, expected %g\n", COLOR_RED,
                 COLOR_RESET, bf16_values[i - base], results[i - base],
                 result_bits.u, expected2);
        }
      }
    }
  }

  // Test 2: Masked operation
  printf("\nTest 2: Masked operation\n");

  uint16_t masked_bf16[4] = {test_cases[0].expected, test_cases[1].expected,
                             test_cases[2].expected, test_cases[3].expected};
  float masked_results[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
  uint8_t mask[4] = {1, 0, 1, 0};
  uint8_t mask_bits[] = {(mask[3] << 3) | (mask[2] << 2) | (mask[1] << 1) |
                         mask[0]};

  asm volatile("csrwi frm,0 \n"
               "vsetivli zero, 4, e32, m2, tu, mu\n"
               "vle32.v v4, (%2)\n"
               "vsetivli zero, 4, e8, m1, tu, mu\n"
               "vlm.v v0, (%1)\n"
               "vsetivli zero, 4, e16, m1, tu, mu\n"
               "vle16.v v2, (%0)\n"
               "vfwcvtbf16.f.f.v v4, v2, v0.t\n"
               "vsetivli zero, 4, e32, m1, ta, ma\n"
               "vse32.v v4, (%2)\n"
               "fence rw, rw\n"
               :
               : "r"(masked_bf16), "r"(mask_bits), "r"(masked_results)
               : "v0", "v1", "v2", "v4", "v5");

  for (int i = 0; i < 4; i++) {
    if (mask[i]) {
      float expected = test_cases[i].input;
      if (float_equal(masked_results[i], expected, 1e-6f, 1e-5f)) {
        printf("%sPASS%s: [masked] 0x%04x -> %g\n", COLOR_GREEN, COLOR_RESET,
               masked_bf16[i], masked_results[i]);
      } else {
        printf("%sFAIL%s: [masked] 0x%04x -> %g, expected %g\n", COLOR_RED,
               COLOR_RESET, masked_bf16[i], masked_results[i], expected);
      }
    } else {
      if (masked_results[i] == -1.0f) {
        printf("%sPASS%s: [unmasked] 0x%04x -> %g (unchanged)\n", COLOR_GREEN,
               COLOR_RESET, masked_bf16[i], masked_results[i]);
      } else {
        printf("%sFAIL%s: [unmasked] 0x%04x -> %g, expected -1.0\n", COLOR_RED,
               COLOR_RESET, masked_bf16[i], masked_results[i]);
      }
    }
  }

  printf("vfwcvtbf16.f.f.v test completed\n\n");
}

int main() {
  printf("=== BF16 Vector Function Tests ===\n\n");

  asm volatile("csrw mtvec, %0" : : "r"(__am_asm_trap));
  INIT();

  // Run tests
  test_vfncvtbf16();
  test_vfwcvtbf16();


  printf("=== All Tests Completed ===\n");
  return 0;
}
