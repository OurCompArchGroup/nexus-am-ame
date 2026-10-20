#include "ame-ew.h"

static unsigned checks;
static unsigned cases;
static const char *current_case;

void case_begin(const char *label) {
  current_case = label;
  cases++;
  printf("[CASE %u] %s\n", cases, label);
}

void check_u32(const char *label, unsigned line, uint32_t got, uint32_t expected) {
  checks++;
  if (got != expected) {
    printf("FAIL case=%s check=%s line=%u got=%08x expected=%08x\n",
           current_case, label, line, got, expected);
    _halt(1);
  }
}

void check_close(const char *label, unsigned line, float got, float expected,
                 float tolerance) {
  checks++;
  float diff = got - expected;
  if (diff < 0) diff = -diff;
  if (!(diff <= tolerance)) {
    printf("FAIL case=%s check=%s line=%u got_bits=%08x expected_bits=%08x tolerance_bits=%08x\n",
           current_case, label, line, f32_bits(got), f32_bits(expected), f32_bits(tolerance));
    _halt(1);
  }
}

int main(const char *args) {
  (void)args;
  uintptr_t enabled = (1ul << 25) | (1ul << 13) | (1ul << 9);
  asm volatile("csrs mstatus, %0" :: "r"(enabled) : "memory");
  AME_CSR_WRITE(0x809, 0); /* matrix RNE */
  AME_CSR_WRITE(0x80a, 0); /* integer wrap */
  AME_CSR_WRITE(0x003, 0); /* scalar fcsr */
  ame_flags_clear();
  printf("AME EW/Reduce positive examples: default 128x128, 32 LV\n");
  test_ew();
  test_sfu();
  test_reduce();
  test_memory();
  test_convert();
  test_operators();
  printf("AME_EW_RESULT cases=%u checks=%u failures=0 signature=4557504f\n", cases, checks);
  return 0;
}
