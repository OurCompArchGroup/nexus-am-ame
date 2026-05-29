#include <klib.h>
#include <stdint.h>

void matrix_init(void);
void matrix_restore(void);
int matrix_mcfg_roundtrip(void);
int matrix_mcfg_behavior(void);
int matrix_mfence(void);
int matrix_nofence(void);

extern uint8_t mat_na[];
extern uint32_t mat_dist[];
extern uint32_t mcfg_observed[];
extern uint32_t mcfg_expected[];

static void run(const char *name, int (*fn)(void)) {
  if (fn() == 0) {
    printf("%s Test passed!\n", name);
  } else {
    printf("%s Test failed!\n", name);
  }
  matrix_restore();
}

int main() {
  matrix_init();
  run("MCFG_ROUNDTRIP", matrix_mcfg_roundtrip);
  run("MCFG_BEHAVIOR", matrix_mcfg_behavior);
  run("MFENCE", matrix_mfence);
  // Flip the whole 8192-byte mat_na tile to 1 right before matrix_nofence.
  // Static init is -1, so a stale matrix-side read would yield 131; a fresh
  // read of the new value yields 1*(-2)*64 + 3 = -125. matrix_nofence checks
  // for -125, so missing scalar->matrix visibility surfaces as a fail.
  for (int i = 0; i < 8192; i++) {
    mat_na[i] = 1;
  }
  if (matrix_nofence() == 0) {
    printf("PASSED even w/o mfence.\n");
  } else {
    printf("FAILED because of no mfence!\n");
  }

  return 0;
}
