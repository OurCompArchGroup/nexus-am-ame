#include <klib.h>
#include <stdint.h>

void matrix_init_128(void);
void matrix_init_64(void);
void matrix_restore(void);
int matrix_mcfg_roundtrip(void);
int matrix_mcfg_behavior_128(void);
int matrix_mcfg_behavior_64(void);
int matrix_mfence_128(void);
int matrix_mfence_64(void);
int matrix_nofence_128(void);
int matrix_nofence_64(void);

extern uint8_t mat_na[];
extern uint32_t mat_dist[];
extern uint32_t mcfg_observed[];
extern uint32_t mcfg_expected[];

static void print_roundtrip_mismatch(void) {
  for (int i = 0; i < 8; i++) {
    if (mcfg_observed[i] != mcfg_expected[i]) {
      printf("mcfg%d expect=0x%08x actual=0x%08x\n",
             i, mcfg_expected[i], mcfg_observed[i]);
      return;
    }
  }
  printf("mcfg mismatch not found\n");
}

static void print_dist_heads(int region_words) {
  printf("dist[0]      = 0x%08x\n", mat_dist[0]);
  printf("dist[%d]  = 0x%08x\n", region_words, mat_dist[region_words]);
}

static void run_mcfg_roundtrip(void) {
  if (matrix_mcfg_roundtrip() == 0) {
    printf("MCFG_ROUNDTRIP Test passed!\n");
  } else {
    printf("MCFG_ROUNDTRIP Test failed!\n");
    print_roundtrip_mismatch();
  }
  matrix_restore();
}

static void run_dist_test(const char *name, int (*fn)(void), int region_words) {
  if (fn() == 0) {
    printf("%s Test passed!\n", name);
  } else {
    printf("%s Test failed!\n", name);
    print_dist_heads(region_words);
  }
  matrix_restore();
}

int main(const char *args) {
  int mode64 = (args && strcmp(args, "64") == 0);
  int region_words = mode64 ? 4096 : 16384;
  int na_bytes = mode64 ? 4096 : 8192;

  if (args && args[0] && !mode64 && strcmp(args, "128") != 0) {
    printf("Unknown mainargs \"%s\"; expected {128, 64}\n", args);
    return 1;
  }

  if (mode64) {
    matrix_init_64();
    printf("MFENCE/MCFG mode=64\n");
  } else {
    matrix_init_128();
    printf("MFENCE/MCFG mode=128\n");
  }

  run_mcfg_roundtrip();
  run_dist_test("MCFG_BEHAVIOR",
                mode64 ? matrix_mcfg_behavior_64 : matrix_mcfg_behavior_128,
                region_words);
  run_dist_test("MFENCE", mode64 ? matrix_mfence_64 : matrix_mfence_128,
                region_words);

  // Flip the logical A tile to 1 right before matrix_nofence.
  // Static init is -1, so a stale matrix-side read would yield 131; a fresh
  // read of the new value yields 1*(-2)*64 + 3 = -125.
  for (int i = 0; i < na_bytes; i++) {
    mat_na[i] = 1;
  }
  if ((mode64 ? matrix_nofence_64 : matrix_nofence_128)() == 0) {
    printf("PASSED even w/o mfence.\n");
  } else {
    printf("FAILED because of no mfence!\n");
    print_dist_heads(region_words);
  }

  return 0;
}
