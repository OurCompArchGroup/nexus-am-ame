#include <klib.h>
#include <stdint.h>

void matrix_init_128(void);
void matrix_init_64(void);
void matrix_restore(void);
int matrix_mmacc0_128(void);
int matrix_mmacc0_64(void);
int matrix_mmacc1_128(void);
int matrix_mmacc1_64(void);
int matrix_mmacc2_128(void);
int matrix_mmacc2_64(void);
int matrix_mmacc3_128(void);
int matrix_mmacc3_64(void);
int matrix_mmacc4_128(void);
int matrix_mmacc4_64(void);
extern uint32_t mat_dist[];

static void print_region_heads(int region_words) {
  printf("dist[0]      = 0x%08x\n", mat_dist[0]);
  printf("dist[%d]  = 0x%08x\n", region_words, mat_dist[region_words]);
  printf("dist[%d]  = 0x%08x\n", region_words * 2, mat_dist[region_words * 2]);
  printf("dist[%d]  = 0x%08x\n", region_words * 3, mat_dist[region_words * 3]);
}

static void run(const char *name, int (*fn)(void), int region_words) {
  if (fn() == 0) {
    printf("%s Test passed!\n", name);
  } else {
    printf("%s bad test!!\n", name);
    print_region_heads(region_words);
  }
  matrix_restore();
}

int main(const char *args) {
  int mode64 = (args && strcmp(args, "64") == 0);
  int region_words = mode64 ? 4096 : 16384;

  if (args && args[0] && !mode64 && strcmp(args, "128") != 0) {
    printf("Unknown mainargs \"%s\"; expected {128, 64}\n", args);
    return 1;
  }

  if (mode64) {
    matrix_init_64();
    printf("MMACC mode=64\n");
  } else {
    matrix_init_128();
    printf("MMACC mode=128\n");
  }

  run("MMACC0", mode64 ? matrix_mmacc0_64 : matrix_mmacc0_128, region_words);
  run("MMACC1", mode64 ? matrix_mmacc1_64 : matrix_mmacc1_128, region_words);
  run("MMACC2", mode64 ? matrix_mmacc2_64 : matrix_mmacc2_128, region_words);
  run("MMACC3", mode64 ? matrix_mmacc3_64 : matrix_mmacc3_128, region_words);
  run("MMACC4", mode64 ? matrix_mmacc4_64 : matrix_mmacc4_128, region_words);
  return 0;
}
