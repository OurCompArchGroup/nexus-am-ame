#include <klib.h>
#include <stdint.h>

void matrix_init(void);
void matrix_restore(void);
int matrix_mmacc0(void);
int matrix_mmacc1(void);
int matrix_mmacc2(void);
int matrix_mmacc3(void);
int matrix_mmacc4(void);
extern uint32_t mat_dist[];

static void run(const char *name, int (*fn)(void)) {
  if (fn() == 0) {
    printf("%s Test passed!\n", name);
  } else {
    printf("%s bad test!!\n", name);
    printf("0x%08d ", ((const uint32_t *)mat_dist)[0]);
    printf("0x%08d ", ((const uint32_t *)mat_dist)[16384]);
    printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 2]);
    printf("0x%08d\n", ((const uint32_t *)mat_dist)[16384 * 3]);
  }
  matrix_restore();
}

int main() {
  matrix_init();
  run("MMACC0", matrix_mmacc0);
  run("MMACC1", matrix_mmacc1);
  run("MMACC2", matrix_mmacc2);
  run("MMACC3", matrix_mmacc3);
  run("MMACC4", matrix_mmacc4);
  return 0;
}
