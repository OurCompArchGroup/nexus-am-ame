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
extern uint32_t mat_target[];
extern uint32_t mat_target_double[];

#define PRINT_LIMIT 4

static void print_matrix_part(const char *name, const uint32_t *data) {
  printf("\n%s (first %dx%d elements):\n", name, PRINT_LIMIT, PRINT_LIMIT);
  for (int i = 0; i < PRINT_LIMIT; i++) {
    for (int j = 0; j < PRINT_LIMIT; j++) {
      printf("0x%08x ", data[i * 128 + j]);
    }
    printf("\n");
  }
}

static int run_case(const char *name, int (*test)(void), const uint32_t *target) {
  int result = test();
  if (result == 0) {
    printf("%s passed!\n", name);
  } else {
    printf("%s failed!\n", name);
    print_matrix_part("Matrix destination", mat_dist);
    print_matrix_part("Matrix target", target);
  }
  matrix_restore();
  return result;
}

int main() {
  matrix_init();
  int failures = 0;
  failures += run_case("MMACC FP8 E4M3 -> FP32 case 0", matrix_mmacc0, mat_target);
  failures += run_case("MMACC FP8 E4M3 -> FP32 case 1", matrix_mmacc1, mat_target);
  failures += run_case("MMACC FP8 E4M3 -> FP32 case 2", matrix_mmacc2, mat_target);
  failures += run_case("MMACC FP8 E4M3 -> FP32 case 3", matrix_mmacc3, mat_target);
  failures += run_case("MMACC FP8 E4M3 -> FP32 double case", matrix_mmacc4,
                       mat_target_double);
  return failures == 0 ? 0 : 1;
}
