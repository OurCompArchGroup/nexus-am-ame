
#include <klib.h>
#include <stdint.h>
void matrix_init(void);
void matrix_restore(void);

int matrix_mfmacc0(void);
int matrix_mfmacc1(void);
int matrix_mfmacc2(void);
int matrix_mfmacc3(void);
int matrix_mfmacc4(void);
extern uint64_t mat_dist[];
extern uint64_t mat_target[];
extern uint64_t mat_target_n[];
#define Len 16
void print_matrix_part(const char *name, const void *data, int elem_size,
                       int rows, int cols) {
  printf("\n%s (first %dx%d elements):\n", name, rows < Len ? rows : Len,
         cols < Len ? cols : Len);

  for (int i = 0; i < (rows < Len ? rows : Len); i++) {
    for (int j = 0; j < (cols < Len ? cols : Len); j++) {
      if (elem_size == 1) {
        uint8_t val = ((const uint8_t *)data)[i * cols + j];
        printf("0x%02x ", val);
      } else if (elem_size == 4) {
        uint32_t val = ((const uint32_t *)data)[i * cols + j];
        printf("0x%08x ", val);
      } else if (elem_size == 8) {
        uint64_t val = ((const uint64_t *)data)[i * cols + j];
        printf("0x%016lx ", val);
      }
    }
    printf("\n");
  }
}

// 验证结果，返回0表示正确，1表示错误
// int matrix_verify(void);

int main() {
  matrix_init();

  if (matrix_mfmacc0() == 0) {
    printf("MFMACC0 Test passed!\n");
  } else {
    printf("MFMACC0 bad test!!\n");
    print_matrix_part("Matrix destination", mat_dist, 8, 4, 4);
    print_matrix_part("Matrix target", mat_target, 8, 4, 4);
  }
  matrix_restore();

  if (matrix_mfmacc1() == 0) {
    printf("MFMACC1 Test passed!\n");
  } else {
    printf("MFMACC1 bad test!!\n");
    print_matrix_part("Matrix destination", mat_dist, 8, 4, 4);
    print_matrix_part("Matrix target", mat_target_n, 8, 4, 4);
  }
  matrix_restore();

  if (matrix_mfmacc2() == 0) {
    printf("MFMACC2 Test passed!\n");
  } else {
    printf("MFMACC2 bad test!!\n");
    print_matrix_part("Matrix destination", mat_dist, 8, 4, 4);
    print_matrix_part("Matrix target", mat_target_n, 8, 4, 4);
  }
  matrix_restore();

  if (matrix_mfmacc3() == 0) {
    printf("MFMACC3 Test passed!\n");
  } else {
    printf("MFMACC3 bad test!!\n");
    print_matrix_part("Matrix destination", mat_dist, 8, 4, 4);
    print_matrix_part("Matrix target", mat_target, 8, 4, 4);
  }
  matrix_restore();

  if (matrix_mfmacc4() == 0) {
    printf("MFMACC4 Test passed!\n");
  } else {
    printf("MFMACC4 bad test!!\n");
    print_matrix_part("Matrix destination", mat_dist, 8, 4, 4);
    print_matrix_part("Matrix target", mat_target, 8, 4, 4);
  }
  matrix_restore();

  return 0;
}
