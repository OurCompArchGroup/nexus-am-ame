
#include <klib.h>
#include <stdint.h>

// 验证结果，返回0表示正确，1表示错误
void matrix_init();
int matrix_zero_verify(void);
int matrix_zero_tr_verify(void);
extern uint8_t mat_dist[];
void *get_mat_dist(void);

#define Len 16
void print_matrix_part(const char *name, const void *data, int elem_size,
                       int rows, int cols) {
  printf("\n%s ( %dx%d elements):\n", name, rows < Len ? rows : Len,
         cols < Len ? cols : Len);

  for (int i = 0; i < (rows < Len ? rows : Len); i++) {
    for (int j = 0; j < (cols < Len ? cols : Len); j++) {
      if (elem_size == 1) {
        uint8_t val = ((const uint8_t *)data)[i * cols + j];
        printf("0x%02x ", val);
      } else if (elem_size == 4) {
        uint32_t val = ((const uint32_t *)data)[i * cols + j];
        printf("0x%08x ", val);
      }
    }
    printf("\n");
  }
}

int main() {
  matrix_init();
  if (matrix_zero_verify() == 0) {
    printf("Zero1r for ACC Test passed!\n");
  } else {
    printf("bad test!!\n");
    // print_matrix_part("Matrix ", mat_dist, 1, 8, 8);
    printf("0x%08d ", ((const uint8_t *)mat_dist)[0]);
    printf("0x%08d ", ((const uint8_t *)mat_dist)[16384]);
    printf("0x%08d ", ((const uint8_t *)mat_dist)[16384 * 2]);
    printf("0x%08d ", ((const uint8_t *)mat_dist)[16384 * 3]);
    // for (int i = 0; i < 16384 * 4; i++) {
    //   uint8_t val = ((const uint8_t *)mat_dist)[i];
    //   if (val != 0) {
    //     printf("val=%d i=%d", val, i);
    //     break;
    //   }
    // }
  }

  // if (matrix_zero_tr_verify() == 0) {
  //   printf("Zero1r for TR Test passed!\n");
  // } else {
  //   printf("bad test!!\n");

  //   printf("0x%08d ", ((const uint8_t *)mat_dist)[0]);
  //   printf("0x%08d ", ((const uint8_t *)mat_dist)[16384]);
  //   printf("0x%08d ", ((const uint8_t *)mat_dist)[16384 * 2]);
  //   printf("0x%08d ", ((const uint8_t *)mat_dist)[16384 * 3]);
  //   // print_matrix_part("Matrix ", mat_dist, 1, 8, 8);
  // }
  //  print_matrix_part("Matrix", mat_dist, 1, 8, 8);

  return 0;
}
