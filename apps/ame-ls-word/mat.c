
#include <klib.h>
#include <stdint.h>

// 验证结果，返回0表示正确，1表示错误
void matrix_init(void);
int matrix_ls(void);
int matrix_ltst(void);
int matrix_lts(void);
int matrix_lst(void);
extern uint32_t mat_dist[];
extern uint32_t mat_a_t[];

void print_matrix_part(const char *name, const void *data, int elem_size,
                       int rows, int cols) {
  printf("\n%s (first %dx%d elements):\n", name, rows < 4 ? rows : 4,
         cols < 4 ? cols : 4);

  for (int i = 0; i < (rows < 4 ? rows : 4); i++) {
    for (int j = 0; j < (cols < 4 ? cols : 4); j++) {
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
void checkload() {
  for (int i = 0; i < 16384; i++) {
    int val = ((const int32_t *)mat_dist)[i];
    if (mat_dist[i] != 0 && i != 1) {
      printf("ls:%d %d\n", i, val);
    }
    if (i == 1 && mat_dist[i] != 2) {
      printf("ls:%d %d\n", i, val);
    }
  }
  for (int i = 0; i < 16384; i++) {
    int val = ((const int32_t *)mat_dist)[i + 16384];
    if (mat_dist[i] != 0 && i != 1) {
      printf("ls:%d %d\n", i, val);
    }
    if (i == 1 && mat_dist[i] != 2) {
      printf("ls:%d %d\n", i, val);
    }
  }
  for (int i = 0; i < 16384; i++) {
    int val = ((const int32_t *)mat_dist)[i + 16384 * 2];
    if (mat_dist[i] != 0 && i != 1) {
      printf("ls:%d %d\n", i, val);
    }
    if (i == 1 && mat_dist[i] != 2) {
      printf("ls:%d %d\n", i, val);
    }
  }
  for (int i = 0; i < 16384; i++) {
    int val = ((const int32_t *)mat_dist)[i + 16384 * 3];
    if (mat_dist[i] != 0 && i != 1) {
      printf("ls:%d %d\n", i, val);
    }
    if (i == 1 && mat_dist[i] != 2) {
      printf("ls:%d %d\n", i, val);
    }
  }
}
void checkload2() {

  for (int i = 0; i < 16384; i++) {
    int val = ((const int32_t *)mat_dist)[i];
    if (val != 0 && i != 128) {
      printf("ls:%d %d\n", i, val);
    }
    if (i == 128 && val != 2) {
      printf("ls:%d %d\n", i, val);
    }
  }

  for (int i = 0; i < 16384; i++) {
    int val = ((const int32_t *)mat_dist)[i + 16384];
    if (val != 0 && i != 128) {
      printf("ls:%d %d\n", i, val);
    }
    if (i == 128 && val != 2) {
      printf("ls:%d %d\n", i, val);
    }
  }
  for (int i = 0; i < 16384; i++) {
    int val = ((const int32_t *)mat_dist)[i + 16384 * 2];
    if (val != 0 && i != 128) {
      printf("ls:%d %d\n", i, val);
    }
    if (i == 128 && val != 2) {
      printf("ls:%d %d\n", i, val);
    }
  }
  for (int i = 0; i < 16384; i++) {
    int val = ((const int32_t *)mat_dist)[i + 16384 * 3];
    if (val != 0 && i != 128) {
      printf("ls:%d %d\n", i, val);
    }
    if (i == 128 && val != 2) {
      printf("ls:%d %d\n", i, val);
    }
  }
}
int main() {
  // printf("Test Load/Store\n");
  matrix_init();
  // print_matrix_part("Matrix A (initial)", mat_a, 1, 4, 4);
  if (matrix_ls() == 0) {
    printf("Load/Store Test passed!\n");
  } else {
    printf("Load/Store Bad Test!!\n");
    checkload();
  }
  if (matrix_ltst() == 0) {
    printf("LoadT/StoreT Test passed!\n");
  } else {
    printf("LoadT/StoreT Bad Test!!\n");
    checkload();
  }

  if (matrix_lts() == 0) {
    printf("LoadT/Store Test passed!\n");
  } else {
    printf("LoadT/Store Bad Test!!\n");
    checkload2();
  }


    if (matrix_lst() == 0) {
        printf("Load/StoreT Test passed!\n");
    }else{
        printf("Load/StoreT Bad Test!!\n");
        checkload2();
    }

  return 0;
}
