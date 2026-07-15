
#include <klib.h>
#include <stdint.h>

// 验证结果，返回0表示正确，1表示错误
void matrix_init_128(void);
void matrix_init_64(void);
int matrix_ls_128(void);
int matrix_ls_64(void);
int matrix_ltst_128(void);
int matrix_ltst_64(void);
int matrix_lts_128(void);
int matrix_lts_64(void);
int matrix_lst_128(void);
int matrix_lst_64(void);
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
static void report_first_mismatch(const char *name, int mode64, int transposed) {
  int region_words = mode64 ? 4096 : 16384;
  int special_index = transposed ? (mode64 ? 64 : 128) : 1;

  for (int reg = 0; reg < 4; reg++) {
    int base = reg * region_words;
    for (int i = 0; i < region_words; i++) {
      uint32_t expect = (i == special_index) ? 2u : 0u;
      uint32_t actual = mat_dist[base + i];
      if (actual != expect) {
        printf("%s mismatch: reg=%d idx=%d expect=%u actual=%u\n",
               name, reg, i, expect, actual);
        return;
      }
    }
  }
  printf("%s mismatch not found in logical region\n", name);
}

int main(const char *args) {
  int mode64 = (args && strcmp(args, "64") == 0);

  if (args && args[0] && !mode64 && strcmp(args, "128") != 0) {
    printf("Unknown mainargs \"%s\"; expected {128, 64}\n", args);
    return 1;
  }

  if (mode64) {
    matrix_init_64();
    printf("Load/Store word mode=64\n");
  } else {
    matrix_init_128();
    printf("Load/Store word mode=128\n");
  }

  if ((mode64 ? matrix_ls_64() : matrix_ls_128()) == 0) {
    printf("Load/Store Test passed!\n");
  } else {
    printf("Load/Store Bad Test!!\n");
    report_first_mismatch("Load/Store", mode64, 0);
  }
  if ((mode64 ? matrix_ltst_64() : matrix_ltst_128()) == 0) {
    printf("LoadT/StoreT Test passed!\n");
  } else {
    printf("LoadT/StoreT Bad Test!!\n");
    report_first_mismatch("LoadT/StoreT", mode64, 0);
  }

  if ((mode64 ? matrix_lts_64() : matrix_lts_128()) == 0) {
    printf("LoadT/Store Test passed!\n");
  } else {
    printf("LoadT/Store Bad Test!!\n");
    report_first_mismatch("LoadT/Store", mode64, 1);
  }


  if ((mode64 ? matrix_lst_64() : matrix_lst_128()) == 0) {
    printf("Load/StoreT Test passed!\n");
  } else {
    printf("Load/StoreT Bad Test!!\n");
    report_first_mismatch("Load/StoreT", mode64, 1);
  }

  return 0;
}
