
#include <klib.h>
#include <stdint.h>

void matrix_init_128(void);
void matrix_init_64(void);
int matrix_zero_verify_128(void);
int matrix_zero_verify_64(void);
int matrix_zero_tr_verify_128(void);
int matrix_zero_tr_verify_64(void);
extern uint32_t mat_dist[];
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

static void print_zero_mismatch(int region_words) {
  for (int i = 0; i < region_words; i++) {
    uint32_t val0 = mat_dist[i];
    uint32_t val1 = mat_dist[i + region_words];
    uint32_t val2 = mat_dist[i + region_words * 2];
    uint32_t val3 = mat_dist[i + region_words * 3];
    if (val0 != 0 || val1 != 0 || val2 != 0 || val3 != 0) {
      printf("val0=%u,val1=%u,val2=%u,val3=%u,i=%d\n",
             val0, val1, val2, val3, i);
      return;
    }
  }
  printf("zero mismatch not found in logical region\n");
}

static void print_tr_zero_mismatch(int check_words) {
  for (int i = 0; i < check_words; i++) {
    if (mat_dist[i] != 0) {
      printf("tr zero mismatch: idx=%d val=%u\n", i, mat_dist[i]);
      return;
    }
  }
  printf("tr zero mismatch not found in checked region\n");
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
    printf("Zero word mode=64\n");
  } else {
    matrix_init_128();
    printf("Zero word mode=128\n");
  }

  if ((mode64 ? matrix_zero_verify_64() : matrix_zero_verify_128()) == 0) {
    printf("Zero1r for ACC(word) Test passed!\n");
  } else {
    printf("bad test!!\n");
    print_zero_mismatch(region_words);
  }

  if ((mode64 ? matrix_zero_tr_verify_64() : matrix_zero_tr_verify_128()) == 0) {
      printf("Zero1r for TR(word) Test passed!\n");
  }else{
      printf("bad test!!\n");
      print_tr_zero_mismatch(16384);
  }
  return 0;
}
