
#include <klib.h>
#include <stdint.h>

void matrix_init();
int matrix_zero_verify(void);
int matrix_zero_tr_verify(void);
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

int main() {

  matrix_init();
  if (matrix_zero_verify() == 0) {
    printf("Zero1r for ACC(word) Test passed!\n");
  } else {
    printf("bad test!!\n");
    for (int i = 0; i < 16384; i++) {
      uint32_t val0 = mat_dist[i];
      uint32_t val1 = mat_dist[i+16384];
      uint32_t val2 = mat_dist[i+16384*2];
      uint32_t val3 = mat_dist[i+16384*3];
      if(val0!=0 || val1!=0 || val2!=0 || val3!=0){
        printf("val0=%d,val1=%d,val2=%d,val3=%d,i=%d\n",val0,val1,val2,val3,i);
      }
    }
    // printf("0x%08d ", ((const uint32_t *)mat_dist)[0]);
    // printf("0x%08d ", ((const uint32_t *)mat_dist)[16384]);
    // printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 2]);
    // printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 3]);
  }

  if (matrix_zero_tr_verify() == 0) {
      printf("Zero1r for TR(word) Test passed!\n");
  }else{
      printf("bad test!!\n");
      // print_matrix_part("Matrix ", mat_dist, 1, 8, 8);
  }
  return 0;
}
