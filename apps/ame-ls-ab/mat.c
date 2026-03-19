
#include <klib.h>
#include <stdint.h>

// 验证结果，返回0表示正确，1表示错误
void matrix_init(void);
int matrix_ab(void);
int matrix_atb(void);
// int matrix_abt(void);
// int matrix_abtt(void);
// void matrix_restore(void);
extern uint32_t mat_dist[];
// extern uint8_t mat_a[];
// extern  uint32_t mat_target[];
// extern uint32_t mat_target_t[];

// #define Len 16
// void print_matrix_part(const char* name, const void* data, int elem_size, int
// rows, int cols) {
//     printf("\n%s (first %dx%d elements):\n", name, rows < Len ? rows : Len,
//     cols < Len ? cols : Len);

//     for (int i = 0; i < (rows < Len ? rows : Len); i++) {
//         for (int j = 0; j < (cols < Len ? cols : Len); j++) {
//             if (elem_size == 1) {
//                 uint8_t val = ((const uint8_t*)data)[i * cols + j];
//                 printf("0x%02x ", val);
//             } else if (elem_size == 4) {
//                 uint32_t val = ((const uint32_t*)data)[i * cols + j];
//                 printf("0x%08x ", val);
//             }
//         }
//         printf("\n");
//     }
// }

int main() {
  matrix_init();
  // matrix_restore();

  if (matrix_ab() == 0) {
    printf("Load A/B Test passed!\n");

  } else {
    printf("bad test!!\n");
  }
  if (matrix_atb() == 0) {
    printf("Load AT/B Test passed!\n");

  } else {
    printf("bad test!!\n");
    for(int i=0;i<163;i++){
      int val=((const int32_t *)mat_dist)[i];
    //   if (val!=131){
        printf("C[%d] = %d\n",i,val);
    //   }
    }
  }

  return 0;
}
