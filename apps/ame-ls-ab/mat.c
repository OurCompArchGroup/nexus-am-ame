
#include <klib.h>
#include <stdint.h>

// 验证结果，返回0表示正确，1表示错误
void matrix_init(void);
int matrix_ab(void);
int matrix_atb(void);
int matrix_abt(void);
int matrix_atbt(void);

void restore_mat(void);
extern uint32_t mat_dist[];

void mat_dist_check(void) {
  for (int base = 0; base < 16384 * 4; base += 16384) {
    int old_val = mat_dist[base];
    int start = 0;
    for (int i = 1; i < 16384; i++) {
      if (mat_dist[base + i] != old_val) {
        // interval [start, i) has all values as old_val
        printf("mat_dist[%d, %d) = %d\n", start, i, old_val);
        start = i;
        old_val = mat_dist[base + i];
      }
    }
    // last interval [start, 16384) has all values as old_val
    printf("mat_dist[%d, %d) = %d\n", start, 16384, mat_dist[base + 16383]);
  }
}

int main() {
  matrix_init();

  if (matrix_ab() == 0) {
    printf("Load A/B Test passed!\n");

  } else {
    printf("Load A/B Test failed!!\n");
    mat_dist_check();
  }
  restore_mat();
  if (matrix_atb() == 0) {
    printf("Load AT/B Test passed!\n");
  } else {
    printf("Load AT/B Test failed!!\n");
    mat_dist_check();
  }
  restore_mat();
  if (matrix_abt() == 0) {
    printf("Load AB/T Test passed!\n");
  } else {
    printf("Load AB/T Test failed!!!\n");
    mat_dist_check();
  }
  restore_mat();
  if (matrix_atbt() == 0) {
    printf("Load AT/BT Test passed!\n");
  } else {
    printf("Load AT/BT Test failed!!!\n");
    mat_dist_check();
  }
  restore_mat();

  return 0;
}
