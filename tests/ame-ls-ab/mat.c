
#include <klib.h>
#include <stdint.h>

// 验证结果，返回0表示正确，1表示错误
void matrix_init_128(void);
void matrix_init_64(void);
int matrix_ab_128(void);
int matrix_ab_64(void);
int matrix_atb_128(void);
int matrix_atb_64(void);
int matrix_abt_128(void);
int matrix_abt_64(void);
int matrix_atbt_128(void);
int matrix_atbt_64(void);

void restore_mat(void);
extern uint32_t mat_dist[];

void mat_dist_check(int region_words) {
  for (int base = 0; base < region_words * 4; base += region_words) {
    int old_val = mat_dist[base];
    int start = 0;
    for (int i = 1; i < region_words; i++) {
      if (mat_dist[base + i] != old_val) {
        // interval [start, i) has all values as old_val
        printf("mat_dist[%d, %d) = %d\n", start, i, old_val);
        start = i;
        old_val = mat_dist[base + i];
      }
    }
    printf("mat_dist[%d, %d) = %d\n", start, region_words,
           mat_dist[base + region_words - 1]);
  }
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
    printf("Load A/B mode=64\n");
  } else {
    matrix_init_128();
    printf("Load A/B mode=128\n");
  }

  if ((mode64 ? matrix_ab_64() : matrix_ab_128()) == 0) {
    printf("Load A/B Test passed!\n");

  } else {
    printf("Load A/B Test failed!!\n");
    mat_dist_check(region_words);
  }
  restore_mat();
  if ((mode64 ? matrix_atb_64() : matrix_atb_128()) == 0) {
    printf("Load AT/B Test passed!\n");
  } else {
    printf("Load AT/B Test failed!!\n");
    mat_dist_check(region_words);
  }
  restore_mat();
  if ((mode64 ? matrix_abt_64() : matrix_abt_128()) == 0) {
    printf("Load AB/T Test passed!\n");
  } else {
    printf("Load AB/T Test failed!!!\n");
    mat_dist_check(region_words);
  }
  restore_mat();
  if ((mode64 ? matrix_atbt_64() : matrix_atbt_128()) == 0) {
    printf("Load AT/BT Test passed!\n");
  } else {
    printf("Load AT/BT Test failed!!!\n");
    mat_dist_check(region_words);
  }
  restore_mat();

  return 0;
}
