#include <klib.h>
#include <stdint.h>

void matrix_init(void);
int  matrix_gemm(void);
extern uint32_t mat_dist[];

int main() {
  matrix_init();
  printf("GEMM Test started!\n");
  if (matrix_gemm() == 0) {
    printf("GEMM Test passed!\n");
  } else {
    printf("GEMM Test failed!\n");
    printf("dist[0]      = 0x%08x\n", mat_dist[0]);
    printf("dist[63]     = 0x%08x\n", mat_dist[63]);
    printf("dist[64]     = 0x%08x\n", mat_dist[64]);
    printf("dist[4095]   = 0x%08x\n", mat_dist[4095]);
  }
  return 0;
}
