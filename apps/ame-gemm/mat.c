#include <klib.h>
#include <stdint.h>

void matrix_init(void);
int  matrix_gemm(void);
extern uint32_t mat_dist[];

int main() {
  matrix_init();
  if (matrix_gemm() == 0) {
    printf("GEMM Test passed!\n");
  } else {
    printf("GEMM Test failed!\n");
    printf("dist[0]      = 0x%08x\n", mat_dist[0]);
    printf("dist[127]    = 0x%08x\n", mat_dist[127]);
    printf("dist[128]    = 0x%08x\n", mat_dist[128]);
    printf("dist[32768]  = 0x%08x\n", mat_dist[32768]);
    printf("dist[32896]  = 0x%08x\n", mat_dist[32896]);
  }
  return 0;
}
