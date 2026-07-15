#include <klib.h>
#include <stdint.h>

void matrix_init_128(void);
int  matrix_gemm_128(void);
void matrix_init_64(void);
int  matrix_gemm_64(void);
extern uint32_t mat_dist[];

static void print_failure_samples(const char *mode) {
  printf("dist[0]      = 0x%08x\n", mat_dist[0]);
  printf("dist[63]     = 0x%08x\n", mat_dist[63]);
  if (strcmp(mode, "64") == 0) {
    printf("dist[256]    = 0x%08x\n", mat_dist[256]);
    printf("dist[16191]  = 0x%08x\n", mat_dist[16191]);
  } else {
    printf("dist[64]     = 0x%08x\n", mat_dist[64]);
    printf("dist[127]    = 0x%08x\n", mat_dist[127]);
    printf("dist[128]    = 0x%08x\n", mat_dist[128]);
    printf("dist[32768]  = 0x%08x\n", mat_dist[32768]);
    printf("dist[32896]  = 0x%08x\n", mat_dist[32896]);
  }
}

int main(const char *args) {
  const char *mode = (args && args[0]) ? args : "128";
  int ret = 1;

  if (strcmp(mode, "64") == 0) {
    matrix_init_64();
    printf("GEMM Test started! mode=64 (64x64 * 64x64 -> 64x64)\n");
    ret = matrix_gemm_64();
  } else if (strcmp(mode, "128") == 0) {
    matrix_init_128();
    printf("GEMM Test started! mode=128 (128x64 * 64x128 -> 128x128)\n");
    ret = matrix_gemm_128();
  } else {
    printf("Unknown mainargs \"%s\"; expected {128, 64}\n", mode);
    return 1;
  }

  if (ret == 0) {
    printf("GEMM Test passed!\n");
  } else {
    printf("GEMM Test failed!\n");
    print_failure_samples(mode);
  }
  return ret;
}
