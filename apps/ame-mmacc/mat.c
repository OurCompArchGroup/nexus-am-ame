
#include <klib.h>
#include <stdint.h>
void matrix_init(void);
void matrix_restore(void);
int matrix_mmacc0(void);
int matrix_mmacc1(void);
int matrix_mmacc2(void);
int matrix_mmacc3(void);
int matrix_mmacc4(void);
extern uint32_t mat_dist[];
// extern  uint32_t mat_target[];


// 验证结果，返回0表示正确，1表示错误
// int matrix_verify(void);

int main() {
  matrix_init();
  if (matrix_mmacc0() == 0) {
    printf("MMACC0 Test passed!\n");
  } else {
    printf("bad test!!\n");

    printf("0x%08d ", ((const uint32_t *)mat_dist)[0]);
    printf("0x%08d ", ((const uint32_t *)mat_dist)[16384]);
    printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 2]);
    printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 3]);
  }
  // matrix_restore();
  matrix_restore();
  if (matrix_mmacc1() == 0) {
    printf("MMACC1 Test passed!\n");
  } else {
    printf("MMACC1 bad test!!\n");

    for(int i=0;i<16384;i++){
      int val=((const int32_t *)mat_dist)[i];
      if (val!=131){
        printf("mmaccu.w.b:%d %d\n",i,val);
      }
    }
    for(int i=0;i<16384;i++){
      int val=((const int32_t *)mat_dist)[i+16384];
      if (val!=4145283){
        printf("mmaccu.w.b:%d %d\n",i,val);
      }
    }
      for(int i=0;i<16384;i++){
      int val=((const int32_t *)mat_dist)[i+16384*2];
      if (val!=-16253){
        printf("mmaccsu.w.b:%d %d",i,val);
      }
    }
    for(int i=0;i<16384;i++){
      int val=((const int32_t *)mat_dist)[i+16384*3];
      if (val!=-32637){
        printf("mmaccus.w.b:i=%d val=%d\n",i,val);
      }
    }

    // printf("0x%08d ", ((const uint32_t *)mat_dist)[0]);
    // printf("0x%08d ", ((const uint32_t *)mat_dist)[16384]);
    // printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 2]);
    // printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 3]);
  }
  // matrix_restore();

  // if (matrix_mmacc2() == 0) {
  //   printf("MMACC2 Test passed!\n");
  // } else {
  //   printf("MMACC2 bad test!!\n");
    
  //   for(int i=0;i<16384;i++){
  //     int val=((const int32_t *)mat_dist)[i];
  //     if (val!=131){
  //       printf("mmaccu.w.b:%d %d\n",i,val);
  //     }
  //   }
  //   for(int i=0;i<16384;i++){
  //     int val=((const int32_t *)mat_dist)[i+16384];
  //     if (val!=4145283){
  //       printf("mmaccu.w.b:%d %d\n",i,val);
  //     }
  //   }
  //     for(int i=0;i<16384;i++){
  //     int val=((const int32_t *)mat_dist)[i+16384*2];
  //     if (val!=-16253){
  //       printf("mmaccsu.w.b:%d %d",i,val);
  //     }
  //   }
  //   for(int i=0;i<16384;i++){
  //     int val=((const int32_t *)mat_dist)[i+16384*3];
  //     if (val!=-32637){
  //       printf("mmaccus.w.b:i=%d val=%d\n",i,val);
  //     }
  //   }

  //   // printf("0x%08d ", ((const uint32_t *)mat_dist)[0]);
  //   // printf("0x%08d ", ((const uint32_t *)mat_dist)[16384]);
  //   // printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 2]);
  //   // printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 3]);
  // }
  // matrix_restore();

  // if (matrix_mmacc3() == 0) {
  //   printf("MMACC3 Test passed!\n");
  // } else {
  //   printf("bad test!!\n");

  //   printf("0x%08d ", ((const uint32_t *)mat_dist)[0]);
  //   printf("0x%08d ", ((const uint32_t *)mat_dist)[16384]);
  //   printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 2]);
  //   printf("0x%08d ", ((const uint32_t *)mat_dist)[16384 * 3]);
  // }
  // matrix_restore();
  
  // if (matrix_mmacc4() == 0) {
  //   printf("MMACC4 Test passed!\n");
  // } else {
  //   printf("bad test!!\n");

  //   printf("0x%08d ", ((const uint32_t *)mat_dist)[0]);
  // }
  // matrix_restore();



  return 0;
}
