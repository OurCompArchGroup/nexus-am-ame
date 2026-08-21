#include <klib.h>
#include <stdint.h>

// Whole-register load/store test driver (ref: ame-ls-word).
//
// Each test loads a whole matrix register from mat_src, stores it back to
// mat_dist, and the assembly routine compares the two over the architectural
// register size, which it reads from the size CSRs (tlenb for A/B tile regs,
// alenb for the C acc reg). Tile and acc whole images differ on this target:
//
//   * tile register (tr0-tr3): 128 x 64 x 1 B = 8192 B   (tlenb)
//   * acc  register (acc0-acc3): 128 x 128 x 4 B = 65536 B (alenb)
//
// The shared source/destination buffers must therefore cover the LARGER of the
// two (the acc image); the tile tests only touch the first 8192 bytes.
#define TILE_BYTES 8192             // whole tile register image
#define ACC_BYTES  65536            // whole acc  register image
#define BUF_BYTES  ACC_BYTES        // buffers sized to the largest whole reg
#define SENTINEL   0x5a

void matrix_init_128(void);
void matrix_init_64(void);
int matrix_whole_c(void);
int matrix_whole_a(void);
int matrix_whole_b(void);
int matrix_whole_indep_128(void);
int matrix_whole_indep_64(void);

extern uint8_t mat_src[];
extern uint8_t mat_dist[];

// Fill the source image with a byte pattern in which every position is easy to
// distinguish, so a truncated or misplaced whole transfer is caught.
static void fill_src(void) {
  for (int i = 0; i < BUF_BYTES; i++) {
    mat_src[i] = (uint8_t)(i * 31 + 7);
  }
}

// Reset the destination image to a sentinel so any byte the whole store fails
// to write shows up as a mismatch against the source pattern.
static void reset_dist(void) {
  for (int i = 0; i < BUF_BYTES; i++) {
    mat_dist[i] = SENTINEL;
  }
}

// Report the first differing byte within the register's compared range.
static void report_first_diff(int nbytes) {
  for (int i = 0; i < nbytes; i++) {
    if (mat_dist[i] != mat_src[i]) {
      printf("first diff at byte %d: dist=0x%02x src=0x%02x\n", i, mat_dist[i],
             mat_src[i]);
      return;
    }
  }
  printf("no byte difference found in compared range\n");
}

// nbytes = whole-register image size for this test (TILE_BYTES or ACC_BYTES),
// used only to bound the failure diagnostic; the assembly compare length comes
// from the size CSRs.
static void run(const char *name, int (*fn)(void), int nbytes) {
  reset_dist();
  if (fn() == 0) {
    printf("%s Test passed!\n", name);
  } else {
    printf("%s Bad Test!!\n", name);
    report_first_diff(nbytes);
  }
}

int main(const char *args) {
  int mode64 = (args && strcmp(args, "64") == 0);

  if (args && args[0] && !mode64 && strcmp(args, "128") != 0) {
    printf("Unknown mainargs \"%s\"; expected {128, 64}\n", args);
    return 1;
  }

  fill_src();
  if (mode64) {
    matrix_init_64();
    printf("Whole load/store mode=64\n");
  } else {
    matrix_init_128();
    printf("Whole load/store mode=128\n");
  }

  run("Whole C (mlc.whole/msc.whole)", matrix_whole_c, ACC_BYTES);
  run("Whole A (mla.whole/msa.whole)", matrix_whole_a, TILE_BYTES);
  run("Whole B (mlb.whole/msb.whole)", matrix_whole_b, TILE_BYTES);
  run("Whole tile-independent store",
      mode64 ? matrix_whole_indep_64 : matrix_whole_indep_128, ACC_BYTES);

  return 0;
}
