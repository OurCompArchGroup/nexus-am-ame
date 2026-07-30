#include <klib.h>
#include <stdint.h>

/*
 * This workload checks an A/B transpose load without msa/msb.  The loaded
 * MatrixReg is deliberately re-typed as int8 after the load; an int8 identity
 * multiply then exposes every raw byte through the supported msc path.
 * Build one precision at a time with TRANSPOSE_PROBE_MASK=1/2/4/8.
 */
#ifndef TRANSPOSE_PROBE_MASK
#define TRANSPOSE_PROBE_MASK 1u
#endif

enum {
  LoadMajorDim = 128,
  ComputeK = 64,
  ComputeN = 128,
  ProbeBytes = 64,
};

#if TRANSPOSE_PROBE_MASK == 1
enum { OriginalBytes = 1, OriginalK = 64, SourceStride = 128 };
#define TEST_NAME "e8"
#elif TRANSPOSE_PROBE_MASK == 2
enum { OriginalBytes = 2, OriginalK = 32, SourceStride = 256 };
#define TEST_NAME "e16"
#elif TRANSPOSE_PROBE_MASK == 4
enum { OriginalBytes = 4, OriginalK = 16, SourceStride = 512 };
#define TEST_NAME "e32"
#elif TRANSPOSE_PROBE_MASK == 8
enum { OriginalBytes = 0, OriginalK = 127, SourceStride = 64 };
#define TEST_NAME "e4"
#else
#error "TRANSPOSE_PROBE_MASK must be 1, 2, 4, or 8"
#endif

enum {
  SourceBytes = OriginalK * SourceStride,
  IdentityBytes = ComputeN * ComputeK,
  AResultElements = LoadMajorDim * ComputeN,
  BResultElements = LoadMajorDim * ComputeN,
};

/* The assembly uses these symbols as memory operands. */
uint8_t a_source[SourceBytes] __attribute__((aligned(64)));
uint8_t b_source[SourceBytes] __attribute__((aligned(64)));
uint8_t identity[IdentityBytes] __attribute__((aligned(64)));
uint32_t a_result[AResultElements] __attribute__((aligned(64)));
uint32_t b_result[BResultElements] __attribute__((aligned(64)));

extern void matrix_init(void);
extern void transpose_probe(void);

#if TRANSPOSE_PROBE_MASK != 8
static uint8_t byte_pattern(uint32_t matrix_tag, uint32_t source_row,
                            uint32_t major, uint32_t byte_in_element) {
  /* Distinct bytes make byte order and entry-boundary errors visible. */
  return (uint8_t)(0x21u + matrix_tag * 97u + source_row * 37u + major * 11u +
                   byte_in_element * 53u);
}
#endif

#if TRANSPOSE_PROBE_MASK == 8
static uint8_t nibble_pattern(uint32_t matrix_tag, uint32_t source_row,
                              uint32_t major) {
  return (uint8_t)((0x1u + matrix_tag * 5u + source_row * 7u + major * 3u) &
                   0xfu);
}

static uint8_t prefill_byte(uint32_t matrix_tag, uint32_t major,
                            uint32_t byte_index) {
  const uint8_t high = matrix_tag == 0 ? 0xa0u : 0x50u;
  return (uint8_t)(high | ((major * 3u + byte_index * 5u + 1u) & 0xfu));
}
#endif

static void put_source_element(uint8_t *matrix, uint32_t source_row,
                               uint32_t major, uint32_t matrix_tag) {
#if TRANSPOSE_PROBE_MASK == 8
  const uint32_t byte_index = source_row * SourceStride + (major >> 1);
  const uint8_t value = nibble_pattern(matrix_tag, source_row, major);
  if ((major & 1u) == 0) {
    matrix[byte_index] = (uint8_t)((matrix[byte_index] & 0xf0u) | value);
  } else {
    matrix[byte_index] = (uint8_t)((matrix[byte_index] & 0x0fu) |
                                   (uint8_t)(value << 4));
  }
#else
  const uint32_t byte_index = source_row * SourceStride +
                              major * OriginalBytes;
  for (uint32_t byte = 0; byte < OriginalBytes; byte++) {
    matrix[byte_index + byte] =
        byte_pattern(matrix_tag, source_row, major, byte);
  }
#endif
}

static void prepare_data(void) {
  for (uint32_t row = 0; row < OriginalK; row++) {
    for (uint32_t major = 0; major < LoadMajorDim; major++) {
      put_source_element(a_source, row, major, 0);
      put_source_element(b_source, row, major, 1);
    }
  }

#if TRANSPOSE_PROBE_MASK == 8
  /*
   * Reuse the result buffers as e8 prefill sources.  transpose_probe loads
   * these bytes into tr0/tr1 before the e4 transpose, then later overwrites
   * the buffers with C results.  Distinct non-zero high nibbles make the
   * K=127 read-modify-write preservation observable for both A and B.
   */
  uint8_t *a_prefill = (uint8_t *)a_result;
  uint8_t *b_prefill = (uint8_t *)b_result;
  for (uint32_t major = 0; major < LoadMajorDim; major++) {
    for (uint32_t byte_index = 0; byte_index < ProbeBytes; byte_index++) {
      const uint32_t index = major * ProbeBytes + byte_index;
      a_prefill[index] = prefill_byte(0, major, byte_index);
      b_prefill[index] = prefill_byte(1, major, byte_index);
    }
  }
#endif

  /*
   * CUTE's current compute path consumes K in complete 32-byte reduce groups
   * and stores a full 128-column C row.  Keep the observation multiply at
   * K=64/N=128 and check all 128 MatrixReg rows.
   * Each original precision uses exactly 64 bytes per MatrixReg row; e4 uses
   * K=127 so its final high nibble is intentionally a partial write.
   *
   * mla reads A as [M][K], while normal mlb reads B^T as [N][K].
   * Both therefore use a K-byte row.  The first 64 rows of this 128x64
   * buffer form an identity and can serve both instructions.  The remaining
   * zero rows let both observation loads overwrite the entire tile register,
   * which also makes their full-register difftest state deterministic.
   */
  for (uint32_t i = 0; i < ComputeK; i++) {
    identity[i * ComputeK + i] = 1;
  }
}

static uint8_t expected_byte(uint32_t matrix_tag, uint32_t major,
                             uint32_t byte_index) {
#if TRANSPOSE_PROBE_MASK == 8
  const uint32_t element = byte_index * 2u;
  const uint8_t low = nibble_pattern(matrix_tag, element, major);
  const uint8_t high = (element + 1u < OriginalK)
                         ? (uint8_t)(nibble_pattern(
                               matrix_tag, element + 1u, major) << 4)
                         : (uint8_t)(prefill_byte(
                               matrix_tag, major, byte_index) & 0xf0u);
  return (uint8_t)(low | high);
#else
  const uint32_t element = byte_index / OriginalBytes;
  const uint32_t byte_in_element = byte_index % OriginalBytes;
  return byte_pattern(matrix_tag, element, major, byte_in_element);
#endif
}

static int check_probe_result(const char *which, const uint32_t *result,
                              int b_orientation,
                              uint32_t matrix_tag) {
  int failed = 0;
  for (uint32_t major = 0; major < LoadMajorDim; major++) {
    for (uint32_t byte_index = 0; byte_index < ProbeBytes; byte_index++) {
      const uint32_t index = b_orientation
                               ? byte_index * ComputeN + major
                               : major * ComputeN + byte_index;
      const uint32_t actual = result[index];
      const uint8_t expected = expected_byte(matrix_tag, major, byte_index);
      if ((int32_t)actual != (int32_t)(int8_t)expected) {
        if (!failed) {
          printf("%s %s mismatch at major=%u byte=%u: got=0x%x expected=0x%x"
                 " (sign-extended 0x%x)\n",
                 TEST_NAME, which, major, byte_index, actual, expected,
                 (uint32_t)(int32_t)(int8_t)expected);
        }
        failed = 1;
      }
    }
  }
  return failed;
}

int main(void) {
  prepare_data();
  matrix_init();
  transpose_probe();

  const int a_failed = check_probe_result("A", a_result, 0, 0);
  const int b_failed = check_probe_result("B", b_result, 1, 1);
  const int failed = a_failed | b_failed;
  printf("A/B transpose macc probe %s: %s\n", TEST_NAME,
         failed ? "FAILED" : "passed");
  return failed;
}
