#ifndef AME_EW_EXAMPLES_H
#define AME_EW_EXAMPLES_H

#include <am.h>
#include <klib.h>
#include <stdint.h>
#include <stddef.h>

/* Test-local instruction examples, not compiler intrinsics or a calling ABI.
 * Matrix/LV registers are explicit physical state; each writer names its clobber.
 * All strides below are bytes, and scalar FP operands travel as raw low32 bits.
 */
#define AME_DIM 128u
#define AME_SLOTS (AME_DIM * AME_DIM)
#define AME_I8   0x02ul
#define AME_I32  0x04ul
#define AME_F16  0x09ul
#define AME_BF16 0x0aul
#define AME_F32  0x0cul
#define AME_NX 1u
#define AME_UF 2u
#define AME_OF 4u
#define AME_DZ 8u
#define AME_NV 16u
#define AME_STR1(x) #x
#define AME_STR(x) AME_STR1(x)

#define AME_CFG(cfg, value) \
  asm volatile("msetcfg " #cfg ", %0" :: "r"((uintptr_t)(value)) : #cfg, "memory")
#define AME_CVT(cfg, acc, value) \
  asm volatile("msetcfg.cvt " #cfg ", %0" :: "r"((uintptr_t)(value)) : #cfg, #acc, "memory")
#define AME_GETCFG(out, cfg) \
  asm volatile("mgetcfg %0, " #cfg : "=r"(out) :: "memory")
#define AME_LOAD(op, reg, ptr, stride) \
  asm volatile(AME_STR(op) " " #reg ", (%0), %1" :: \
    "r"(ptr), "r"((uintptr_t)(stride)) : #reg, "memory")
#define AME_STORE(op, reg, ptr, stride) \
  asm volatile(AME_STR(op) " " #reg ", (%0), %1" :: \
    "r"(ptr), "r"((uintptr_t)(stride)) : "memory")
#define AME_LOAD_WHOLE(op, reg, ptr) \
  asm volatile(AME_STR(op) " " #reg ", (%0)" :: "r"(ptr) : #reg, "memory")
#define AME_STORE_WHOLE(op, reg, ptr) \
  asm volatile(AME_STR(op) " " #reg ", (%0)" :: "r"(ptr) : "memory")
/* A is last: msub dst, B, A computes A-B. */
#define AME_EW(op, dst, b, a) \
  asm volatile(AME_STR(op) " " #dst ", " #b ", " #a ::: #dst, "memory")
#define AME_X(op, dst, bits, a) \
  asm volatile(AME_STR(op) " " #dst ", %0, " #a :: \
    "r"((uintptr_t)(bits)) : #dst, "memory")
#define AME_UNARY(op, dst, a) \
  asm volatile(AME_STR(op) " " #dst ", " #a ::: #dst, "memory")
#define AME_REDUCE(op, vd, a, accumulate) \
  asm volatile(AME_STR(op) " " #vd ", " #a ", " AME_STR(accumulate) ::: #vd, "memory")
#define AME_CSR_WRITE(csr, value) \
  asm volatile("csrw " AME_STR(csr) ", %0" :: "r"((uintptr_t)(value)) : "memory")
#define AME_CSR_READ(csr) ({ uintptr_t v_; \
  asm volatile("csrr %0, " AME_STR(csr) : "=r"(v_) :: "memory"); v_; })

static inline void ame_shape(unsigned m, unsigned n, unsigned k) {
  asm volatile("msettilem %0\n\tmsettilen %1\n\tmsettilek %2"
               :: "r"((uintptr_t)m), "r"((uintptr_t)n), "r"((uintptr_t)k) : "memory");
}
static inline void ame_flags_clear(void) {
  AME_CSR_WRITE(0x808, 0); /* mfflags */
  AME_CSR_WRITE(0x807, 0); /* msat */
}
static inline uint32_t f32_bits(float x) {
  union { float f; uint32_t u; } v = { .f = x }; return v.u;
}
static inline float f32_from_bits(uint32_t x) {
  union { float f; uint32_t u; } v = { .u = x }; return v.f;
}

void check_u32(const char *label, unsigned line, uint32_t got, uint32_t expected);
void check_close(const char *label, unsigned line, float got, float expected, float tolerance);
void case_begin(const char *label);
#define CHECK_U32(label, got, expected) check_u32((label), __LINE__, (uint32_t)(got), (uint32_t)(expected))
#define CHECK_CLOSE(label, got, expected, tolerance) check_close((label), __LINE__, (got), (expected), (tolerance))
void test_ew(void);
void test_sfu(void);
void test_reduce(void);
void test_memory(void);
void test_convert(void);
void test_operators(void);
#endif
