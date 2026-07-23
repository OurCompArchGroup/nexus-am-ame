#include <am.h>
#include <klib.h>

extern char _heap_start;
extern char _pmem_end;
int main(const char *args);

void __am_htif_putchar(char ch);
void __am_htif_exit(int code);

_Area _heap = {
  .start = &_heap_start,
  .end = &_pmem_end,
};

// Route console output through HTIF instead of UART-lite / 16550 MMIO.
void _putc(char ch) {
  __am_htif_putchar(ch);
}

// Halt through HTIF instead of the NEMU-trap custom instruction
// (`.word 0x0005006b`), which Sail does not implement.
void _halt(int code) {
  __am_htif_exit(code);
  while (1);  // unreachable
}

void _trm_init() {
  extern const char __am_mainargs;
  int ret = main(&__am_mainargs);
  _halt(ret);
}
