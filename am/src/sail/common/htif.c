#include <stdint.h>

// HTIF mailbox.
//
// Sail scans the main ELF's symbol table for `tohost` and only enables its
// HTIF device when that symbol is present; `fromhost` is provided for the
// handshake. Both must live in loadable RAM (they do: the image is based at
// 0x80000000). They are referenced by the putchar/exit paths below, so
// --gc-sections keeps them.
volatile uint64_t tohost   __attribute__((aligned(8)));
volatile uint64_t fromhost __attribute__((aligned(8)));

// HTIF console: device 1, command 1 (write). The payload byte is printed by
// the host. Sail clears `tohost` once it has consumed the request.
void __am_htif_putchar(char ch) {
  tohost = ((uint64_t)1 << 56) | ((uint64_t)1 << 48) | (uint8_t)ch;
  while (tohost != 0);
  fromhost = 0;
}

// HTIF exit: bit 0 signals halt, the upper bits carry the exit code.
void __am_htif_exit(int code) {
  tohost = ((uint64_t)code << 1) | 1;
  while (1);
}
