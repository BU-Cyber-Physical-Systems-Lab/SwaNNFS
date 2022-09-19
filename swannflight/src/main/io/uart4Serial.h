#ifndef UART4SERIAL_H
#define UART4SERIAL_H

#include "io/serial.h"
#include "neuroflight/byte_utils.h"
extern serialPort_t* getUART4();
extern void write_byte(unsigned char byte);
#define write_little_endian(x) (on_little_endian(x, write_byte)) 

#endif