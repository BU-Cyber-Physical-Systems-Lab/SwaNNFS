#include "uart4Serial.h"
#include "drivers/time.h"

serialPort_t *uart4Serial = NULL;

serialPort_t* getUART4() {
	if(uart4Serial == NULL) {
		uart4Serial = openSerialPort(SERIAL_PORT_USART3, FUNCTION_BLACKBOX, NULL, NULL, 115200, MODE_RXTX, SERIAL_HW_FLOW_CTRL);
		if(millis() < 10000) // waiting 10 seconds so that the xbee connects since drone power on
			delay(10000 - millis());
		serialWrite(uart4Serial, ' '); 
		serialWrite(uart4Serial, 'b'); 
		serialWrite(uart4Serial, '\r'); // setting the xbee into bypass mode
		delay(300);
		while (serialRxBytesWaiting(getUART4())) {
			serialRead(getUART4());
		}
		delay(300);

	}
	return uart4Serial;
}

void write_byte(unsigned char byte) {
	serialWrite(getUART4(), byte);
}