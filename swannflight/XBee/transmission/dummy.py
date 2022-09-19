import transmitter
import receiver
import time
import serial
from threading import Thread

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyUSB0', 9600, rtscts=False)  # open first serial port
    while (True):
        for i in range(10):
            ser.write(chr(97+i).encode("ASCII"))
            print(ser.read(1))
            #time.sleep(0.1)