import time
import serial


def empty_read_buffer(ser):
    time.sleep(0.1)
    read_stuff = b""
    while ser.in_waiting > 0:
        read_stuff += ser.read(ser.in_waiting)
    if(len(read_stuff)>0):
        print("Emptying the read buffer:", read_stuff)

def check_xbee_bypass_mode(ser):
    print("Checking XBEE mode")
    time.sleep(1)
    ser.write(b'+++') #this would result in getting a "Unknown" reply
    time.sleep(1.1)
    # print(ser.in_waiting)
    reply = ser.read(ser.in_waiting)
    if(reply == b"+++"):
        ser.write(b'b')
        ser.write(b'\r')
        print("Now in bypass mode")
    elif reply == b"OK\r":
        print("Was already in bypass mode")
        ser.write(b'\r')
    else: 
        print("Unexpected reply to +++ command: ", reply)
        print("is minicom open? :)")
    empty_read_buffer(ser)

def init(baud=115200, path='/dev/ttyUSB0', rtscts=False):
    ser = serial.Serial(path, baud, rtscts=rtscts)  # open first serial port
    print("connected to: " + ser.portstr)
    empty_read_buffer(ser)
    check_xbee_bypass_mode(ser)
    return ser




