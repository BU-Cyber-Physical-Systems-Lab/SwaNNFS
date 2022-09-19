import serial
import struct
import sys
import numpy
import time
import ctypes
from crccheck.crc import Crc16Mcrf4XX
import xbee
import obs_utils


def block_crc(block):
    crc = Crc16Mcrf4XX().calc(block)
    return ctypes.c_uint16(crc)


def keep_receiving(
        ser,
        keep_checking=False,
        check=True,
        avg_loop_time=1,
        loop_num=0,
        first_sync_size = 1,
        second_sync_size=0,
        expected_num=1,
        debug=True
    ):

    checked_observation = obs_utils.checked_observation_t()
    while True:
        loop_num += 1
        begin = time.time()
        if check:
            while True:
                sync_byte = ser.read(1)
                if ord(sync_byte)==228:
                    check = keep_checking
                    break
        else:
            ser.read()

        for i in range(expected_num):
            #print(ctypes.sizeof(checked_observation))
            #print(ctypes.sizeof(observation_t))
            #print(ctypes.sizeof(ctypes.c_uint16))
            data = ser.read(ctypes.sizeof(checked_observation)) #array of bytes    
            #f_data = struct.unpack("f",data)
            ctypes.memmove(ctypes.pointer(checked_observation),
                data, ctypes.sizeof(checked_observation))
            received_crc = ctypes.c_uint16(checked_observation.crc)
            crc = block_crc(bytes(checked_observation.observation))
            packed_crc = bytes(crc)
        #expected_crc = ser.read(second_sync_size)
        expected_crc = checked_observation.crc
        #print(type(expected_crc))
        if debug:    
            print("calculated_crc: ", crc.value, "   recieved_crc:", expected_crc)
            #print("calculated_crc_first_byte: ", packed_crc[0], "calculated_crc_second_byte: ", packed_crc[1] )
            
            if(expected_crc != crc.value):
                print("not match in crc")
                #exit()
            else:
                #print(avg_loop_time)
                #print(checked_observation.observation.ang_vel.yaw)
                # print(checked_observation.observation.ang_vel.yaw)
                # print(checked_observation.observation.ang_acc.roll)
                print("recieved struct:")
                print(checked_observation.observation,"\n\n")
                
        avg_loop_time = avg_loop_time*0.9 + 0.1*(time.time()-begin)

            
def receive_obs(ser, check, keep_checking=False, debug=True):
    checked_observation = obs_utils.checked_observation_t()
    if check:
        while True:
            sync_byte = ser.read(1)
            if ord(sync_byte)==228:
                check = keep_checking
                break
    else:
        ser.read()
        
    data = ser.read(ctypes.sizeof(checked_observation)) #array of bytes    
    ctypes.memmove(ctypes.pointer(checked_observation),
        data, ctypes.sizeof(checked_observation))
    received_crc = ctypes.c_uint16(checked_observation.crc)
    crc = block_crc(bytes(checked_observation.observation))
    expected_crc = checked_observation.crc

    if debug:    
        print("calculated_crc: ", crc.value, "   recieved_crc:", expected_crc)
        if(expected_crc != crc.value):
            print("not match in crc")
        else:
            print("recieved struct:")
            print(checked_observation.observation,"\n\n")
    return check, checked_observation.observation


def send_and_recieve_byte(
        ser,
        byte=b"A"
    ):
    print("sending: ", byte)
    ser.write(byte)
    ans = ser.read(1)
    print("recieved: ", ans)


        
if __name__ == '__main__':
    ser = xbee.init()
    
    print("Reading the dummy byte from drone:")
    dummy  = ser.read(4)
    print("Recieved byte is: ",dummy)

    #ser.read(4000)
    #send_and_recieve_byte(ser)
    keep_receiving(ser, keep_checking=True)
    ser.close()