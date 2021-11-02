import serial
import time
ser = serial.Serial('/dev/cu.usbserial-1130')
time.sleep(2)
print(ser.name)

minval=65
maxval = 1024

ser.readline()
ser.readline()

def write_read(x):
    ser.write(bytes(x))
    time.sleep(0.05)
    data = ser.readline()
    return data

for iter in range(100):
    data = write_read(b'read \n')
    # desY = 60.0*((float(int(data) - minval)/float(maxval - minval)) - 0.5)
    # print(data, desY)
    # data = write_read(str.encode('goto 0,'+str(20)+',-45\n'))
    data=write_read(b'goto 0,30,-45 \n')
    time.sleep(0.05)
