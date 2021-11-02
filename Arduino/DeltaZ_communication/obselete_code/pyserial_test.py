import serial
import time
ser = serial.Serial('/dev/cu.usbserial-10')
time.sleep(2)
print(ser.name)

ser.readline()
ser.readline()

minval=65
maxval=1024

def write_read(x):
  ser.write(bytes(x))
  time.sleep(0.05)
  data = ser.readline()
  return data

# data = write_read(b'goto 0,30,-45 \n')
# print(data)
# time.sleep(.5)
# data = write_read(b'goto 0,-30,-45 \n')
# print(data)
# time.sleep(.5)
# data = write_read(b'goto 0,0,-45 \n')
# print(data)

for iter in range(100):
  data=write_read(b'read \n')
  desY=60.0*((float(int(data)-minval)/float(maxval-minval))-0.5)
  print(data,desY)
  data=write_read(b'goto 0, '+str(desY).encode('ASCII')+b',-45 \n')

ser.close()