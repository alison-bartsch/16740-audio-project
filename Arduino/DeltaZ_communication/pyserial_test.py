import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque 
import matplotlib
import re
# matplotlib.use('agg')
ser = serial.Serial('/dev/cu.usbserial-1130')
time.sleep(2)
print(ser.name)

ser.readline()
ser.readline()

minval=65
maxval=1024

def write_read(x):
  ser.write(bytes(x))
  time.sleep(0.001)
  data = ser.readline()
  return data

def read_microphones():
  data=write_read(b'read \n')
  # print(data)
  data = str(data)
  data = [int(s) for s in re.findall(r'\b\d+\b', data)]
  return data


def collect_sound(steps=100):
  l1 = []
  l2 = []
  for i in range(steps):
    data = read_microphones()
    # print("iter ",i)
    # print(data)
    l1.append(data[0])
    l2.append(data[1])
  return l1, l2

def collect_baselines(steps):
  l1, l2 = collect_sound()
  print(l1)
  print(l2)
  return np.mean(l1), np.mean(l2)

def peak_value(steps, percent):
  l1, l2 = collect_sound(steps)
  l1 = sorted(l1)
  l2 = sorted(l2)
  idx = int(percent * len(l1))
  return l1[idx], l2[idx]

def plot_data(time_steps, mic1, mic2):
  plt.clf()
  plt.plot(time_steps, mic1)
  plt.plot(time_steps, mic2)
  plt.ylim(-100, 100)
  plt.draw()
  plt.pause(0.0001)

def move_to(x,y,z):
  s = str.encode('goto {},{},{} \n'.format(x,y,z))
  print("move to command:")
  print(s)
  ret = write_read(s)
  print(ret)
  # data = write_read(b'goto 0,30,-45 \n')
  # data=write_read(b'goto 0,30,-45 \n')
  time.sleep(0.5)

def move_around():
  # xs = np.linspace(-30, 30, num=11)
  # ys = np.linspace(-30, 30, num=11)
  xs = np.arange(-30,31,1)
  ys = np.arange(-30,31,1)
  z = -50
  data = []
  for x in xs:
    for y in ys:
      if x**2 + y**2 <= 900:
        print(x,y)
        move_to(x,y,z)
        data.append(((x,y), peak_value(100, 0.9)))
  return data


shift1, shift2 = collect_baselines(1000)

# move_around()
# collect_sound()
# read_microphones()
# exit()



# data = write_read(b'goto 0,30,-45 \n')
# print(data)
# time.sleep(.5)
# data = write_read(b'goto 0,-30,-45 \n')
# print(data)
# time.sleep(.5)
# data = write_read(b'goto 0,0,-45 \n')
# print(data)

mic1 = deque()
mic2 = deque()
time_steps = deque()
tot_len = 100

fig = plt.figure()
plt.ion()
plt.show()


read_time = 0
plot_time = 0


for iter in range(1000):
  if (iter % 100 == 0):
    print(iter)
  t1 = time.time()
  data = read_microphones()
  mic1.append(data[0] - shift1)
  mic2.append(data[1] - shift2)

  # time_steps.append(time.time())
  time_steps.append(iter)

  # if len(mic1) > tot_len:
  #   mic1.popleft()
  #   mic2.popleft()
  #   time_steps.popleft()

  # desY=60.0*((float(int(data)-minval)/float(maxval-minval))-0.5)
  # print(data,desY)

  t2 = time.time()
  # if len(time_steps) == tot_len:
  #   plot_data(time_steps, mic1, mic2)
  t3 = time.time()

  read_time += t2 - t1
  plot_time += t3 - t2
  # time.sleep(0.001)


plot_data(time_steps, mic1, mic2)
time.sleep(5)
print(read_time)
print(plot_time)
  # data_num = data.index()
  # data=write_read(b'goto 0, '+str(desY).encode('ASCII')+b',-45 \n')

ser.close()