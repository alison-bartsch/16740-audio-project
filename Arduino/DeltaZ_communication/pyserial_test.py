import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque 
import matplotlib
import re
# matplotlib.use('agg')
ser = serial.Serial('/dev/cu.wchusbserial1410')
time.sleep(2)
print(ser.name)

ser.readline()
ser.readline()

minval=65
maxval=1024

def write_read(x):
    ser.write(bytes(x))
    time.sleep(0.001)
    data = ser.readline().decode("utf-8")
    return data

def collect_sound(steps=1):
    cmd = str.encode('read {} \n'.format(steps))
    data = write_read(cmd)
    data = data.split(";")[:-1]
    data = [list(map(int, s.split(','))) for s in data]
    data = np.array(data).transpose()
    return data[0], data[1]

def collect_baselines(steps):
    move_to(0,0,-45)
    l1, l2 = collect_sound()
    print(l1)
    print(l2)
    return np.array([np.mean(l1), np.mean(l2)])

def peak_value(steps, percent):
    l1, l2 = collect_sound(steps)
    l1 = sorted(l1)
    l2 = sorted(l2)
    idx = int(percent * len(l1))
    return np.array([l1[idx], l2[idx]])

def plot_data(time_steps, mic1, mic2):
    # plt.clf()
    plt.figure()
    plt.plot(time_steps, mic1)
    plt.plot(time_steps, mic2)
    # plt.ylim(-100, 100)
    plt.show()
    # plt.draw()
    # plt.pause(0.0001)

def move_to(x,y,z):
    s = str.encode('goto {},{},{} \n'.format(x,y,z))
    print("move to command:", s)
    ret = write_read(s)
    time.sleep(1)

def collect_data():
    shifts = collect_baselines(1000)

    xs = np.arange(-30,31,5)
    ys = np.arange(-30,31,5)
    z = -45
    data = []

    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            if j == 0:
                    time.sleep(1)
            if x**2 + y**2 <= 900:
                move_to(x,y,z)
                pv = peak_value(100, 0.9) - shifts
                print(pv)
                data.append(((x,y), pv))
    return data

def visualize_data(data_file):
    npzfile = np.load(data_file)
    data = npzfile["data"]

    x,y = np.meshgrid(np.arange(-30,31,5), np.arange(-30,31,5))
    l = np.ones((13,13)) * -10
    r = np.ones((13,13)) * -10
    for i in range(len(data)):
        xx = int(data[i][0][0])
        yy = int(data[i][0][1])
        l[xx//5 + 6][yy//5 + 6] = data[i][1][0]
        r[xx//5 + 6][yy//5 + 6] = data[i][1][1]

    fig, axs = plt.subplots(1,2)
    c = axs[0].pcolormesh(x, y, l, cmap='RdBu', vmin=-10, vmax=40)
    axs[0].set_title('Left')
    axs[0].axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=axs[0])

    c = axs[1].pcolormesh(x, y, r, cmap='RdBu', vmin=-10, vmax=40)
    axs[1].set_title('Right')
    axs[1].axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=axs[1])

    plt.show()

def test():
    shifts = collect_baselines(1000)
    l, r = collect_sound(step=100)
    l = l - shifts[0]
    r = r - shifts[1]
    plot_data(range(100), l, r)

def collect_and_save_data(save_path):
    data = collect_data()
    np.savez(save_path, data=data)
    visualize_data(save_path)

def load_data(save_path):
    np.savez(save_path, data=data)
    visualize_data(save_path)



collect_and_save_data("front_data.npz")
ser.close()