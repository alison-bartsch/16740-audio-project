import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque 
import matplotlib
import re
# matplotlib.use('agg')
# ser = serial.Serial('/dev/cu.usbserial-120')
ser = serial.Serial('/dev/cu.usbserial-130')
time.sleep(2)
print(ser.name)

ser.readline()
ser.readline()

minval=65
maxval=1024

def write_read(x):
    ser.write(bytes(x))
    time.sleep(0.1)
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
    l1, l2 = collect_sound(steps=steps)
    return np.array([np.mean(l1), np.mean(l2)])

def peak_value(steps, percent, shifts):
    l1, l2 = collect_sound(steps)
    l1 = np.abs(l1 - shifts[0])
    l2 = np.abs(l2 - shifts[1])
    l1 = sorted(l1)
    l2 = sorted(l2)
    idx = int(percent * (len(l1)-1))
    return np.array([l1[idx], l2[idx]])
    # return np.array([np.mean(l1), np.mean(l2)])

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

def collect_data(sep):
    shifts = collect_baselines(200)
    input("\nTURN ON METRONOME! HIT ENTER TO CONTINUE.\n")
    xs = np.arange(-30,31,sep)
    ys = np.arange(-30,31,sep)
    z = -45
    data = []

    for i,x in enumerate(xs):
        first_time=True
        for j,y in enumerate(ys):
            if x**2 + y**2 <= 900:
                move_to(x,y,z)
                time.sleep(1)
                if first_time:
                    # time.sleep(2)
                    first_time=False
                pv = peak_value(100, 1.0, shifts)
                # time.sleep(1.5)
                print(pv)
                data.append(((x,y), pv))
    return data

def visualize_data(data_file, sep):
    npzfile = np.load(data_file)
    data = npzfile["data"]

    x,y = np.meshgrid(np.arange(-30-sep//2,31+sep//2,sep), np.arange(-30-sep//2,31+sep//2,sep))
    l = np.ones((60//sep+1,60//sep+1)) * -10
    r = np.ones((60//sep+1,60//sep+1)) * -10
    shift = 30//sep
    for i in range(len(data)):
        xx = int(data[i][0][0])
        yy = int(data[i][0][1])
        l[xx//sep + shift][yy//sep + shift] = data[i][1][0]
        r[xx//sep + shift][yy//sep + shift] = data[i][1][1]

    # print("x")
    # print(x)
    # print("y")
    # print(y)
    # print("l")
    # print(l)
    fig, axs = plt.subplots(1,2)
    c = axs[0].pcolormesh(x, y, l, cmap='RdBu', vmin=40, vmax=250)
    axs[0].set_title('Left')
    axs[0].axis([x.min(), x.max(), y.min(), y.max()])
    axs[0].set_aspect('equal')
    fig.colorbar(c, ax=axs[0])

    c = axs[1].pcolormesh(x, y, r, cmap='RdBu', vmin=40, vmax=250)
    axs[1].set_title('Right')
    axs[1].axis([x.min(), x.max(), y.min(), y.max()])
    axs[1].set_aspect('equal')
    fig.colorbar(c, ax=axs[1])
    fig.set_size_inches(12, 5)

    
    # save the plot
    plot_name = data_file.split('.')[0].split('/')
    plot_path = "figures/" + plot_name[1] + ".png"
    plt.savefig(plot_path)

    plt.show()

def test():
    shifts = collect_baselines(1000)
    l, r = collect_sound(step=100)
    l = l - shifts[0]
    r = r - shifts[1]
    plot_data(range(100), l, r)

def collect_and_save_data(save_path):
    sep=5
    data = collect_data(sep=sep)
    np.savez(save_path, data=data)
    visualize_data(save_path,sep)

def load_data(save_path):
    visualize_data(save_path)


# Data Collection Loop - front 
for i in range(10):
    data_path = "data/front2_data_" + str(i) + ".npz"
    collect_and_save_data(data_path)
    input("\nTURN OFF METRONOME! HIT ENTER TO CONTINUE.\n")

ser.close()