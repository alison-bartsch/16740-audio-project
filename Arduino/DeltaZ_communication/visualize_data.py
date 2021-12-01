import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque 
import matplotlib
import re


sources = ["front", "back", "left", "right"]

sound_loc = {"front": (30, 0),
            "back": (-30, 0),
            "left": (0, 30),
            "right": (0, -30)}
sep=5

def label_source(ax):
    for source in sources:
        ax.scatter(sound_loc[source][0], sound_loc[source][1], label=source)
    ax.legend()

def load_data(data_file, sep):
    npzfile = np.load(data_file)
    data = npzfile["data"]

    x,y = np.meshgrid(np.arange(-30-sep//2,31+sep+sep//2,sep), np.arange(-30-sep//2,31+sep+sep//2,sep))
    l = np.ones((60//sep+2,60//sep+2)) * -10
    r = np.ones((60//sep+2,60//sep+2)) * -10
    shift = 30//sep
    for i in range(len(data)):
        xx = int(data[i][0][0])
        yy = int(data[i][0][1])
        l[xx//sep + shift][yy//sep + shift] = data[i][1][0]
        r[xx//sep + shift][yy//sep + shift] = data[i][1][1]
        print(data[i][0][0], xx//sep + shift, yy//sep + shift)
    return x, y, l, r

def plot_data(x, y, l, r, ax0, ax1):
    
    c = ax0.pcolormesh(x, y, l, cmap='RdBu', vmin=40, vmax=250)
    ax0.set_title('Left')
    ax0.axis([x.min(), x.max(), y.min(), y.max()])
    ax0.set_aspect('equal')
    # fig.colorbar(c, ax=ax0)

    c = ax1.pcolormesh(x, y, r, cmap='RdBu', vmin=40, vmax=250)
    ax1.set_title('Right')
    ax1.axis([x.min(), x.max(), y.min(), y.max()])
    ax1.set_aspect('equal')

    # save the plot
    # plot_name = data_file.split('.')[0].split('/')
    # plot_path = "figures/" + plot_name[1] + ".png"
    # plt.savefig(plot_path)
    # plt.show()
    return c

def visualize_all():
    # iterate through sources to get the data files: sources[i] + "_data_" + str(i) + ".npz"
    for location in sources:
        s = sound_loc[location]
        print(s)
        fig, axs = plt.subplots(2,10)
        fig.set_size_inches(20, 5)
        fig.suptitle(location, fontsize=16)
        # iterate through the .npz files
        for i in range(10):
            file_path = "data/" + location + "_data_" + str(i) + ".npz"
            x, y, l, r = load_data(file_path, sep)
            plot_data(x, y, l, r, axs[0,i], axs[1,i])
        # plt.show()
        # break
        plt.savefig("./figures/"+location+"_all.png")

def visualize_average():
    # iterate through sources to get the data files: sources[i] + "_data_" + str(i) + ".npz"
    for location in sources:
        s = sound_loc[location]
        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(10, 5)
        fig.suptitle(location, fontsize=16)

        # iterate through the .npz files
        l_avg = []
        r_avg = []
        for i in range(10):
            file_path = "data/" + location + "_data_" + str(i) + ".npz"
            x, y, l, r = load_data(file_path, sep)
            l_avg.append(l)
            r_avg.append(r)
        # l_avg = np.mean(l_avg, axis=0)
        # r_avg = np.mean(r_avg, axis=0)
        l_avg = np.max(l_avg, axis=0)
        r_avg = np.max(r_avg, axis=0)

        c = plot_data(x, y, l_avg, r_avg, axs[0], axs[1])
        # fig.colorbar(c, ax=axs[1])
        label_source(axs[0])
        label_source(axs[1])
        # plt.show()
        plt.savefig("./figures/"+location+"_average.png")

visualize_average()
visualize_all()