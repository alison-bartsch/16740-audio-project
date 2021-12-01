
from model import FC
from numpy.core.numeric import Inf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets import AudioDataset

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

def plot_traj(traj, ax):
    cm = plt.get_cmap("jet")
    n = len(traj)
    # ax.set_color_cycle([cm(1.0*i/(n-1)) for i in range(n-1)])
    for i in range(n-1):
        ax.plot(traj[:,0], traj[:,1])


def visualize_all():
    # iterate through sources to get the data files: sources[i] + "_data_" + str(i) + ".npz"
    for location in sources:
        s = sound_loc[location]
        fig, axs = plt.subplots(2,10)
        fig.set_size_inches(20, 5)
        # iterate through the .npz files
        for i in range(10):
            file_path = "data/" + location + "_data_" + str(i) + ".npz"
            x, y, l, r = load_data(file_path, sep)

def find_nearest_pos(pos):
    pos = np.array([int(5 * round(pos[0]/5.0)), int(5 * round(pos[1]/5.0))])
    pos[0] = min(pos[0], 30)
    pos[0] = max(pos[0], -30)
    pos[1] = min(pos[1], 30)
    pos[1] = max(pos[1], -30)
    return pos

def scale_trajectory(cum_pred):
    traj_list = np.linspace(10,30,5)
    for i in range(len(traj_list)):
        if cum_pred < (0.4 + 0.15*i):
            return traj_list[i]
    
def rollout(model, l, r, scale, shift):
    pos = np.array([0,0])
    goal = np.array([0,0])
    pos_list = [pos]
    cum_pred = torch.tensor(np.zeros(4))
    for i in range(7):
        input = (np.array([pos[0], pos[1], l[tuple(pos)], r[tuple(pos)]]) - shift) / scale
        pred = model(torch.tensor(input.astype("float32")))
        cum_pred = cum_pred * 0.8 + pred
        loc = cum_pred.argmax()
        goal = sound_loc[sources[loc]]
        if np.linalg.norm(goal-pos)  < 1e-6:
            break
        direction = (goal-pos) / np.linalg.norm(goal-pos)
        print(pos)
        pos = pos + direction * 10

        # # update position with scaled trajectory 
        # pos = pos + direction * scale_trajectory(cum_pred)

        print(pos)
        pos = find_nearest_pos(pos)
        print(pos)
        
        pos_list.append(pos)
    
    return np.array(pos_list)

def simulate():
    # iterate through sources to get the data files: sources[i] + "_data_" + str(i) + ".npz"
    model = FC(2, 4, 1000, 4)
    model.load_state_dict(torch.load("./models/FC3-1000.pth"))

    dataset = AudioDataset("./data")

    for location in sources:
        s = sound_loc[location]
        fig, axs = plt.subplots(2,10)
        fig.set_size_inches(20, 5)
        # iterate through the .npz files
        for i in range(10):
            file_path = "data/" + location + "_data_" + str(i) + ".npz"
            x, y, l, r = load_data(file_path, sep)
            l_map, r_map = load_data_map(file_path)
            plot_data(x, y, l, r, axs[0,i], axs[1,i])
            traj = rollout(model, l_map, r_map, dataset.scale, dataset.shift)
            plot_traj(traj, axs[0,i])

        # label_source(axs[0])
        # label_source(axs[1])
        plt.show()
        
def load_data_map(data_file):
    npzfile = np.load(data_file)
    data = npzfile["data"]

    l_map = dict()
    r_map = dict()
    for i in range(len(data)):
        x = int(data[i][0][0])
        y = int(data[i][0][1])
        l_map[(x,y)] = data[i][1][0]
        r_map[(x,y)] = data[i][1][1]

    return l_map, r_map


simulate()