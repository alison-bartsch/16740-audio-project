
from model import FC
from numpy.core.numeric import Inf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as patches
import datasets

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque 
import matplotlib
import re
import math
import random


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

def plot_data(x, y, l, r, ax0, ax1):
    
    c = ax0.pcolormesh(x, y, l, cmap='RdBu', vmin=40, vmax=250, shading='auto')
    ax0.set_title('Left')
    ax0.axis([x.min(), x.max(), y.min(), y.max()])
    ax0.set_aspect('equal')
    # fig.colorbar(c, ax=ax0)

    c = ax1.pcolormesh(x, y, r, cmap='RdBu', vmin=40, vmax=250, shading='auto')
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
        ax.plot(traj[:,0], traj[:,1], c='g')

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
    if abs(pos[0]) >= 30:
        pos[1] = 0
    if abs(pos[1]) >= 30:
        pos[0] = 0
    return pos
    
def rollout(model, start_pos, l, r, scale, shift):
    pos = np.array(start_pos)
    goal = np.array([0,0])
    pos_list = [pos]
    cum_pred = torch.tensor(np.zeros(4))
    for i in range(30):
        input = (np.array([pos[0], pos[1], l[tuple(pos)], r[tuple(pos)]]) - shift) / scale
        pred = model(torch.tensor(input.astype("float32")))
        cum_pred = cum_pred * 0.8 + pred
        loc = cum_pred.argmax()
        goal = sound_loc[sources[loc]]
        if np.linalg.norm(goal-pos)  < 1e-6:
            break
        direction = (goal-pos) / np.linalg.norm(goal-pos)
        confidence = cum_pred[loc].detach().numpy() * 2.5
        # confidence = 10
        # print(confidence)
        pos = pos + direction * confidence
        # print(np.linalg.norm(pos))
        pos = find_nearest_pos(pos)
        
        pos_list.append(pos)
    
    return np.array(pos_list)

def simulate():
    # iterate through sources to get the data files: sources[i] + "_data_" + str(i) + ".npz"
    model = FC(2, 4, 1000, 4)
    model.load_state_dict(torch.load("./models/FC3-1000-8:2.pth"))

    dataset = datasets.OneStepDataset("./data", test=True)

    for location in sources:
        s = sound_loc[location]
        fig, axs = plt.subplots(2,10)
        fig.set_size_inches(20, 5)
        # iterate through the .npz files
        for i in range(8,10):
            file_path = "data/" + location + "_data_" + str(i) + ".npz"
            x, y, l, r = load_data(file_path, sep)
            l_map, r_map = load_data_map(file_path)
            plot_data(x, y, l, r, axs[0,i], axs[1,i])
            traj = rollout(model, l_map, r_map, dataset.scale, dataset.shift)
            plot_traj(traj, axs[0,i])

        # label_source(axs[0])
        # label_source(axs[1])
        plt.show()

def evaluate_one_step():
    # iterate through sources to get the data files: sources[i] + "_data_" + str(i) + ".npz"
    model = FC(2, 4, 1000, 4)
    model.load_state_dict(torch.load("./models/FC3-1000-8:2.pth"))

    dataset = datasets.OneStepDataset("./data")

    for location in sources:
        s = sound_loc[location]
        # fig, axs = plt.subplots(4,1)
        fig, axs = plt.subplots(4,25)
        fig.set_size_inches(20, 5)
        # iterate through the .npz files
        success = 0
        tot_steps = 0
        for i in range(8,10):
            cnt = 0
            file_path = "data/" + location + "_data_" + str(i) + ".npz"
            x, y, l, r = load_data(file_path, sep)
            l_map, r_map = load_data_map(file_path)
            # plot_data(x, y, l, r, axs[(i-8)*2], axs[(i-8)*2+1])
            for sx in range(-10,11,5):
                for sy in range(-10,11,5):
                    plot_data(x, y, l, r, axs[(i-8)*2,cnt], axs[(i-8)*2+1,cnt])
                    traj = rollout(model, [sx,sy], l_map, r_map, dataset.scale, dataset.shift)
                    # plot_traj(traj, axs[(i-8)*2])
                    plot_traj(traj, axs[(i-8)*2, cnt])
                    cnt += 1
                    tot_steps += len(traj)
                    if np.linalg.norm(traj[-1] - s) < 1e-3:
                        success += 1
        [axi.set_axis_off() for axi in axs.ravel()]
        plt.savefig("./figures/eval_adaptive_"+location+".png")
        plt.show()
        print("success rate:", success/50)
        print("average steps:", tot_steps/50)
        # label_source(axs[0])
        # label_source(axs[1])

def heuristic_locations():
    # randomly sample an angle
    th1 = math.radians(random.randint(0,360))
    r = 25

    th2 = th1 + math.radians(120)
    th3 = th1 - math.radians(120)

    x1 = math.cos(th1)*r
    y1 = math.sin(th1)*r

    x2 = math.cos(th2)*r
    y2 = math.sin(th2)*r

    x3 = math.cos(th3)*r
    y3 = math.sin(th3)*r

    # round all the points to the nearest mutliple of 5
    x1 = 5 * round(x1/5)
    y1 = 5 * round(y1/5)
    x2 = 5 * round(x2/5)
    y2 = 5 * round(y2/5)
    x3 = 5 * round(x3/5)
    y3 = 5 * round(y3/5)

    # print([(x1, y1), (x2, y2), (x3, y3)], '\n')
    return [(x1, y1), (x2, y2), (x3, y3)]


def heuristic(l_map, r_map, point_locations):
    # point_locations = [(-15, -15), (0, 20), (15, -5)]
    LR_difference = []
    LR_avg = []
    x_loc = []
    for point in point_locations:
        L = l_map[point]
        R = r_map[point]
        LR_difference.append(L-R)
        LR_avg.append(np.mean([L, R]))
        x_loc.append(point[0])

    diff_avg = np.mean(LR_difference)
    # print("LR Difference: ", diff_avg, "\n")
    max_idx = np.argmax(LR_avg)

    other_pts = np.delete(x_loc, max_idx)
    x_dir = x_loc[max_idx] - other_pts[0] + x_loc[max_idx] - other_pts[1]
    # print("X Direction: ", x_dir, "\n")

    # look at if sound source is from left or right
    if diff_avg > 35:
        return "left"
    elif diff_avg < -35:
        return "right"

    # look at if sound source is front or back
    elif x_dir > 0:
        return "front"
    else:
        return "back"


def evaluate_heuristic():
    # iterate through sources to get the data files: sources[i] + "_data_" + str(i) + ".npz"
    
    dataset = datasets.OneStepDataset("./data")

    for location in sources:
        s = sound_loc[location]
        # fig, axs = plt.subplots(4,1)
        fig, axs = plt.subplots(4,25)
        fig.set_size_inches(20, 5)
        # iterate through the .npz files
        success = 0
        tot_steps = 0
        num_scenarios = 0
        for i in range(8,10):
            cnt = 0
            file_path = "data/" + location + "_data_" + str(i) + ".npz"
            x, y, l, r = load_data(file_path, sep)
            l_map, r_map = load_data_map(file_path)

            # iterate through the different sampled positions - get 10 different random samples
            for j in range(25):
                point_locations = heuristic_locations()
                pred = heuristic(l_map, r_map, point_locations)
                # plot_data(x, y, l, r, axs[(i-8)*2,cnt], axs[(i-8)*2+1,cnt])

                num_scenarios += 1

                # print("ground truth:", location)
                # print("pred:", pred)
                if pred == location:
                    success += 1

        # [axi.set_axis_off() for axi in axs.ravel()]
        # plt.savefig("./figures/eval_heuristic_"+location+".png")
        # plt.show()
        print("success rate for ", location, ": ", success, '/', num_scenarios)
        # label_source(axs[0])
        # label_source(axs[1])

# simulate()
evaluate_heuristic()