
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque 
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
from robot import Robot


sources = ["front", "back", "left", "right"]
sound_loc = {"front": (30, 0),
            "back": (-30, 0),
            "left": (0, 30),
            "right": (0, -30)}
sep=5

def find_nearest_pos(pos):
    pos = np.array([int(5 * round(pos[0]/5.0)), int(5 * round(pos[1]/5.0))])
    pos[0] = min(pos[0], 30)
    pos[0] = max(pos[0], -30)
    pos[1] = min(pos[1], 30)
    pos[1] = max(pos[1], -30)
    return pos

def execute():
    robot = Robot()
    print("collecting environment volume")
    quiet = robot.collect_baselines(200)
    input("done, hit enter to continue")
    # iterate through sources to get the data files: sources[i] + "_data_" + str(i) + ".npz"
    model = FC(2, 4, 1000, 4)
    model.load_state_dict(torch.load("./models/FC3-1000.pth"))
    dataset = AudioDataset("./data")

    pos = np.array([0,0])
    goal = np.array([0,0])
    pos_list = [pos]
    cum_pred = torch.tensor(np.zeros(4))
    while True:
        lr = robot.peak_value(100, 1.0, quiet)
        print(lr)
        X = (np.array([pos[0], pos[1], lr[0], lr[1]]) - dataset.shift) / dataset.scale
        pred = model(torch.tensor(X.astype("float32")))
        
        cum_pred = cum_pred * 0.5 + pred
        loc = cum_pred.argmax()
        goal = sound_loc[sources[loc]]
        direction = (goal-pos) / np.linalg.norm(goal-pos)  if np.linalg.norm(goal-pos) > 1e-6 else np.zeros(2)
        pos = pos + direction * 10
        pos = find_nearest_pos(pos)
        robot.move_to(pos[0], pos[1])

execute()