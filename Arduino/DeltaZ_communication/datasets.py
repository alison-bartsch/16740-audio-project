import random
import os, sys
import numpy as np
import torch 
import json
import ipaddress
from os import read

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        sources = ["front", "back", "left", "right"]

        source_label = {"front": 0,
                        "back": 1,
                        "left": 2,
                        "right": 3}

        sep=5

        self.data = []
        self.label = []

        for location in sources:
            label = source_label[location]
           
            for i in range(10):
                data_file = "data/" + location + "_data_" + str(i) + ".npz"
                npzfile = np.load(data_file)
                data = npzfile["data"]
                for i in range(len(data)):
                    x = data[i][0][0]
                    y = data[i][0][1]
                    l = data[i][1][0]
                    r = data[i][1][1]
                    self.data.append(np.array([x,y,l,r]))
                    self.label.append(label)

        self.scale = np.max(self.data, axis=0) - np.min(self.data, axis=0)
        self.shift = np.min(self.data, axis=0).astype('float32')

        self.data = (np.array(self.data) - self.shift) / self.scale
        self.data = self.data.astype('float32')
        self.label = np.array(self.label)


    def __getitem__(self, idx):
        return (self.data[idx], self.label[idx])

    def __len__(self):
        return len(self.data)
