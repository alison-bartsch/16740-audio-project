import importlib
from numpy.core.numeric import Inf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import datasets

import matplotlib.patches as patches
from model import FC
torch.set_printoptions(precision=10)

def train(dataloader, model, loss_fn, optimizer, device, acc_cnt_fn, verbose=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        correct += acc_cnt_fn(pred, y)
        if verbose and batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    correct = correct * 1. / size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    return correct, train_loss

def test(dataloader, model, loss_fn, device, acc_cnt_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    digit_size, digit_correct = 0, 0
    positive = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += acc_cnt_fn(pred, y)

    test_loss /= num_batches
    correct = correct * 1. / size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def train_and_test(task):
    
    print("Using {} device".format(task.device))

    # Create data loaders.
    train_dataloader = DataLoader(task.training_data, batch_size=task.batch_size)
    test_dataloader = DataLoader(task.testing_data, batch_size=task.batch_size)

    train_accs = []
    train_losses = []

    test_accs = []
    test_losses = []
    
    for t in range(task.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        acc, loss = train(train_dataloader, task.model, task.loss_fn, task.optimizer, task.device, task.acc_cnt_fn)
        train_accs.append(acc)
        train_losses.append(loss)

        acc, loss = test(test_dataloader, task.model, task.loss_fn, task.device, task.acc_cnt_fn)
        test_accs.append(acc)
        test_losses.append(loss)

    print("Done!")

    return train_accs, train_losses, test_accs, test_losses

class Task():
    def __init__(self):
        self.set_seed()
        self.training_data, self.testing_data = self.get_data()
        self.model = self.get_model()
        self.set_learning_params()
    
    def set_seed(self):
        torch.manual_seed(1)

    def set_learning_params(self):
        raise NotImplemented
        
    def is_finished(self, results, acc):
        raise NotImplemented
    
    def acc_cnt_fn(self, pred, y):
        raise NotImplemented

    def get_model(self):
        raise NotImplemented

    def get_data(self):
        raise NotImplemented

class OneStepEstimate(Task):
    # input: 4d, xy-position, volumes of two microphones
    # output: 4d, probability of the source at different locations.

    def set_learning_params(self):
        self.save_prefix="./models/"
        self.device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 1000
        self.batch_size = 100000
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def is_finished(self, results, acc):
        return acc > 0.9
    
    def acc_cnt_fn(self, pred, y):
        return (pred.argmax(1) == y).type(torch.float).sum().item()

    def get_model(self):
        return FC(2, 4, 1000, 4)

    def get_data(self):
        normalize=True
        data_path = './data/'
        training_data = datasets.OneStepDataset(data_path, test=False)
        testing_data = datasets.OneStepDataset(data_path, test=True)
        return training_data, testing_data

    def save_model(self, save_name):
        torch.save(self.model.state_dict(), self.save_prefix+"/"+save_name+".pth")

class TwoStepEstimate(Task):
    # input: 4d, x2-x1, y2-y1, l2-l1, r2-r1, l1-r1, l2-r2
    # output: 4d, probability of the source at different locations.

    def set_learning_params(self):
        self.save_prefix="./models/"
        self.device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 1000
        self.batch_size = 100000
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def is_finished(self, results, acc):
        return acc > 0.9
    
    def acc_cnt_fn(self, pred, y):
        return (pred.argmax(1) == y).type(torch.float).sum().item()

    def get_model(self):
        return FC(2, 4, 1000, 4)

    def get_data(self):
        normalize=True
        data_path = './data/'
        data = datasets.TwoStepDataset(data_path)
        return data, data

    def save_model(self, save_name):
        torch.save(self.model.state_dict(), self.save_prefix+"/"+save_name+".pth")

if __name__ == "__main__":
    task = OneStepEstimate()
    train_accs, train_losses, test_accs, test_losses = train_and_test(task)
    
    task.save_model("FC3-1000-8:2")