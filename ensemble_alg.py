import torch.optim as optim
from collections import Counter
from data_loader import *
# from adatune.hd_adam import AdamHD
# from adatune.hd_sgd import SGDHD
from network import *
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
import numpy as np
import random

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# Prints per class test accuracy 
def test(testloader, net):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            #images, labels = data
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, pred = torch.max(outputs, 1)
            # print(outputs)
            # print(outputs.data) # test and print
            c = (pred == labels).squeeze()
            for i in range(4): # why is this 4? test and print
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return (correct/total)
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))


def train_and_eval_model(network_name, dataset, trainloader, valloader, batch_size, num_epochs):
    # needs to return model and validation error
    net = network(network_name, dataset)
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # same number of epochs same for both datasets?
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data
            inputs, labels = data[0].cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training '+network_name)
    val_loss = test(valloader, net)
    return net, val_loss
    # print('Testing ' + network_name)
    # test(testloader, net)

def get_ensemble_preds(ensemble, dataloader):
    # poor_subsets = []
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader:  # per batch
            predicteds = []
            images, labels = data[0].cuda(), data[1].cuda()
            for model in ensemble:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)  # get median predicted
                print(predicted.shape)
                predicteds.append(predicted)
            predicteds = torch.stack(predicteds, dim = 1)
            print(predicteds.shape)
            mode_vals, predicted_modes = torch.mode(predicteds) # fix
            total += labels.size(0)
            correct += (predicted_modes == labels).sum().item()
            # _, pred = torch.max(outputs, 1)
            # print(outputs)
            # print(outputs.data)
            # c = (pred == labels).squeeze()
            # for i in range(4): # why is this 4?
            #     label = labels[i]
            #     class_correct[label] += c[i].item()
            #     class_total[label] += 1
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return (correct/total)

def get_poor_subset(ensemble, trainloader,batch_size):
    poor_subsets = []
    with torch.no_grad():
        for data in trainloader:  # per batch
            predicteds = []
            images, labels = data[0].cuda(), data[1].cuda()
            for model in ensemble:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)  # get median predicted
                print(predicted.shape)
                predicteds.append(predicted)
            predicteds = torch.stack(predicteds, dim = 1)
            print(len(ensemble),predicteds.shape)
            mode_vals, predicted_modes = torch.mode(predicteds)
            print(predicted_modes.shape)
            poor_subset = images[predicted_modes!=labels]
            # print(poor_subset.shape)
            poor_subsets.append(poor_subset)
        print(len(poor_subsets))
    poor_loader = torch.utils.data.DataLoader(poor_subsets, shuffle=True, batch_size=batch_size, num_workers=1)
    return poor_loader

def get_mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=data_loc, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_loc, train=False, transform=transform, )
    testloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    # split into validation 
    train_size = round(0.75*len(dataset))
    val_size = len(dataset) - train_size 
    train, val = torch.utils.data.random_split(dataset, [train_size, val_size]) #generator=torch.Generator().manual_seed(42))
    trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=1)
    valloader = torch.utils.data.DataLoader(val, shuffle=True, batch_size=batch_size, num_workers=1)
    return trainloader,valloader,testloader

# what is the baseline? 
# num epochs?
# best model trained on entire dataset 
# pick random model to ensemble
# pick random subsample
# check time
# don't start with the entire dataset
# parameters- number of epochs, starting ratio, (0.1) subsample ratio, 
# parameters - weight of each model decaying learning rate, starting and total num models in ensemble
# starting with multiple parts of the dataset
# train on subsample, predict on entire training set
# decaying weights in the majority voting for the base learners gamma, gamma^2, gamma^3
# validation set to choose weights..least squares problem- sketch n solve that ??
# removing the last layer


def algorithm2_loop():    
    dataset = "mnist"          
    network_names = ["vgg11", "vgg13", "lenet","resnet18", "resnet34","mlp"] # use mlp just for mnist
    batch_size = 128
    num_epochs = 10
    trainloader,valloader,testloader = get_mnist(batch_size)
    ensemble = {}
    curr =  network_names[0]  #huh?
    model, val_loss = train_and_eval_model(curr, dataset, trainloader, valloader, batch_size, num_epochs) # don't use full dataset
    ensemble[model] = val_loss
    ensemble_nets = set()
    ensemble_nets.add(curr)
    while len(ensemble) < len(network_names) :
        # Take subset of points poorly predicted poor_subset
        poor_subset_loader = get_poor_subset(ensemble, trainloader, batch_size)
        val_losses = []
        models = []
        inds= []
        for i,network in enumerate(network_names): # maybe can just randomly select a model and train
            if network_name not in ensemble_nets: #are we okay repeating?
                # evaluate or train on subset and choose best
                model, val_loss = train_and_eval_model(network_name, dataset, poor_subset_loader, valloader, batch_size, num_epochs)
                models.append(model)
                val_losses.append(val_loss)   #do we want to pick based on val_loss?
                inds.append(i)
        best_ind = np.argmin(val_loss)
        best_model = models[best_ind]
        ensemble[best_model] = val_loss
        best_i = inds[best_ind]
        ensemble_nets.add(network_names[best_i])
    test_acc= get_ensemble_preds(ensemble, testloader)
    print(len(ensemble))
    print(test_acc)
    #test

    # Current model: look at gradient error w.r.t the parameters = residual
    
    # Pick best model for that subset
def algorithm2_random():    
    dataset = "mnist"          
    network_names = ["vgg11", "vgg13", "lenet","resnet18", "resnet34","mlp"] # use mlp just for mnist
    batch_size = 128
    num_epochs = 10
    trainloader,valloader,testloader = get_mnist(batch_size)
    ensemble = {}
    curr =  network_names[0]  #huh?
    model, val_loss = train_and_eval_model(curr, dataset, trainloader, valloader, batch_size, num_epochs) # don't use full dataset
    ensemble[model] = val_loss
    ensemble_nets = set()
    ensemble_nets.add(curr)
    while len(ensemble) < len(network_names) :
        # Take subset of points poorly predicted poor_subset
        poor_subset_loader = get_poor_subset(ensemble, trainloader, batch_size)
        val_losses = []
        models = []
        inds= []
        network_name = np.random.choice(network_names)
        # evaluate or train on subset and choose best
        model, val_loss = train_and_eval_model(network_name, dataset, poor_subset_loader, valloader, batch_size, num_epochs)
        models.append(model)
        val_losses.append(val_loss)   #do we want to pick based on val_loss?
        inds.append(i)

        best_ind = np.argmin(val_loss)
        best_model = models[best_ind]
        ensemble[best_model] = val_loss
        best_i = inds[best_ind]
        ensemble_nets.add(network_names[best_i])
    test_acc= get_ensemble_preds(ensemble, testloader)
    print(len(ensemble))
    print(test_acc)

if __name__ == '__main__':
    algorithm2_random()
