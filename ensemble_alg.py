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
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# from multiprocessing import set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

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
    print('Accuracy of the network on the validation set: %d %%' % (
        100 * correct / total))
    return (correct/total)
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))


def train_and_eval_model(network_name, dataset, trainloader, valloader, batch_size, num_epochs, trained_model = None):
    # needs to return model and validation error
    if trained_model is None:
        net = network(network_name, dataset)
        net.cuda()
    else:
        net = trained_model
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

def get_ensemble_preds(ensemble, dataloader, test_or_val):
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
                val_loss = ensemble[model]
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)  # get median predicted
                predicteds.append(predicted)
            predicteds = torch.stack(predicteds, dim = 1)
            predicted_modes, mode_inds = torch.mode(predicteds) # fix
            total += labels.size(0)
            correct += (predicted_modes == labels).sum().item()
        # print("shape of stacked predicteds of models in ensemble",predicteds.shape)
        # print(predicted_modes)
        # print(labels)
            # _, pred = torch.max(outputs, 1)
            # print(outputs)
            # print(outputs.data)
            # c = (pred == labels).squeeze()
            # for i in range(4): # why is this 4?
            #     label = labels[i]
            #     class_correct[label] += c[i].item()
            #     class_total[label] += 1
    print("Accuracy of the ensemble on the {} images: {}".format(test_or_val,100*correct/total))
    # print("correct:", correct)
    # print("total:",total)
    return (correct/total)

def get_poor_subset_og(ensemble, trainloader, train, batch_size, cap_size):
    # Take subset of points poorly predicted poor_subset
    poor_subsets = []
    indices = []
    with torch.no_grad():
        for i,data in enumerate(trainloader):  # per batch_size
            inds = torch.arange(i*batch_size,i*batch_size+data[1].shape[0])
            predicteds = []
            images, labels = data[0].cuda(), data[1].cuda()
            for model in ensemble:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)  # get median predicted
                predicteds.append(predicted)
            predicteds = torch.stack(predicteds, dim = 1)
            # print(len(ensemble),predicteds.shape)
            predicted_modes, mode_inds = torch.mode(predicteds)
            # if i==3:
            #     print("predicted modes:",predicted_modes)
            #     print("true labels:",labels)
            #     print("predicted modes shape",predicted_modes.shape )
            poor_subsets.extend(images[predicted_modes!=labels])
            indices.extend(inds[predicted_modes!=labels])
            # print(poor_subset.shape)
            # poor_subsets.append(poor_subset)
        print("num images in poor subset: ",len(poor_subsets))
    if len(indices)>cap_size:
        indices = indices[:cap_size]
    subset = torch.utils.data.Subset(train, indices)
    # check_distribution(dataset,top_help_list)
    poor_loader = torch.utils.data.DataLoader(subset, shuffle=True, batch_size=batch_size, num_workers=1)
    # poor_loader = torch.utils.data.DataLoader(poor_subsets, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=1)
    return poor_loader, indices

    def get_poor_subset(ensemble, trainloader, train, batch_size, cap_size):
    # Take subset of points poorly predicted poor_subset
    poor_subsets = []
    indices = []
    with torch.no_grad():
        for i,data in enumerate(train):  # per batch_size
            # inds = torch.arange(i*batch_size,i*batch_size+data[1].shape[0])
            predicteds = []
            image, label = data[0].cuda(), data[1].cuda()
            for model in ensemble:
                output = model(images)
                _, predicted = torch.max(outputs.data, 1)  # get median predicted
                predicteds.append(predicted)
            predicteds = torch.stack(predicteds, dim = 1)
            # print(len(ensemble),predicteds.shape)
            predicted_mode, mode_ind = torch.mode(predicteds)
            # if i==3:
            #     print("predicted modes:",predicted_modes)
            #     print("true labels:",labels)
            #     print("predicted modes shape",predicted_modes.shape )
            if predicted_mode!= label:
                poor_subsets.append(image)
                indices.append(i)
            # print(poor_subset.shape)
            # poor_subsets.append(poor_subset)
        print("num images in poor subset: ",len(poor_subsets))
    if len(indices)>cap_size:
        indices = np.random.choice(indices, cap_size)
    subset = torch.utils.data.Subset(train, indices)
    # check_distribution(dataset,top_help_list)
    poor_loader = torch.utils.data.DataLoader(subset, shuffle=True, batch_size=batch_size, num_workers=1)
    # poor_loader = torch.utils.data.DataLoader(poor_subsets, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=1)
    return poor_loader, indices

def get_mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=data_loc, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_loc, train=False, transform=transform, )
    testloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True) #not sure
    # split into validation 
    train_size = round(0.75*len(dataset))
    val_size = len(dataset) - train_size 
    train, val = torch.utils.data.random_split(dataset, [train_size, val_size]) #generator=torch.Generator().manual_seed(42))
    trainloader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(val, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=1)
    return train, val, trainloader,valloader,testloader

def filter_cifar10(dataset, batch_size):
    new_dataset_inds = []
    for i,ds in enumerate(dataset):
        data,label = ds
        if label == 0 or label==1:
            new_dataset_inds.append(i)
    subset = torch.utils.data.Subset(dataset, new_dataset_inds)
    # check_distribution(dataset,top_help_list)
    return subset

def get_cifar10(batch_size,filter=True):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    dataset = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transform_train)    
    test_dataset = datasets.CIFAR10(root=data_loc, train=False, transform=transform_test)
    # print("original lengths of dataset",len(dataset),len(test_dataset))
    if filter:
        print("filter is on")
        dataset = filter_cifar10(dataset, batch_size)
        test_dataset = filter_cifar10(test_dataset, batch_size)
        # print("new lengths of datasets",len(dataset),len(test_dataset))
    testloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    train_size = round(0.75*len(dataset))
    val_size = len(dataset) - train_size 
    train, val = torch.utils.data.random_split(dataset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(val, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=1)
    return train, val, trainloader,valloader,testloader
# what is the baseline? 
# num epochs? 5 for now
# best model trained on entire dataset 
# pick random model to ensemble
# pick random subsample ..
# check time
# don't start with the entire dataset
# parameters- number of epochs, starting ratio, (0.1) subsample ratio or cap(1/num_models * len(data)), 
# parameters - weight of each model decaying learning rate, starting and total num models in ensemble
# starting with multiple parts of the dataset
# train on subsample, predict on entire training set
# decaying weights in the majority voting for the base learners gamma, gamma^2, gamma^3
# validation set to choose weights..least squares problem- sketch n solve that ??
# removing the last layer

def get_dataset(batch_size, dname, filtered):
    if dname == "mnist":
        return get_mnist(batch_size)
    elif dname == "cifar10":
        return get_cifar10(batch_size, filtered)
    # Current model: look at gradient error w.r.t the parameters = residual
    
    # Pick best model for that subset
def algorithm2_random(dname, network_names, batch_size, num_epochs, filtered=True):         
    train, val, trainloader,valloader,testloader = get_dataset(batch_size, dname, filtered) # get_mnist(batch_size)
    subsample_size = round(len(train)/len(network_names)) #round(0.1*len(train))
    # print(subsample_size)
    ensemble = {}
    train_sub, _ = torch.utils.data.random_split(train,[subsample_size,len(train)-subsample_size])
    tr_sub_ld = torch.utils.data.DataLoader(train_sub, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=1)
    model, val_loss = train_and_eval_model(network_names[0], dname, tr_sub_ld, valloader, batch_size, num_epochs) # don't use full dataset
    ensemble[model] = val_loss
    val_losses = [val_loss]
    ensemble_vals = [val_loss]
    models = [model]
    data_inds = set()
    # while len(ensemble) < len(network_names) :
    for network_name in network_names[1:]:
        # network_name = np.random.choice(network_names)
        poor_subset_loader, indices = get_poor_subset(ensemble, trainloader, train, batch_size, subsample_size)
        # print(indices)
        model, val_loss = train_and_eval_model(network_name, dname, poor_subset_loader, valloader, batch_size, num_epochs)
        ens_acc = get_ensemble_preds(ensemble, valloader,"validation")
        data_inds.update(indices)
        models.append(model)
        val_losses.append(val_loss)   #do we want to pick or weigh based on val_loss?
        ensemble[model] = val_loss
        ensemble_vals.append(ens_acc)
    test_acc = get_ensemble_preds(ensemble, testloader,"test")
    data_prop = (len(data_inds)+subsample_size)/len(train)*100
    print(test_acc)
    print("data percentage used",data_prop)
    return val_losses, ensemble_vals, test_acc, data_prop

def baseline1(dname, network_names, batch_size, filtered):
    print("Baseline 1 results")
    # network_names = ["vgg11", "vgg13", "lenet","resnet18", "resnet34"]#"mlp"] # use mlp just for mnist
    num_epochs = 25
    train, val, trainloader,valloader,testloader = get_dataset(batch_size, dname, filtered) #get_mnist(batch_size)
    network_name = "lenet" #np.random.choice(network_names)
    model, val_loss = train_and_eval_model(network_name, dname, trainloader, valloader, batch_size, 5)
    val_losses = [val_loss]
    for i in range(4):
        model, val_loss = train_and_eval_model(network_name, dname, trainloader, valloader, batch_size, 5, trained_model = model)
        val_losses.append(val_loss)
    test_loss = test(testloader, model)
    print("val loss:", val_loss)
    print("test loss:", test_loss)
    return val_losses, test_loss

if __name__ == '__main__':
    pp = PdfPages('iterative_refinement_plots.pdf')
    filtered = True
    batch_size = 128
    num_epochs = 5 # 15
    dname = "cifar10"
    # network_names = ["vgg11","resnet18", "resnet34"] filter false
    network_names = ["vgg11", "vgg13", "lenet","resnet18", "resnet34"]#"mlp"] # use mlp just for mnist
    for i in range(1):  #need to plot means?
        # np.random.shuffle(network_names)
        vals, ensemble_vals, e_test, data_prop =  algorithm2_random(dname, network_names, batch_size, num_epochs, filtered)
        b_vals, b_test = baseline1(dname, network_names, batch_size, filtered)
        # plot it
        fig = plt.figure()
        xs = np.arange(len(vals))
        p1, = plt.plot(xs, vals, 'bo', label = 'model val acc') # ind member val acc
        p2, = plt.plot(xs, ensemble_vals,'-r', linewidth = 4, label = 'ensemble val acc') # ensemble validation acc
        p3, = plt.plot(xs, b_vals, linestyle='dashed', color = 'c', label = 'baseline val acc') # make style same as above
        p4, = plt.plot([b_test]*len(b_vals), linestyle='dashdot', color = 'c', label = 'baseline test acc') # 
        p5, = plt.plot([e_test]*len(b_vals), linestyle='dashdot', color = 'r', label = 'ensemble test acc')
        plt.title("Dataset:{} using {} and num_models:{}".format(dname,round(data_prop),len(network_names)))
        plt.xticks(xs,network_names)
        plt.ylabel("Accuracy")
        plt.legend(handles=[p1, p2, p3, p4, p5], title='title')#, bbox_to_anchor=(1.05, 1), loc='upper left')
    pp.savefig(fig)
    pp.close()
