import argparse
import os
import json
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
import pytorch_influence_functions as ptif

# Gets influence dictionary from saved json file in models folder
def get_influences_from_json(network_name):
    path = "models/"+network_name + '/' + "influence_results_0_1.json"
    d = json.load(open(path))
    return d


# TODO: Implement same model with different hyperparameter settings
# network names available: wide_resnet, resnet, lenet, mlp, densenet, vgg

# Runs the calc_img_wise function from ptif library which saves influence output in json file in 'outdir'
def run_influence_calc(network_names, dataset, trainloader, testloader, batch_size = 128):
#num_epoch, batch_size, optim_name, lr, momentum, wd, lr_scheduler_type, #hyper_lr, step_size, lr_decay, t_0, t_mult, model_loc, seed):
    for network_name in network_names:
        print(network_name+" is being run")
        if not os.path.exists(network_name):
            os.makedirs(network_name)
        model = network(network_name, dataset)  #how to pick hyperparameters?
        model.cuda()
        # model.to(device).apply(init_weights)
        ptif.init_logging()
        config = ptif.get_default_config()
        config['outdir'] = network_name
        config['gpu'] = 1
        ptif.calc_img_wise(config, model, trainloader, testloader)

# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

# def show(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


# Prints the distribution of the labels for each set of helpful images found per class per model
# Also saves a grid with 500 images to depict the images chosen for that class for that model
def check_distribution(dataset, top_help_list):
    labels = []
    img_list = []
    for ind in range(len(top_help_list)):
        # print(ind)
        i = top_help_list[ind]
        labels.append(dataset[i][1])
        img_list.append(dataset[i][0])
        if ind != 0 and (ind%500 == 0 or ind==len(top_help_list)-1): 
        # every 500 helpful indices are for one class for one model
        # there are 10 classes and 3 models
            per_model_class = labels[ind-500:ind]
            a = Counter(per_model_class)
            print(a)
            torchvision.utils.save_image(img_list[ind-500:ind], "images/img_"+str(ind)+".jpg")
            # show(make_grid(img_list, padding=100))

# Creates a random trainloader for a randomly subsampled dataset. Size = 5000
def create_random_ds(network_names, trainloader, batch_size, sub_size = 5000):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    dataset = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transform_train)
    subset = torch.utils.data.Subset(dataset, np.arange(sub_size))
    random_trainloader = torch.utils.data.DataLoader(subset, shuffle=False, batch_size=batch_size, num_workers=1)
        return random_trainloader


# Creates a trainloader for the "super dataset." The super dataset comprises of the 500 most helpful images
# per class per model. (Since we have 3 models and 10 classes this dataset is 3 * 10 * 500)
# TODO: Must check that there is no randomization that is disturbing the indexing
def create_super_ds(network_names, trainloader, batch_size):
    top_help_set = set()
    top_help_list = []
    for network_name in network_names:
        influence_d = get_influences_from_json(network_name)
        for i in range(10):
            top_help = influence_d[str(i)]["helpful"]  # these are 500
            top_help_set.update(top_help)
            top_help_list.extend(top_help)

    print("length of list "+str(len(top_help_list)))
    print("length of set "+str(len(top_help_set)))

    a = list(map(lambda x:x ,trainloader))
    helpful_l = list(top_help_set)
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    dataset = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transform_train)
    subset = torch.utils.data.Subset(dataset, helpful_l)
    check_distribution(dataset,top_help_list)
    super_trainloader = torch.utils.data.DataLoader(subset, shuffle=False, batch_size=batch_size, num_workers=1)
    return super_trainloader

    # img_ind = helpful_l[0]
    # batch_num = int(img_ind/batch_size)
    # batch_id = img_ind - (batch_size*batch_num)
    # image = a[batch_num][0][batch_id]
    # label = a[batch_num][1][batch_id]
    # image = image.unsqueeze_(0)
    
    # new_train_set = []
    # for i in range(1,len(helpful_l)):
    #     img_ind = helpful_l[i]
    #     batch_num = int(img_ind/batch_size)
    #     batch_id = img_ind - (batch_size*batch_num)
    #     image = a[batch_num][0][batch_id]
    #     label = a[batch_num][1][batch_id]
    #     new_train_set.append((image,int(label)))

    # t_ind = int(0.75*len(new_train_set))  #make sure it is balanced?
    # super_trainset = InfluenceDataset(new_train_set)
    # super_trainloader = torch.utils.data.DataLoader(super_trainset, batch_size = batch_size, shuffle = False)
    # # super_valset = InfluenceDataset(new_train_set[t_ind:])
    # # super_valloader = torch.utils.data.DataLoader(super_valset, batch_size = batch_size, shuffle = False)
    # return super_trainloader

#___________________________
# Trains all the models in the pool on the dataloader provided and calls test function to see the per class accuracy
def train_models(network_names, dataset, trainloader, testloader, batch_size):
    for network_name in network_names:
        net = network(network_name, dataset)
        net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(30):  # number of epochs same for both datasets?
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
        print('Testing ' + network_name)
        test(testloader, net)



def save_model(net):
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


def load_model():
    PATH = './cifar_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.cuda()
    return net

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
            c = (pred == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def influence_dataset():
    batch_size = 128
    network_names = ["vgg", "lenet","resnet"] 
    # network_names = ["models/vgg", "models/lenet","models/resnet"] 
    dataset = "cifar_10"
    model = network(network_names[0], dataset)  #how to pick hyperparameters?
    model.cuda()
    trainloader, testloader = data_loader(model, dataset, batch_size)
    # run_influence_calc(network_names, "cifar_10", trainloader, testloader, batch_size)
    super_trainloader = create_super_ds(network_names, trainloader, batch_size)
    train_models(network_names, "cifar_10", super_trainloader, testloader, batch_size)
    # print("-------------- Training and Testing using Original DS --------------")
    # train_models(network_names, "cifar_10", trainloader, testloader, batch_size)

def baseline_dataset():
    # create function that tests accuracy on a randomly subsampled, well distributed dataset
    batch_size = 128
    network_names = ["vgg", "lenet","resnet"] 
    # network_names = ["models/vgg", "models/lenet","models/resnet"] 
    dataset = "cifar_10"
    model = network(network_names[0], dataset)  #how to pick hyperparameters?
    model.cuda()
    trainloader, testloader = data_loader(model, dataset, batch_size)
    # experiment1(network_names, "cifar_10", trainloader, testloader, batch_size)
    random_trainloader = create_random_ds(network_names, trainloader, batch_size)
    train_models(network_names, "cifar_10", random_trainloader, testloader, batch_size)

def full_dataset():
    batch_size = 128
    network_names = ["vgg", "lenet","resnet"] 
    # network_names = ["models/vgg", "models/lenet","models/resnet"] 
    dataset = "cifar_10"
    model = network(network_names[0], dataset)  #how to pick hyperparameters?
    model.cuda()
    trainloader, testloader = data_loader(model, dataset, batch_size)
    print("-------------- Training and Testing using Original DS --------------")
    train_models(network_names, "cifar_10", trainloader, testloader, batch_size)
#---------------------------------------
if __name__ == '__main__':
    # influence_dataset()
    baseline_dataset()
    # full_dataset()
    



