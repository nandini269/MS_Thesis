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
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
import numpy as np
import random
# from adatune.utils import *


# def cli_def():
#     parser = argparse.ArgumentParser(description='CLI for running automated Learning Rate scheduler methods')
#     parser.add_argument('--network', type=str, default='vgg', choices=['resnet', 'vgg'])
#     parser.add_argument('--dataset', type=str, choices=['cifar_10', 'cifar_100'], default='cifar_10')
#     parser.add_argument('--num-epoch', type=int, default=200)
#     parser.add_argument('--batch-size', type=int, default=128)
#     parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
#     parser.add_argument('--lr', type=float, default=0.01)
#     parser.add_argument('--momentum', type=float, default=0.0)
#     parser.add_argument('--wd', type=float, default=5e-4)
#     parser.add_argument('--lr-scheduler', type=str, default='hd', choices=['hd', 'ed', 'cyclic', 'staircase'])
#     parser.add_argument('--hyper-lr', type=float, default=1e-8, help='beta, only applicable for HD')
#     parser.add_argument('--step-size', type=int, default=30, help='step-size, applicable for staircase')
#     parser.add_argument('--lr-decay', type=float, default=1.0, help='Decay factor, applicable for staircase and ED')
#     parser.add_argument('--t-0', type=int, default=10, help='T_0 for Cosine Annealing with Restarts')
#     parser.add_argument('--t-mult', type=int, default=2, help='T_Mult for Cosine Annealing with Restarts')
#     parser.add_argument('--model-loc', type=str, default='./model.pt')
#     parser.add_argument('--seed', type=int, default=42)
#     return parser


# def train_baselines(network_name, dataset, num_epoch, batch_size, optim_name, lr, momentum, wd, lr_scheduler_type,
#                     hyper_lr, step_size, lr_decay, t_0, t_mult, model_loc, seed):
#     torch.manual_seed(seed)

#     # We are using cuda for training - no point trying out on CPU for ResNet
#     device = torch.device("cuda")

#     net = network(network_name, dataset)
#     net.to(device).apply(init_weights)

#     # assign argparse parameters
#     criterion = nn.CrossEntropyLoss().to(device)
#     best_val_accuracy = 0.0
#     timestep = 0
#     cur_lr = lr
#     train_data, test_data = data_loader(network, dataset, batch_size)
#     lr_scheduler = None

#     if lr_scheduler_type == 'hd':
#         if optim_name == 'adam':
#             optimizer = AdamHD(net.parameters(), lr=lr, weight_decay=wd, eps=1e-4, hypergrad_lr=hyper_lr)
#         else:
#             optimizer = SGDHD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd, hypergrad_lr=hyper_lr)
#     else:
#         if optim_name == 'adam':
#             optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd, eps=1e-4)
#         else:
#             optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

#         if lr_scheduler_type == 'ed':
#             lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
#         elif lr_scheduler_type == 'staircase':
#             lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay)
#         elif lr_scheduler_type == 'cyclic':
#             lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult,
#                                                                                 eta_min=lr * 1e-4)

#     for epoch in range(num_epoch):
#         train_correct = 0
#         train_loss = 0

#         iter_len = len(train_data)

#         for i, (inputs, labels) in enumerate(train_data):
#             net.train()
#             timestep += 1

#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             train_loss += loss.item()

#             train_pred = outputs.argmax(1)
#             train_correct += train_pred.eq(labels).sum().item()

#             loss.backward()

#             optimizer.step()
#             optimizer.zero_grad()
#             if lr_scheduler and lr_scheduler_type == 'cyclic':
#                 lr_scheduler.step(epoch + (i / iter_len))

#         train_acc = 100.0 * (train_correct / len(train_data.dataset))
#         val_loss, val_acc = compute_loss_accuracy(net, test_data, criterion, device)

#         if val_acc > best_val_accuracy:
#             best_val_accuracy = val_acc
#             torch.save(net.state_dict(), model_loc)

#         if lr_scheduler and lr_scheduler_type != 'cyclic':
#             lr_scheduler.step(epoch)

#         print('train_accuracy at epoch :{} is : {}'.format(epoch, train_acc))
#         print('val_accuracy at epoch :{} is : {}'.format(epoch, val_acc))
#         print('best val_accuracy is : {}'.format(best_val_accuracy))

#         cur_lr = 0.0
#         for param_group in optimizer.param_groups:
#             cur_lr = param_group['lr']
#         print('learning_rate after epoch :{} is : {}'.format(epoch, cur_lr))

import pytorch_influence_functions as ptif
def get_influences_from_json(network_name):
    path = "models/"+network_name + '/' + "influence_results_0_1.json"
    d = json.load(open(path))
    return d
# Use same model with different hyperparameter settings
# network names available: wide_resnet, resnet, lenet, mlp, densenet, vgg
def experiment1(network_names, dataset, trainloader, testloader, batch_size = 128):#num_epoch, batch_size, optim_name, lr, momentum, wd, lr_scheduler_type,
                    #hyper_lr, step_size, lr_decay, t_0, t_mult, model_loc, seed):
    for network_name in network_names:
        # Supplied by the user:
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

def check_distribution(dataset, top_help_list):
    # Prints the distribution of the lables for the label-wise helpful indices from the influences json dictionary.
    labels = []
    img_list = []
    for ind in range(len(top_help_list)):
        # print(ind)
        i = top_help_list[ind]
        labels.append(dataset[i][1])
        img_list.append(dataset[i][0])
        if ind != 0 and (ind%500 == 0 or ind==len(top_help_list)-1):
            per_model_class = labels[ind-500:ind]
            a = Counter(per_model_class)
            print(a)
            torchvision.utils.save_image(img_list[ind-500:ind], "images/img_"+str(ind)+".jpg")
            # show(make_grid(img_list, padding=100))



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
    return super_trainloader

        # how to get the data from indices
#___________________________
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

#---------------------------------------
if __name__ == '__main__':
    # args = cli_def().parse_args()
    # print(args)

    # if os.path.exists(args.model_loc):
    #     os.remove(args.model_loc)

    # train_baselines(args.network, args.dataset, args.num_epoch, args.batch_size, args.optimizer, args.lr, args.momentum,
    #                 args.wd, args.lr_scheduler, args.hyper_lr, args.step_size, args.lr_decay, args.t_0, args.t_mult,
    #                 args.model_loc, args.seed)
    batch_size = 128
    network_names = ["vgg", "lenet","resnet"] 
    # network_names = ["models/vgg", "models/lenet","models/resnet"] 
    dataset = "cifar_10"
    model = network(network_names[0], dataset)  #how to pick hyperparameters?
    model.cuda()
    trainloader, testloader = data_loader(model, dataset, batch_size)
    # experiment1(network_names, "cifar_10", trainloader, testloader, batch_size)
    super_trainloader = create_super_ds(network_names, trainloader, batch_size)
    # train_models(network_names, "cifar_10", super_trainloader, testloader, batch_size)
    # print("-------------- Training and Testing using Original DS --------------")
    # train_models(network_names, "cifar_10", trainloader, testloader, batch_size)
    


