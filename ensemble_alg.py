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
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-d", "--dataset", type = "string", dest="dname", default = "cifar10")
parser.add_option("-f", "--filtered", dest="filtered", default = False)
parser.add_option("-i", "--num_iters", type = "int", dest="num_iters", default=5)
parser.add_option("-e", "--num_epochs", type = "int", dest="num_epochs", default=6)                        # change back
parser.add_option("-e", "--num_classes", type = "int", dest="num_classes", default=10)

opts,args = parser.parse_args()

# Prints per class test accuracy
def test(testloader, net):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    net.eval()
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
            for i in range(4): # why is this 4? test and print
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    # print('Accuracy of the network on the validation set: %d %%' % (
        # 100 * correct / total))
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
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #checkk

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
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
        print(epoch,running_loss)

    print('Finished Training '+network_name)
    val_loss = test(valloader, net)
    train_loss = test(trainloader, net)
    return net, val_loss, train_loss
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
                model.eval()
                val_loss = ensemble[model]
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)  # get median predicted
                predicteds.append(predicted)
            predicteds = torch.stack(predicteds, dim = 1)
            predicted_modes, mode_inds = torch.mode(predicteds) # fix
            total += labels.size(0)
            correct += (predicted_modes == labels).sum().item()
        # print("shape of stacked predicteds of models in ensemble",predicteds.shape)
            # _, pred = torch.max(outputs, 1)
            # c = (pred == labels).squeeze()
            # for i in range(4): # why is this 4?
            #     label = labels[i]
            #     class_correct[label] += c[i].item()
            #     class_total[label] += 1
    print("Accuracy of the ensemble on the {} images: {}".format(test_or_val,100*correct/total))
    return (correct/total)


def get_poor_subset(ensemble, trainloader, train, batch_size, cap_size, num_classes):
    # Take subset of points poorly predicted poor_subset
    corr_inds = {}
    indices = []
    labels = []  # maybe keep half of previous indices?
    with torch.no_grad():
        l_d = {}
        for i,data in enumerate(train):  # per batch_size
            predicteds = []
            image, label = data[0].cuda(), data[1]
            for model in ensemble:
                model.eval() # check
                output = model(torch.unsqueeze(image,0))
                _, predicted = torch.max(output.data, 1)  # get median predicted
                predicteds.append(predicted)
            predicteds = torch.stack(predicteds)
            predicted_mode, mode_ind = torch.mode(predicteds, dim = 0)
            # if i==0:
            #     print("predicted mode:",predicted_mode)
            #     print("true label:",label)
            #     print("predicteds",predicteds)
            if predicted_mode!= label:
                labels.append(label)
                if label in l_d:
                    l_d[label].append(i)
                else:
                    l_d[label] = [i]
            else:
                if label in corr_inds:
                    corr_inds[label].append(i)
                else:
                    corr_inds[label] = [i]
        # print("indices areeeeee:",indices)
        print("labels areeeeee:", Counter(labels))


    val_lens = []
    for l in l_d:
        val_lens.append(len(l_d[l]))
    sorted_lens = np.sort(val_lens)
    mid_len = int(round(np.median(sorted_lens)))

    print("sorted lens of poor subset",sorted_lens)
    # mid_i = round(len(sorted_lens)/2)  # can use np.median
    # mid_len = sorted_lens[mid_i]
    # balance datasets
    for l in range(num_classes):
        if l in l_d:  # l_d: label -> list of indices of wrongly predicted images
            indices.extend(np.random.choice(l_d[l],mid_len))
        else:
            indices.extend(np.random.choice(corr_inds[l],mid_len))
    print("mid_len:",mid_len)
    if len(indices)<cap_size/2:
        indices.extend(np.random.choice(np.arange(len(train)),round(0.7*cap_size)))
    if len(indices)>cap_size:
        indices = np.random.choice(indices, cap_size)
    subset = torch.utils.data.Subset(train, indices)
    # check_distribution(dataset,top_help_list)
    print("num images in poor subset: ",len(indices))
    poor_loader = torch.utils.data.DataLoader(subset, shuffle=True, batch_size=batch_size, num_workers=1)
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
def algorithm_random(data_all, dname, network_names, batch_size, num_epochs, filtered=True):
    train, val, trainloader,valloader,testloader = data_all # get_dataset(batch_size, dname, filtered) # get_mnist(batch_size)
    subsample_size = round(len(train)/len(network_names)) # round(0.1*len(train))
    ensemble = {}
    if dname == "cifar10" and filtered==True:
        num_classes = 2
    else:
        num_classes = 10
    train_sub, _ = torch.utils.data.random_split(train,[subsample_size,len(train)-subsample_size])
    tr_sub_ld = torch.utils.data.DataLoader(train_sub, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=1)
    model, val_loss, train_loss = train_and_eval_model(network_names[0], dname, tr_sub_ld, valloader, batch_size, num_epochs) # don't use full dataset
    ensemble[model] = val_loss
    ensemble_vals = [val_loss]
    val_losses = [val_loss]
    test_acc = get_ensemble_preds(ensemble, testloader,"test")
    test_accs = [test_acc]
    # models = [model]
    data_inds = set()
    for network_name in network_names[1:]:
        # network_name = np.random.choice(network_names)
        poor_subset_loader, indices = get_poor_subset(ensemble, trainloader, train, batch_size, subsample_size, num_classes)
        # print(indices)
        model, val_loss, train_loss = train_and_eval_model(network_name, dname, poor_subset_loader, valloader, batch_size, num_epochs)
        ens_acc = get_ensemble_preds(ensemble, valloader,"validation")
        data_inds.update(indices)
        # models.append(model)
        val_losses.append(val_loss)   # do we want to pick or weigh based on val_loss?
        ensemble[model] = val_loss
        ensemble_vals.append(ens_acc)
        test_acc = get_ensemble_preds(ensemble, testloader,"test")
        test_accs.append(test_acc)
    test_acc = get_ensemble_preds(ensemble, testloader,"test")
    test_accs.append(test_acc)
    data_prop = (len(data_inds)+subsample_size)/len(train)*100
    print(test_acc)
    print("data percentage used",data_prop)
    return val_losses, ensemble_vals, test_accs, data_prop

def algorithm2_random(data_all, dname, network_name, batch_size, num_epochs, filtered=True): # Uses same model
    train, val, trainloader,valloader,testloader = data_all # get_dataset(batch_size, dname, filtered) # get_mnist(batch_size)
    subsample_size = round(len(train)/opts.num_iters) # 5 iterations in total 7500/5 = 1500
    ensemble = {}
    if filtered==True:
        print("Filtered is TRUE")
        num_classes = 2
    else:
        num_classes = 10
    # set seed
    train_sub, _ = torch.utils.data.random_split(train,[subsample_size,len(train)-subsample_size])
    tr_sub_ld = torch.utils.data.DataLoader(train_sub, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=1)
    model, val_loss, train_loss = train_and_eval_model(network_name, dname, tr_sub_ld, valloader, batch_size, num_epochs) # don't use full dataset
    ensemble[model] = val_loss
    ensemble_vals = [val_loss]
    val_losses = [val_loss]
    train_losses = [train_loss]
    test_acc = get_ensemble_preds(ensemble, testloader,"test")
    test_accs = [test_acc]
    data_inds = []
    for i in range(opts.num_iters):
        poor_subset_loader, indices = get_poor_subset(ensemble, trainloader, train, batch_size, subsample_size, num_classes)
        model, val_loss, train_loss = train_and_eval_model(network_name, dname, poor_subset_loader, valloader, batch_size, num_epochs, trained_model = model)
        ens_acc = get_ensemble_preds(ensemble, valloader,"validation")
        data_inds.extend(indices)
        val_losses.append(val_loss)   # do we want to pick or weigh based on val_loss?
        train_losses.append(train_loss)
        ensemble[model] = val_loss
        ensemble_vals.append(ens_acc)
        test_acc = get_ensemble_preds(ensemble, testloader,"test")
        test_accs.append(test_acc)
    data_prop = (len(data_inds)+subsample_size)/len(train)*100
    print(test_acc)
    print("data percentage used",data_prop)
    return train_losses, val_losses, ensemble_vals, test_accs, data_prop

def baseline1(data_all, dname, network_names, batch_size, num_epochs, filtered):
    print("Baseline 1 results")
    # network_names = ["vgg11", "vgg13", "lenet","resnet18", "resnet34"]#"mlp"] # use mlp just for mnist
    # num_epochs = 20
    train, val, trainloader,valloader,testloader = data_all #get_dataset(batch_size, dname, filtered) #get_mnist(batch_size)
    network_name = np.random.choice(network_names)
    model, val_loss, train_loss = train_and_eval_model(network_name, dname, trainloader, valloader, batch_size, 5)
    test_loss = test(testloader, model)
    val_losses = [val_loss]
    test_losses = [test_loss]
    for i in range(4):
        model, val_loss, train_loss = train_and_eval_model(network_name, dname, trainloader, valloader, batch_size, 5, trained_model = model)
        val_losses.append(val_loss)
        test_loss = test(testloader, model)
        test_losses.append(test_loss)
    print("val loss:", val_loss)
    print("test loss:", test_losses[-1])
    return val_losses, test_losses

def baseline3(data_all, dname, network_name, batch_size, num_epochs, filtered): #random subset
    print("Baseline 3 results")
    train, val, trainloader,valloader,testloader = data_all #get_dataset(batch_size, dname, filtered)
    subsample_size = round(len(train)/opts.num_iters)
    train_sub, _ = torch.utils.data.random_split(train,[subsample_size,len(train)-subsample_size])
    tr_sub_ld = torch.utils.data.DataLoader(train_sub, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=1)
    model, val_loss, train_loss = train_and_eval_model(network_name, dname, tr_sub_ld, valloader, batch_size, num_epochs)
    test_loss = test(testloader, model)
    val_losses = [val_loss]
    test_losses = [test_loss]
    train_losses = [train_loss]
    for i in range(opts.num_iters):
        train_sub, _ = torch.utils.data.random_split(train,[subsample_size,len(train)-subsample_size])
        tr_sub_ld = torch.utils.data.DataLoader(train_sub, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=1)
        model, val_loss, train_loss = train_and_eval_model(network_name, dname, tr_sub_ld, valloader, batch_size, num_epochs, trained_model = model)
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        test_loss = test(testloader, model)
        test_losses.append(test_loss)
    print("val loss:", val_loss)
    print("test loss:", test_losses[-1])
    return train_losses, val_losses, test_losses

def baseline2(data_all, dname, network_name, batch_size, num_epochs, filtered): #random subset
    print("Baseline 2 results")
    # network_names = ["vgg11", "vgg13", "lenet","resnet18", "resnet34"]#"mlp"] # use mlp just for mnist
    # num_epochs = 20
    train, val, trainloader,valloader,testloader = data_all #get_dataset(batch_size, dname, filtered)
    model, val_loss, train_loss = train_and_eval_model(network_name, dname, trainloader, valloader, batch_size, num_epochs)
    test_loss = test(testloader, model)
    val_losses = [val_loss]
    test_losses = [test_loss]
    train_losses = [train_loss]
    for i in range(opts.num_iters):
        model, val_loss, train_loss = train_and_eval_model(network_name, dname, trainloader, valloader, batch_size, num_epochs, trained_model = model)
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        test_loss = test(testloader, model)
        test_losses.append(test_loss)
    print("val loss:", val_loss)
    print("test loss:", test_losses[-1])
    return train_losses, val_losses, test_losses


if __name__ == '__main__':
    filtered = opts.filtered
    print(filtered)
    batch_size = 128
    num_epochs = opts.num_epochs # 15
    num_iters = opts.num_iters # add argument
    dname = opts.dname
    data_all = get_dataset(batch_size, dname, filtered)
    pp = PdfPages('iref_'+dname+'_'+str(filtered)+'_'+str(num_epochs)+'_3.pdf')
    network_names_mnist = ["vgg11","vgg13","resnet18", "resnet34","mlp"] # use mlp just for mnist
    network_names = ["vgg11","vgg13","resnet18", "resnet34"]
    for i in range(3):  #need to plot means?
        network_name = "resnet18"#np.random.choice(network_names)
        e_trains, vals, ensemble_vals, e_tests, data_prop =  algorithm2_random(data_all, dname, network_name, batch_size, num_epochs, filtered)
        b_trains, b_vals, b_tests = baseline3(data_all, dname, network_name, batch_size, num_epochs, filtered)
        # plot it
        fig = plt.figure()
        xs = (np.arange(len(vals))+1)*5
        p1, = plt.plot(xs, vals, '-r', linewidth = 2, label = 'model val acc') # ind member val acc
        p2, = plt.plot(xs, e_trains,'ro', linewidth = 2, label = 'model train acc') # ensemble validation acc
        p5, = plt.plot(xs, e_tests, linestyle='dashdot', color = 'r', label = 'model test acc')
        p3, = plt.plot(xs, b_vals, '-c', linewidth = 2, label = 'baseline val acc') # make style same as above
        p6, = plt.plot(xs, b_trains,'co', linewidth = 2, label = 'baseline train acc')
        p4, = plt.plot(xs, b_tests, linestyle='dashdot', color = 'c', label = 'baseline test acc') #
        plt.title("Dataset:{} using {} percent and model:{}".format(dname,round(data_prop),network_name))
        # plt.xticks(xs,network_names)
        plt.xlabel("Number of epochs")
        plt.ylabel("Accuracy")
        plt.legend(handles=[p1, p3, p4, p2, p5, p6])#, bbox_to_anchor=(1.05, 1), loc='upper left')
        pp.savefig(fig)
    pp.close()
    # add bash script with all three experiments fed in
