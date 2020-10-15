import os
import json
from datetime import datetime
from statistics import mean
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from models import SqueezeNet
from datasets import GazeDataset
from utils import plot_confusion_matrix


parser = argparse.ArgumentParser('Options for training GazeNet in PyTorch...')
parser.add_argument('--dataset-root-path', type=str, default=None, help='path to dataset')
parser.add_argument('--version', type=str, default=None, help='which version of SqueezeNet to load (1_0/1_1)')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=4e-4, metavar='LR', help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='WD', help='weight decay')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--random-transforms', action='store_true', default=False, help='apply random transforms to input while training')


args = parser.parse_args()
# check args
if args.dataset_root_path is None:
    assert False, 'Path to dataset not provided!'
if all(args.version != x for x in ['1_0', '1_1']):
    assert False, 'Model version not recognized!'

# Output class labels
activity_classes = ['Eyes Closed', 'Forward', 'Shoulder', 'Left Mirror', 'Lap', 'Speedometer', 'Radio', 'Rearview', 'Right Mirror']
merged_activity_classes = ['Eyes Closed/Lap', 'Forward', 'Left Mirror', 'Speedometer', 'Radio', 'Rearview', 'Right Mirror']
args.num_classes = len(activity_classes)

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M")
    args.output_dir = os.path.join('.', 'experiments', 'gazenet', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    assert False, 'Output directory already exists!'

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 6}
train_loader = torch.utils.data.DataLoader(GazeDataset(args.dataset_root_path, activity_classes, 'train', args.random_transforms), **kwargs)
val_loader = torch.utils.data.DataLoader(GazeDataset(args.dataset_root_path, activity_classes, 'val', False), **kwargs)

# global var to store best validation accuracy across all epochs
best_accuracy = 0.0


# training function
def train(netGaze, epoch):
    epoch_loss = list()
    correct = 0
    netGaze.train()
    for b_idx, (data, targets) in enumerate(train_loader):
        if args.cuda:
            data, targets = data.cuda(), targets.cuda()
        # convert the data and targets into Variable and cuda form
        data, targets = Variable(data), Variable(targets)

        # train the network
        optimizer.zero_grad()

        scores, masks = netGaze(data)
        scores = scores.view(-1, args.num_classes)
        loss = F.nll_loss(scores, targets)

        # compute the accuracy
        pred = scores.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data).cpu().sum()

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if b_idx % args.log_schedule == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * len(data), len(train_loader.dataset),
                100. * (b_idx+1)*len(data) / len(train_loader.dataset), loss.item()))
            with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
                f.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, (b_idx+1) * len(data), len(train_loader.dataset),
                100. * (b_idx+1)*len(data) / len(train_loader.dataset), loss.item()))

    # now that the epoch is completed calculate statistics and store logs
    avg_loss = mean(epoch_loss)
    print("------------------------\nAverage loss for epoch = {:.2f}".format(avg_loss))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\n------------------------\nAverage loss for epoch = {:.2f}\n".format(avg_loss))

    train_accuracy = 100.0*float(correct)/float(len(train_loader.dataset))
    print("Accuracy for epoch = {:.2f}%\n------------------------".format(train_accuracy))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("Accuracy for epoch = {:.2f}%\n------------------------\n".format(train_accuracy))
    
    return netGaze, avg_loss, train_accuracy


# validation function
def val(netGaze):
    global best_accuracy
    netGaze.eval()
    pred_all = np.array([], dtype='int64')
    target_all = np.array([], dtype='int64')
    
    for idx, (data, target) in enumerate(val_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # do the forward pass
        scores = netGaze(data)[0]
        scores = scores.view(-1, args.num_classes)
        pred = scores.data.max(1)[1]  # got the indices of the maximum, match them
        print('Done with image {} out of {}...'.format(min(args.batch_size*(idx+1), len(val_loader.dataset)), len(val_loader.dataset)))
        pred_all   = np.append(pred_all, pred.cpu().numpy())
        target_all = np.append(target_all, target.cpu().numpy())

    val_accuracy, _ = plot_confusion_matrix(target_all, pred_all, merged_activity_classes)
    print("\n------------------------")
    print("Validation accuracy = {:.2f}%\n------------------------".format(val_accuracy))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\n------------------------\n")
        f.write("Validation accuracy = {:.2f}%\n------------------------\n".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    if val_accuracy > best_accuracy:
        # save the model
        torch.save(netGaze.state_dict(), os.path.join(args.output_dir, 'netGaze.pth'))
        best_accuracy, _ = plot_confusion_matrix(target_all, pred_all, merged_activity_classes, args.output_dir)

    return val_accuracy


if __name__ == '__main__':
    # get the model, load pretrained weights, and convert it into cuda for if necessary
    netGaze = SqueezeNet(args.version, num_classes=args.num_classes)

    if args.snapshot is not None:
        netGaze.load_state_dict(torch.load(args.snapshot), strict=False)

    if args.cuda:
        netGaze.cuda()

    # create a temporary optimizer
    optimizer = optim.Adam(netGaze.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    fig1, ax1 = plt.subplots()
    plt.grid(True)
    train_loss = list()

    fig2, ax2 = plt.subplots()
    plt.grid(True)
    ax2.plot([], 'g', label='Train accuracy')
    ax2.plot([], 'b', label='Validation accuracy')
    ax2.legend()
    train_acc, val_acc = list(), list()
    for i in range(1, args.epochs+1):
        netGaze, avg_loss, acc = train(netGaze, i)
        # plot the loss
        train_loss.append(avg_loss)
        ax1.plot(train_loss, 'k')
        fig1.savefig(os.path.join(args.output_dir, "train_loss.jpg"))

        # plot the train and val accuracies
        train_acc.append(acc)
        val_acc.append(val(netGaze))
        ax2.plot(train_acc, 'g', label='Train accuracy')
        ax2.plot(val_acc, 'b', label='Validation accuracy')
        fig2.savefig(os.path.join(args.output_dir, 'trainval_accuracy.jpg'))
    plt.close('all')
