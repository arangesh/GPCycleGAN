import os
import json
from datetime import datetime
from statistics import mean
import argparse
import itertools

import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from models import Generator, Discriminator, SqueezeNet
from datasets import GazeDataset
from utils import gan2gaze, gaze2gan, plot_confusion_matrix


parser = argparse.ArgumentParser('Options for running inference using GazeNet/GazeNet++ in PyTorch...')
parser.add_argument('--dataset-root-path', type=str, default=None, help='path to dataset')
parser.add_argument('--split', type=str, default='val', help='split to evaluate (train/val/test)')
parser.add_argument('--version', type=str, default=None, help='which version of SqueezeNet to load (1_0/1_1)')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot-dir', type=str, default=None, help='directory with pre-trained model snapshots')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='batch size for training')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')


args = parser.parse_args()
# check args
if args.dataset_root_path is None:
    assert False, 'Path to dataset not provided!'
if all(args.version != x for x in ['1_0', '1_1']):
    assert False, 'Model version not recognized!'

# determine if ir or rgb data
if 'ir_' in args.dataset_root_path:
    args.data_type = 'ir'
    args.nc = 1
else:
    args.data_type = 'rgb'
    args.nc = 3

# Output class labels
activity_classes = ['Eyes Closed', 'Forward', 'Shoulder', 'Left Mirror', 'Lap', 'Speedometer', 'Radio', 'Rearview', 'Right Mirror']
merged_activity_classes = ['Eyes Closed/Lap', 'Forward', 'Left Mirror', 'Speedometer', 'Radio', 'Rearview', 'Right Mirror']
args.num_classes = len(activity_classes)

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M")
    args.output_dir = os.path.join('.', 'experiments', 'inference', args.output_dir)

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


kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 6}
test_loader = torch.utils.data.DataLoader(GazeDataset(args.dataset_root_path, args.split, False), **kwargs)


# validation function
def test(netG_B2A, netGaze):
    if netG_B2A is not None:
        netG_B2A.eval()
    netGaze.eval()
    pred_all = np.array([], dtype='int64')
    target_all = np.array([], dtype='int64')
    
    for idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data[:, :args.nc, :, :].cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # do the forward pass
        if netG_B2A is not None:
            data = gaze2gan(data, test_loader.dataset.mean[0:args.nc], test_loader.dataset.std[0:args.nc])
            fake_data = netG_B2A(data)
            fake_data = gan2gaze(fake_data, test_loader.dataset.mean[0:args.nc], test_loader.dataset.std[0:args.nc])
            scores = netGaze(fake_data.repeat(1, int(3 / args.nc), 1, 1))[0]
        else:
            scores = netGaze(data.repeat(1, int(3 / args.nc), 1, 1))[0]
        scores = scores.view(-1, args.num_classes)
        pred = scores.data.max(1)[1]  # got the indices of the maximum, match them
        print('Done with image {} out {}...'.format(min(args.batch_size*(idx+1), len(test_loader.dataset)), len(test_loader.dataset)))
        pred_all   = np.append(pred_all, pred.cpu().numpy())
        target_all = np.append(target_all, target.cpu().numpy())

    test_accuracy =  plot_confusion_matrix(target_all, pred_all, merged_activity_classes, args.output_dir)
    print("\n------------------------")
    print("Test accuracy = {:.2f}%\n------------------------".format(test_accuracy))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\n------------------------\n")
        f.write("Test accuracy = {:.2f}%\n------------------------\n".format(test_accuracy))

    return test_accuracy


if __name__ == '__main__':
    # networks
    netG_B2A = Generator(args.nc, args.nc)
    netGaze = SqueezeNet(args.version)
    
    if args.snapshot_dir is not None:
        if os.path.exists(os.path.join(args.snapshot_dir, 'netG_B2A.pth')):
            netG_B2A.load_state_dict(torch.load(os.path.join(args.snapshot_dir, 'netG_B2A.pth')), strict=False)
            if args.cuda:
                netG_B2A.cuda()
        else:
            netG_B2A = None
        if os.path.exists(os.path.join(args.snapshot_dir, 'netGaze.pth')):
            netGaze.load_state_dict(torch.load(os.path.join(args.snapshot_dir, 'netGaze.pth')), strict=False)
            if args.cuda:
                netGaze.cuda()
    else:
        assert False, 'No model snapshot provided!'

    test_acc = test(netG_B2A, netGaze)
    savemat(os.path.join(args.output_dir, 'accuracy.mat'), {'acc': test_acc})

    plt.close('all')
