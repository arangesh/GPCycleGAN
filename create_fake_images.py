import os
import glob
import random
import time
import json
from datetime import datetime
from statistics import mean
import argparse

from PIL import Image
import numpy as np
from scipy.io import loadmat
import cv2

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from models import Generator
from utils import tensor2image


parser = argparse.ArgumentParser('Options for finetuning GazeNet++ in PyTorch...')
parser.add_argument('--dataset-root-path', type=str, default=None, help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot-dir', type=str, default=None, help='directory with pre-trained model snapshots')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')


args = parser.parse_args()
# check args
if args.dataset_root_path is None:
    assert False, 'Path to dataset not provided!'

# determine if ir or rgb data
args.dataset_root_path = os.path.normpath(args.dataset_root_path)
if 'ir_' in args.dataset_root_path:
    args.data_type = 'ir'
    args.nc = 1
else:
    args.data_type = 'rgb'
    args.nc = 3

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = os.path.join(os.path.dirname(args.dataset_root_path), os.path.basename(args.dataset_root_path) + '_fake')

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    assert False, 'Output directory already exists!'


# validation function
def infer(netG_B2A, im_path):
    transforms_ = [ transforms.Resize(args.size, Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ]
    transforms_ = transforms.Compose(transforms_)
    real_im = Image.open(im_path)
    data = transforms_(real_im)
    if args.cuda:
        data = data.cuda()
    data = data.unsqueeze(0)
    data = data[:, :args.nc, :, :]

    # do the forward pass
    fake_data = netG_B2A(data)
    fake_im = tensor2image(fake_data.detach(), np.array([0.5 for _ in range(args.nc)], dtype='float32'), 
        np.array([0.5 for _ in range(args.nc)], dtype='float32'))
    fake_im = np.transpose(fake_im, (1, 2, 0))
    fake_im = fake_im[:, :, ::-1]
    out_path = os.path.join(args.output_dir, im_path[len(args.dataset_root_path) + 1:])
    cv2.imwrite(out_path, fake_im) 
    return 


if __name__ == '__main__':
    # get the model, load pretrained weights, and convert it into cuda for if necessary
    netG_B2A = Generator(args.nc, args.nc)
    
    if args.snapshot_dir is not None:
        if os.path.exists(os.path.join(args.snapshot_dir, 'netG_B2A.pth')):
            netG_B2A.load_state_dict(torch.load(os.path.join(args.snapshot_dir, 'netG_B2A.pth')), strict=False)
        
    if args.cuda:
        netG_B2A.cuda()

    im_paths = sorted(glob.glob(os.path.join(args.dataset_root_path, '*', '*', '*.jpg')))
        
    for i, im_path in enumerate(im_paths):
        (head, tail) = os.path.split(im_path)
        os.makedirs(os.path.join(args.output_dir, head[len(args.dataset_root_path) + 1:]), exist_ok=True)
        infer(netG_B2A, im_path)
        print("Done creating image %d/%d" % (i+1, len(im_paths)))
