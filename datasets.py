import os
import glob
import random
import time

from PIL import Image
import numpy as np
from scipy.io import loadmat
import cv2

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_classification_data(dataset_root_path, split, activity_classes):
    images = []
    labels = []

    for idx, act in enumerate(activity_classes):
        dir_tmp = os.path.join(dataset_root_path, split, act, '*.*g') # .jpg/jpeg/.png
        tmp = sorted(glob.glob(dir_tmp))
        labels += [idx]*len(tmp)
        images += tmp

    print('Loaded %d eye images!' % len(labels))
    time.sleep(1)
    return images, labels

class GANDataset(Dataset):
    def __init__(self, opt, a_datasets, b_datasets, activity_classes, random_transforms=True, unaligned=False):
        #self.mean = loadmat(os.path.join(opt.dataset_root_path, 'all_data', 'mean_std.mat'))['mean'][0, 0]
        #self.std = loadmat(os.path.join(opt.dataset_root_path, 'all_data', 'mean_std.mat'))['std'][0, 0]
        self.mean = np.array([0.5 for _ in range(opt.nc)], dtype='float32')
        self.std = np.array([0.5 for _ in range(opt.nc)], dtype='float32')
        if random_transforms:
            transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std) ]
            self.transforms = transforms.Compose(transforms_)
        else:
            self.transforms = transforms.Normalize(self.mean, self.std)
        self.unaligned = unaligned

        self.images_A, self.targets_A = [], []
        for dset in a_datasets:
            images, targets = get_classification_data(os.path.join(opt.dataset_root_path, dset), 'train', activity_classes)
            self.images_A.extend(images)
            self.targets_A.extend(targets)
        self.images_B, self.targets_B = [], []
        for dset in b_datasets:
            images, targets = get_classification_data(os.path.join(opt.dataset_root_path, dset), 'train', activity_classes)
            self.images_B.extend(images)
            self.targets_B.extend(targets)

    def __getitem__(self, index):
        idx_A = index % len(self.images_A)
        item_A = self.transforms(Image.open(self.images_A[idx_A]))
        target_A = self.targets_A[idx_A]

        if self.unaligned:
            idx_B = random.randint(0, len(self.images_B) - 1)
            item_B = self.transforms(Image.open(self.images_B[idx_B]))
            target_B = self.targets_B[idx_B]
        else:
            idx_B = index % len(self.images_B)
            item_B = self.transforms(Image.open(self.images_B[idx_B]))
            target_B = self.targets_B[idx_B]

        return {'A': item_A, 'B': item_B, 'targets_A': target_A, 'targets_B': target_B}

    def __len__(self):
        return max(len(self.images_A), len(self.images_B))

class GazeDataset(Dataset):
    def __init__(self, dataset_root_path, activity_classes, split='train', random_transforms=False):
        'Initialization'
        print('Preparing '+split+' dataset...')
        self.split = split
        
        self.mean = loadmat(os.path.join(dataset_root_path, 'mean_std.mat'))['mean'][0]
        self.std = loadmat(os.path.join(dataset_root_path, 'mean_std.mat'))['std'][0]
        self.prepare_input = transforms.Compose([transforms.Resize(224), transforms.ToTensor()]) # ToTensor() normalizes image to [0, 1]
        self.normalize = transforms.Normalize(self.mean, self.std)
        if random_transforms:
            self.transforms = transforms.Compose([transforms.Resize(256),
                transforms.RandomRotation((-10, 10)), 
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ToTensor()]) # ToTensor() normalizes image to [0, 1]
        else:
            self.transforms = None

        self.images, self.labels = get_classification_data(dataset_root_path, self.split, activity_classes)
        print('Finished preparing '+split+' dataset!')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        y = self.labels[index]
        im = Image.fromarray(cv2.cvtColor(cv2.imread(self.images[index]), cv2.COLOR_BGR2RGB)) #cv2 loads 3 channel image by default

        if self.transforms is None:
            X = self.normalize(self.prepare_input(im))
        else:
            X = self.normalize(self.transforms(im))
        return X, y
