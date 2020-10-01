import random
import time
import datetime
import os
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
import numpy as np
from visdom import Visdom

from torch.autograd import Variable
import torch


def gan2gaze(tensor, mean, std):
    mean = mean[np.newaxis, ..., np.newaxis, np.newaxis] # (1, nc, 1, 1)
    mean = np.tile(mean, (tensor.size()[0], 1, tensor.size()[2], tensor.size()[3])) # (B, nc, H, W)
    mean = torch.from_numpy(mean.astype(np.float32)).cuda()
    std = std[np.newaxis, ..., np.newaxis, np.newaxis] # (1, nc, 1, 1)
    std = np.tile(std, (tensor.size()[0], 1, tensor.size()[2], tensor.size()[3])) # (B, nc, H, W)
    std = torch.from_numpy(std.astype(np.float32)).cuda()
    return (tensor*0.5+0.5 - mean)/std

def gaze2gan(tensor, mean, std):
    mean = mean[np.newaxis, ..., np.newaxis, np.newaxis] # (1, nc, 1, 1)
    mean = np.tile(mean, (tensor.size()[0], 1, tensor.size()[2], tensor.size()[3])) # (B, nc, H, W)
    mean = torch.from_numpy(mean.astype(np.float32)).cuda()
    std = std[np.newaxis, ..., np.newaxis, np.newaxis] # (1, nc, 1, 1)
    std = np.tile(std, (tensor.size()[0], 1, tensor.size()[2], tensor.size()[3])) # (B, nc, H, W)
    std = torch.from_numpy(std.astype(np.float32)).cuda()
    return (tensor*std+mean - 0.5)/0.5

def tensor2image(tensor, mean, std):
    mean = mean[..., np.newaxis, np.newaxis] # (nc, 1, 1)
    mean = np.tile(mean, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)
    std = std[..., np.newaxis, np.newaxis] # (nc, 1, 1)
    std = np.tile(std, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)

    image = 255.0*(std*tensor[0].cpu().float().numpy() + mean) # (nc, H, W)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8) # (3, H, W)

class Logger():
    def __init__(self, n_epochs, batches_epoch, mean=0.0, std=1.0):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.mean = mean
        self.std = std

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        #sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            #if (i+1) == len(losses.keys()):
                #sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            #else:
                #sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        #sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data, self.mean, self.std), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data, self.mean, self.std), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            #sys.stdout.write('\n')
        else:
            self.batch += 1        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def plot_confusion_matrix(y_true, y_pred, classes, output_dir=None, normalize=True, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # merge "Eyes Closed" and "Lap" classes
    y_true[y_true == 4] = 0
    y_pred[y_pred == 4] = 0
    # change GT "Shoulder" to "Left Mirror"
    y_true[np.logical_and(y_true == 2, y_pred == 3)] = 3
    # change GT "Shoulder" to "Right Mirror"
    y_true[np.logical_and(y_true == 2, y_pred == 8)] = 8
    # change prediction "Shoulder" to "Left Mirror"
    y_pred[np.logical_and(y_pred == 2, y_true == 3)] = 3
    # change prediction "Shoulder" to "Right Mirror"
    y_pred[np.logical_and(y_pred == 2, y_true == 8)] = 8
    # remove "Shoulder" class
    retain = np.logical_and(y_pred != 2, y_true != 2)
    y_true = y_true[retain]
    y_pred = y_pred[retain]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, 'confusion_matrix.jpg'))
    plt.close(fig)
    return 100.0*accuracy_score(y_true, y_pred)
