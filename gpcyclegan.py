import os
import json
from datetime import datetime
from statistics import mean
import argparse
import itertools

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from models import Generator, Discriminator, SqueezeNet
from datasets import GANDataset, GazeDataset
from utils import ReplayBuffer, LambdaLR, Logger, gan2gaze, gaze2gan, plot_confusion_matrix


parser = argparse.ArgumentParser('Options for training GPCycleGAN in PyTorch...')
parser.add_argument('--dataset-root-path', type=str, default=None, help='path to dataset')
parser.add_argument('--data-type', type=str, default='ir', help='which data type to load (ir/rgb)')
parser.add_argument('--version', type=str, default=None, help='which version of SqueezeNet to load (1_0/1_1)')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot-dir', type=str, default=None, help='directory with pre-trained model snapshots')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='batch size for training')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=0.0002, metavar='LR', help='learning rate')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--random-transforms', action='store_true', default=False, help='apply random transforms to input while training')
parser.add_argument('--decay-epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--train-gaze', action='store_true', default=False, help='train GazeNet simultaneously')


args = parser.parse_args()
# check args
if args.dataset_root_path is None:
    assert False, 'Path to dataset not provided!'
if args.data_type == 'ir':
    args.nc = 1
elif args.data_type == 'rgb':
    args.nc = 3
else:
    assert False, 'Incorrect data type specified!'
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
    args.output_dir = os.path.join('.', 'experiments', 'gpcyclegan', args.output_dir)

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
train_loader = torch.utils.data.DataLoader(GANDataset(args, [args.data_type + '_no_glasses'], [args.data_type + '_with_glasses'], random_transforms=args.random_transforms, unaligned=True), **kwargs)
val_loader = torch.utils.data.DataLoader(GazeDataset(os.path.join(args.dataset_root_path, args.data_type + '_all_data'), 'val', False), **kwargs)

# global var to store best validation accuracy across all epochs
best_accuracy = 0.0


# training function
def train(netG_A2B, netG_B2A, netD_A, netD_B, netGaze, epoch):
    epoch_loss = list()
    correct = 0
    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()
    netGaze.train()
    for b_idx, batch in enumerate(train_loader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A'])) # (B, C, H, W)
        real_B = Variable(input_B.copy_(batch['B'])) # (B, C, H, W)
        targets_A, targets_B = batch['targets_A'], batch['targets_B']
        if args.cuda:
            targets_A, targets_B = targets_A.cuda(), targets_B.cuda()

        ###### Generators A2B, B2A and GazeNet ######
        optimizer_G.zero_grad()
        optimizer_gaze.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real.expand_as(pred_fake))

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real.expand_as(pred_fake))

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
        
        # Gaze consistency loss
        real_A_gaze = gan2gaze(real_A, val_loader.dataset.mean[0:args.nc], val_loader.dataset.std[0:args.nc])
        _, masks_real_A = netGaze(real_A_gaze.repeat(1, int(3 / args.nc), 1, 1))
        recovered_A_gaze = gan2gaze(recovered_A, val_loader.dataset.mean[0:args.nc], val_loader.dataset.std[0:args.nc])
        _, masks_rec_A = netGaze(recovered_A_gaze.repeat(1, int(3 / args.nc), 1, 1))
        loss_gaze = criterion_gaze(masks_real_A, masks_rec_A)*10.0

        # compute the train accuracy of (netB2A-->netGaze) model
        same_A_gaze = gan2gaze(same_A, val_loader.dataset.mean[0:args.nc], val_loader.dataset.std[0:args.nc])
        scores_same_A, _ = netGaze(same_A_gaze.repeat(1, int(3 / args.nc), 1, 1))
        fake_A_gaze = gan2gaze(fake_A, val_loader.dataset.mean[0:args.nc], val_loader.dataset.std[0:args.nc])
        scores_fake_A, _ = netGaze(fake_A_gaze.repeat(1, int(3 / args.nc), 1, 1))

        scores_same_A = scores_same_A.view(-1, args.num_classes)
        pred = scores_same_A.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets_A.data).cpu().sum()
        scores_fake_A = scores_fake_A.view(-1, args.num_classes)
        pred = scores_fake_A.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets_B.data).cpu().sum()

        # Total loss for Generators and GazeNet
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_gaze
        loss_G.backward()
        
        optimizer_G.step()
        optimizer_gaze.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real.expand_as(pred_real))

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake.expand_as(pred_fake))

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real.expand_as(pred_real))
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake.expand_as(pred_fake))

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_gaze': loss_gaze, 'loss_D': (loss_D_A + loss_D_B)}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B, 'recovered_A': recovered_A, 'recovered_B': recovered_B})

        loss = loss_G + (loss_D_A + loss_D_B)
        epoch_loss.append(loss.item())
        if b_idx % args.log_schedule == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * args.batch_size, len(train_loader.dataset),
                100. * (b_idx+1) * args.batch_size / len(train_loader.dataset), loss.item()))
            with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
                f.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, (b_idx+1) * args.batch_size, len(train_loader.dataset),
                100. * (b_idx+1) * args.batch_size / len(train_loader.dataset), loss.item()))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # now that the epoch is completed calculate statistics and store logs
    avg_loss = mean(epoch_loss)
    print("------------------------\nAverage loss for epoch = {:.2f}".format(avg_loss))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\n------------------------\nAverage loss for epoch = {:.2f}\n".format(avg_loss))

    train_accuracy = 100.0*float(correct)/float(2*len(train_loader.dataset))
    print("Accuracy for epoch = {:.2f}%\n------------------------".format(train_accuracy))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("Accuracy for epoch = {:.2f}%\n------------------------\n".format(train_accuracy))
    
    return netG_A2B, netG_B2A, netD_A, netD_B, netGaze, avg_loss, train_accuracy


# validation function
def val(netG_A2B, netG_B2A, netD_A, netD_B, netGaze):
    global best_accuracy
    netG_B2A.eval()
    netGaze.eval()
    pred_all = np.array([], dtype='int64')
    target_all = np.array([], dtype='int64')
    
    for idx, (data, target) in enumerate(val_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data[:, :args.nc, :, :]), Variable(target)

        # do the forward pass
        data = gaze2gan(data, val_loader.dataset.mean[0:args.nc], val_loader.dataset.std[0:args.nc])
        fake_data = netG_B2A(data)
        fake_data = gan2gaze(fake_data, val_loader.dataset.mean[0:args.nc], val_loader.dataset.std[0:args.nc])
        scores = netGaze(fake_data.repeat(1, int(3 / args.nc), 1, 1))[0]
        scores = scores.view(-1, args.num_classes)
        pred = scores.data.max(1)[1]  # got the indices of the maximum, match them
        print('Done with image {} out of {}...'.format(min(args.batch_size*(idx+1), len(val_loader.dataset)), len(val_loader.dataset)))
        pred_all   = np.append(pred_all, pred.cpu().numpy())
        target_all = np.append(target_all, target.cpu().numpy())

    val_accuracy =  plot_confusion_matrix(target_all, pred_all, merged_activity_classes)
    print("\n------------------------")
    print("Validation accuracy = {:.2f}%\n------------------------".format(val_accuracy))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\n------------------------\n")
        f.write("Validation accuracy = {:.2f}%\n------------------------\n".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    if val_accuracy > best_accuracy:
        # save the model
        torch.save(netG_A2B.state_dict(), os.path.join(args.output_dir, 'netG_A2B.pth'))
        torch.save(netG_B2A.state_dict(), os.path.join(args.output_dir, 'netG_B2A.pth'))
        torch.save(netD_A.state_dict(), os.path.join(args.output_dir, 'netD_A.pth'))
        torch.save(netD_B.state_dict(), os.path.join(args.output_dir, 'netD_B.pth'))
        torch.save(netGaze.state_dict(), os.path.join(args.output_dir, 'netGaze.pth'))
        best_accuracy = plot_confusion_matrix(target_all, pred_all, merged_activity_classes, args.output_dir)

    return val_accuracy


if __name__ == '__main__':
    # networks
    netG_A2B = Generator(args.nc, args.nc)
    netG_B2A = Generator(args.nc, args.nc)
    netD_A = Discriminator(args.nc)
    netD_B = Discriminator(args.nc)
    netGaze = SqueezeNet(args.version)
    
    if args.snapshot_dir is not None:
        if os.path.exists(os.path.join(args.snapshot_dir, 'netG_A2B.pth')):
            netG_A2B.load_state_dict(torch.load(os.path.join(args.snapshot_dir, 'netG_A2B.pth')), strict=False)
        if os.path.exists(os.path.join(args.snapshot_dir, 'netG_B2A.pth')):
            netG_B2A.load_state_dict(torch.load(os.path.join(args.snapshot_dir, 'netG_B2A.pth')), strict=False)
        if os.path.exists(os.path.join(args.snapshot_dir, 'netD_A.pth')):
            netD_A.load_state_dict(torch.load(os.path.join(args.snapshot_dir, 'netD_A.pth')), strict=False)
        if os.path.exists(os.path.join(args.snapshot_dir, 'netD_B.pth')):
            netD_B.load_state_dict(torch.load(os.path.join(args.snapshot_dir, 'netD_B.pth')), strict=False)
        if os.path.exists(os.path.join(args.snapshot_dir, 'netGaze.pth')):
            netGaze.load_state_dict(torch.load(os.path.join(args.snapshot_dir, 'netGaze.pth')), strict=False)

    if args.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()
        netGaze.cuda()

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_gaze = torch.nn.MSELoss()

    # Optimizers & LR schedulers
    optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), 
        lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_gaze = optim.SGD(netGaze.parameters(), lr=0.0002 if args.train_gaze else 0.0, 
        momentum=0.9 if args.train_gaze else 0.0, weight_decay=0.0002 if args.train_gaze else 0.0)

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.epochs, 0, args.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
    input_A = Tensor(args.batch_size, args.nc, args.size, args.size)
    input_B = Tensor(args.batch_size, args.nc, args.size, args.size)
    target_real = Variable(Tensor(args.batch_size, 1, 1, 1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(args.batch_size, 1, 1, 1).fill_(0.0), requires_grad=False)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Visdom logger
    logger = Logger(args.epochs, len(train_loader), mean=train_loader.dataset.mean, std=train_loader.dataset.std)

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
        netG_A2B, netG_B2A, netD_A, netD_B, netGaze, avg_loss, acc = \
        train(netG_A2B, netG_B2A, netD_A, netD_B, netGaze, i)
        # plot the loss
        train_loss.append(avg_loss)
        ax1.plot(train_loss, 'k')
        fig1.savefig(os.path.join(args.output_dir, "train_loss.jpg"))

        # plot the train and val accuracies
        train_acc.append(acc)
        val_acc.append(val(netG_A2B, netG_B2A, netD_A, netD_B, netGaze))
        ax2.plot(train_acc, 'g', label='Train accuracy')
        ax2.plot(val_acc, 'b', label='Validation accuracy')
        fig2.savefig(os.path.join(args.output_dir, 'trainval_accuracy.jpg'))
    plt.close('all')
