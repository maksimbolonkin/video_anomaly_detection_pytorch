import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import shutil

from ucsd_dataset import UCSDAnomalyDataset
from video_CAE import VideoAutoencoderLSTM

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# training parameters
num_epochs = 15
start_epoch = 1
resume = True
resume_file = ''

batch_size = 50
print_after_iter = 100
save_every_epochs = 1
snapshot_dir = './snapshots/'
init_lr = 0.01
lr_change_step = 5
lr_gamma = 0.1
weight_decay = 0.001

timestamp = str(datetime.datetime.now())
log_fn = 'training-%s.log'%timestamp
#snapshot_pref = 'snapshot_epoch'


# check cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
	cudnn.benchmark = True
	print('Using cuda')

# create dataset
train_loader = torch.utils.data.DataLoader(
	UCSDAnomalyDataset('./data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train', time_stride=3),
	batch_size=batch_size, shuffle=True, num_workers=4)

print('Dataloader created')

#create a model
model = VideoAutoencoderLSTM()
if use_cuda:
	model.cuda()

print("Model created")
flog = open(log_fn, 'w')

#create criterion
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)

if resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

# training/testing procedures
def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if use_cuda:
			data, target = data.cuda(), target.cuda().float()

		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % print_after_iter == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))
			flog.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))
			flog.flush()

def test():
	model.eval()
	test_loss = 0
	for data, target in test_loader:
		if use_cuda:
			data, target = data.cuda(), target.cuda().float()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)

        # sum up batch loss
		test_loss += criterion(output, target).data[0]

	test_loss /= len(test_loader)
	print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
	flog.write('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
	flog.flush()

def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')

def scheduler_step(epoch):
	if epoch % lr_change_step == 0:
		lr = init_lr * (lr_gamma ** (epoch // lr_change_step))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr


print("Start training")
# save learning hyperparameters to log file
flog.write("init lr: %f, wd: %f\n"%(init_lr, weight_decay))

for epoch in range(start_epoch, num_epochs+1):
	scheduler_step(epoch)
	train(epoch)
	test()
	if epoch % save_every_epochs == 0:
		print("Saving snapshot...")
		chk_fn = os.path.join(snapshot_dir, 'snapshot_%s_epoch%d.pth.tar'%(timestamp, epoch))
		save_checkpoint({
			'epoch': epoch,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
		}, is_best = False, filename = chk_fn)

flog.close()