'''Train MNIST NOISE with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import scipy.io as sio
import time
import os
import argparse

from models import *
from utils import progress_bar
from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


parser = argparse.ArgumentParser(description='PyTorch MNIST NOISE Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
#--------------------------------#
parser.add_argument('--logfile', default='Foo', type=str, help='filename of log file')
parser.add_argument('--span', default=5, type=int, help='number of previous gradients used for prediction')                    
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')


args = parser.parse_args()
print (args.logfile)
use_cuda = torch.cuda.is_available()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

##########  convert .mat to .torch data ##################################
mat = sio.loadmat('./M_datasets/M_rand/train.mat')
x = mat['X']
x = torch.from_numpy(x)
x = torch.reshape(x, [-1, 1, 28, 28]).float()
y = mat['Label']
y = torch.from_numpy(y)
y = torch.transpose(y,0,1)
y = torch.reshape(y,[-1]).long()

trainset = data.TensorDataset(x,y)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


mat = sio.loadmat('./M_datasets/M_rand/test.mat')
x = mat['X']
x = torch.from_numpy(x)
x = torch.reshape(x, [-1, 1, 28, 28]).float()
y = mat['Label']
y = torch.from_numpy(y)
y = torch.transpose(y,0,1)
y = torch.reshape(y,[-1]).long()

testset = data.TensorDataset(x,y)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
##########  convert .mat to .torch data ##################################
criterion = nn.CrossEntropyLoss()
net = Net().to(device)
betas = (args.beta1, args.beta2)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

#optimizer = Optadam2.Optadam2(model.parameters(), lr=args.lr, span = args.span, weight_decay = args.wd, betas = betas)    
#optimizer = Optadam.Optadam(model.parameters(), lr=args.lr, span = args.span, weight_decay = args.wd, betas = betas)    
optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad = True, weight_decay = args.wd, betas = betas)    
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Training
def train(epoch, trloss_rec, tracc_rec, time_rec, t0):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    trloss_rec_aux = []
    tracc_rec_aux  = []
    time_rec_aux   = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if batch_idx % args.log_interval == 0:
            trloss_rec_aux.append( train_loss/(batch_idx+1) )
            tracc_rec_aux.append( 100.*correct/total )
            time_rec_aux.append( time.time()-t0 )

    trloss_rec.append( trloss_rec_aux )
    tracc_rec.append( tracc_rec_aux )
    time_rec.append( time_rec_aux )

def test(epoch, tsloss_rec, tsacc_rec):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num = 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            num += 1

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    tsloss_rec.append( test_loss/(num+1) )
    tsacc_rec.append( acc )

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


trloss_rec = []
tracc_rec  = []
time_rec   = []

tsloss_rec = []
tsacc_rec  = []

t0 = time.time()

for epoch in range( args.epochs):
#    scheduler.step() #
#    print ('\nEpoch: %d' % epoch, ' Learning rate:', scheduler.get_lr())#       
    train(epoch, trloss_rec, tracc_rec, time_rec, t0)
    test(epoch, tsloss_rec, tsacc_rec)

sio.savemat(args.logfile, {'train_loss': trloss_rec,'train_acc':tracc_rec,'time_rec':time_rec,'test_loss':tsloss_rec,'test_acc':tsacc_rec})
