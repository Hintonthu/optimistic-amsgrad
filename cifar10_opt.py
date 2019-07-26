'''Train CIFAR10 with PyTorch.'''
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

import Optadam_torch

from models_new import resnet
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
#--------------------------------#
parser.add_argument('--logfile', default='Foo', type=str, help='filename of log file')
parser.add_argument('--span', default=5, type=int, help='number of previous gradients used for prediction')                    
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')




args = parser.parse_args()
print (args.logfile)
betas = (args.beta1, args.beta2)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
net = resnet.ResNet18(num_classes = 10)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
####################
#net_aux = ResNet18()         
net_aux = resnet.ResNet18(num_classes = 10)
net_aux = net_aux.to(device) 
####################
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    net_aux = torch.nn.DataParallel(net_aux)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad = True, weight_decay = args.wd, betas = betas)    
optimizer     = Optadam_torch.Optadam_torch(net.parameters(), lr=args.lr, span = args.span, weight_decay = args.wd, betas = betas)    
#optimizer     = Optadam.Optadam(net.parameters(), lr=args.lr, span = args.span, weight_decay = args.wd, betas = betas)    
optimizer_aux = optim.SGD(net_aux.parameters(), lr=args.lr)

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
        #########################
        net_aux.to(device)
        optimizer_aux.zero_grad()
        output_aux = net_aux(inputs)
        loss_aux   = F.nll_loss(output_aux, targets) 
        loss_aux.backward()

        #########################
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step(optimizer_aux)

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
        torch.save(state, './checkpoint/ckpt2.pth')
        best_acc = acc


trloss_rec = []
tracc_rec  = []
time_rec   = []

tsloss_rec = []
tsacc_rec  = []

t0 = time.time()

for epoch in range( args.epochs):
    train(epoch, trloss_rec, tracc_rec, time_rec, t0)
    test(epoch, tsloss_rec, tsacc_rec)

sio.savemat(args.logfile, {'train_loss': trloss_rec,'train_acc':tracc_rec,'time_rec':time_rec,'test_loss':tsloss_rec,'test_acc':tsacc_rec})
