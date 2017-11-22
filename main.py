import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import models
from models import init_weights

# command line arguements for different training configurations
parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outDir', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--model', type=int, default=1, help='1 for dcgan, 2 for illustrationGAN-like-GAN')
parser.add_argument('--d_labelSmooth', type=float, default=0, help='for D, use soft label "1-labelSmooth" for real samples')
parser.add_argument('--n_extra_layers_d', type=int, default=0, help='number of extra conv layers in D')
parser.add_argument('--n_extra_layers_g', type=int, default=1, help='number of extra conv layers in G')
parser.add_argument('--binary', action='store_true', help='z from bernoulli distribution, with prob=0.5')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outDir)
except OSError:
    pass

opt.manualSeed = random.randint(1, 10_000) # fixed seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("You have a CUDA device. I'd recommend running this with --cuda")

# training options gathered from terminal
nc = 3
ngpu = opt.ngpu
nz = opt.nz
ngf = opt.ngf
ndf = opt.ndf
n_extra_d = opt.n_extra_layers_d
n_extra_g = opt.n_extra_layers_g

dataset = datasets.ImageFolder(
    root=opt.dataRoot,
    transform=transforms.Compose([
        transforms.Scale(opt.imageSize),
        transforms.ToTensor(), # converts image to multidimensional array
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # bring images to (-1, 1)
    ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                        shuffle=True, num_workers=opt.workers)

# load our models
if opt.model == 1:
    netG = models.NetG1(ngpu, nz, nc, ngf, n_extra_g)
    netD = models.NetD1(ngpu, nz, nc, ndf, n_extra_d)
elif opt.model == 2:
    netG = models.NetG2(ngpu, nz, nc, ngf)
    netD = models.NetD2(ngpu, nz, nc, ndf)

netG.apply(init_weights)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD.apply(init_weights)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()
criterion_MSE = nn.MSELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
if opt.binary:
    bernoulli_prob = torch.FloatTensor(opt.batchSize, nz, 1, 1).fill_(0.5)
    fixed_noise = torch.bernoulli(bernoulli_prob)
else:
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

# move all params and buffers to GPUs if enabled
if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterion_MSE.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# wrap tensors
input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# setup optimizers
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

# let's start training!
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        start_iter = time.time()
        # Update Discriminator Network: maximize log(D(x)) + log(1 - D(G(z)))
        
        # train with real data
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label - opt.d_labelSmooth) # smooth label for discriminator

        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake data
        noise.data.resize_(batch_size, nz, 1, 1)
        if opt.binary:
            bernoulli_prob.resize_(noise.data.resize())
            noise.data.copy_(2*(torch.bernoulli(bernoulli_prob)-0.5))
        else:
            noise.data.normal_(0, 1)
        fake = netG(noise)
        label.data.fill_(fake_label)
        output = netD(fake.detach()) # avoid backpropogation through generator
        errD_fake = criterion(output, label)
        errD_fake.backward() # accumulate gradients for real and fake data
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update Discriminator Network: maximize log(D(x)) + log(1 - D(G(z)))
        netG.zero_grad()
        label.data.fill_(real_label) # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward(retain_graph=True) # True if backward through graph a second time
        if opt.model == 2: # using z predictor
            errG_z = criterion_MSE(z_prediction, noise)
            errG_z.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        end_iter = time.time()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, end_iter-start_iter))
        if i % 100 == 0:
            # save the first 64 samples in an 8x8 grid
            vutils.save_image(real_cpu[0:64,:,:,:],
                    '%s/real_samples.png' % opt.outDir, nrow=8)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data[0:64,:,:,:],
                    '%s/fake_samples_epoch_%03d.png' % (opt.outDir, epoch), nrow=8)
    if epoch % 1 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outDir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outDir, epoch))


    