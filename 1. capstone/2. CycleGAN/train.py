import argparse
import itertools
import os
import random
import utils
import visualize_tools

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from cycleGAN_model import Generator
from cycleGAN_model import Discriminator
from utils_cyclegan import ReplayBuffer
from utils_cyclegan import LambdaLR

from utils_cyclegan import Logger
from utils_cyclegan import weights_init_normal
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type = int, default=0, help = 'starting epoch')
parser.add_argument('--n_epochs', type = int, default=2000, help = 'number of epochs of training')
parser.add_argument('--batchSize', type = int, default = 1, help='size of the batches')
parser.add_argument('--dataroot2', type = str, default = '/home/cheeze/PycharmProjects/KJW/1. Dataset/Dataset_animal')
parser.add_argument('--dataroot', type = str, default = '/home/cheeze/PycharmProjects/KJW/1. Dataset/Dataset_face')
parser.add_argument('--lr', type = float, default = 0.0002, help = 'initial learning rate')
parser.add_argument('--decay_epoch', type = int, default = 100, help = 'epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type = int, default = 256, help = 'size of the data crop(squared assumed)')
parser.add_argument('--input_nc', type = int, default = 3, help = 'number of channels of input data')
parser.add_argument('--output_nc', type = int, default = 3, help = 'number of channels of output data')
parser.add_argument('--cuda', default = True, action = 'store_true', help = 'use GPU computation')
parser.add_argument('--n_cpu', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
parser.add_argument('--seed', type = int, help = 'manual seed')
options = parser.parse_args()

print(options)


# seed set  ============================================================================================================
if options.seed is None:
    options.seed = random.randint(1, 10000)
print("Random Seed: ", options.seed)
random.seed(options.seed)
torch.manual_seed(options.seed)

# cuda set  ============================================================================================================
if options.cuda:
    torch.cuda.manual_seed(options.seed)

torch.backends.cudnn.benchmark = True
cudnn.benchmark = True
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


#======================================================================================================================#
## Definition of variables ##

# Networks
netG_A2B = Generator(options.input_nc, options.output_nc)
netG_B2A = Generator(options.output_nc, options.input_nc)
netD_A = Discriminator(options.input_nc)
netD_B = Discriminator(options.output_nc)


netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Loss function
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR Schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=options.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=options.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=options.lr, betas=(0.5, 0.999))

### learning rate schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(options.n_epochs,options.epoch, options.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(options.n_epochs, options.epoch, options.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(options.n_epochs, options.epoch, options.decay_epoch).step)


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if options.cuda else torch.Tensor
input_A = Tensor(options.batchSize, options.input_nc, options.size, options.size)
input_B = Tensor(options.batchSize, options.output_nc, options.size, options.size)
target_real = Variable(Tensor(options.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(options.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Cuda options
if options.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()
    input_A.cuda()
    input_B.cuda()
    target_real.cuda()
    target_fake.cuda()

# Dataset loader
transforms = transforms.Compose([
    transforms.Resize(int(options.size*1.12), Image.BICUBIC),
    transforms.RandomCrop(options.size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5,), (0.5, 0.5, 0.5))
])

dataloader = DataLoader(ImageDataset(options.dataroot, options.dataroot2, transforms=transforms, unaligned=True),
                        batch_size=options.batchSize, shuffle=True, num_workers=options.n_cpu)

# Loss plot
logger = Logger(options.n_epochs, len(dataloader))
win_dict = visualize_tools.win_dict()
line_win_dict = visualize_tools.win_dict()
line_win_dict_val = visualize_tools.win_dict()
###################################

###### Training ######
for epoch in range(options.epoch, options.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        real_A = real_A.cuda()
        real_B = real_B.cuda()

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        same_B = same_B.cuda()
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        same_A = same_A.cuda()
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        #loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        loss_GAN_A2B = criterion_GAN(target_real, pred_fake)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        #loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        loss_GAN_B2A = criterion_GAN(target_real, pred_fake)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(target_real, pred_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(target_fake, pred_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(target_real, pred_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(target_fake, pred_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                   images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        line_win_dict = visualize_tools.draw_lines_to_windict(line_win_dict, [loss_G.cpu().detach().numpy(), (loss_D_A + loss_D_B).cpu().detach().numpy(), (loss_identity_A + loss_identity_B).cpu().detach().numpy()],
                                                              ['Loss_G', 'Loss_D', 'Loss_G_identity'],
                                                              epoch, i, len(dataloader))

        # Save model checkpoints
        torch.save(netG_A2B.state_dict(),
                   '/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/1. BearNetwork/Bear_netG_A2B.pth')
        torch.save(netG_B2A.state_dict(),
                   '/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/1. BearNetwork/Bear_netG_B2A.pth')
        torch.save(netD_A.state_dict(),
                   '/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/1. BearNetwork/Bear_netD_A.pth')
        torch.save(netD_B.state_dict(),
                   '/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/1. BearNetwork/Bear_netD_B.pth')



    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

