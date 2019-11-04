import argparse
import sys
import os
import random
import torch.backends.cudnn as cudnn
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from cycleGAN_model import Generator
from datasets2 import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot2', type=str, default='/home/cheeze/PycharmProjects/KJW/1. Dataset/Dataset_animal')
parser.add_argument('--dataroot', type=str, default='/home/cheeze/PycharmProjects/KJW/1. Dataset/Dataset_animal')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data(squared assumed')
parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/2. TestParameter/Bear_netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/2. TestParameter/Bear_netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--seed', type=int, help='manual seed')
options=parser.parse_args()
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


######## Definition of Variables ##########
# Newtworks

netG_A2B = Generator(options.input_nc, options.output_nc)
netG_B2A = Generator(options.output_nc, options.input_nc)

if options.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(options.generator_A2B))
netG_B2A.load_state_dict(torch.load(options.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs and Targets memory allocation
Tensor = torch.cuda.FloatTensor if options.cuda else torch.Tensor
input_A = Tensor(options.batchSize, options.input_nc, options.size, options.size)
input_B = Tensor(options.batchSize, options.output_nc, options.size, options.size)

# Dataset Loader
transforms_ = transforms.Compose([
    transforms.Resize((int(options.size), int(options.size)), Image.BICUBIC),
    #transforms.RandomCrop(options.size),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5,), (0.5, 0.5, 0.5))
])

dataloader = DataLoader(ImageDataset(options.dataroot, options.dataroot2, transforms=transforms_, mode='test'),
                        batch_size=options.batchSize, shuffle=False, num_workers=options.n_cpu)

########################################################################################################################

######## Testing ###########
# Create output dirs if they don't exist
if not os.path.exists('/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/3. TestAnimal'):
    os.makedirs('/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/3. TestAnimal')
if not os.path.exists('/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/3. TestHuman'):
    os.makedirs('/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/3. TestHuman')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))
    real_A = real_A.cuda()
    real_B = real_B.cuda()

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(fake_A, '/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/3. TestAnimal/TYPE_A_%04d.jpg'%(i+1))
    save_image(fake_B, '/home/cheeze/PycharmProjects/KJW/2. Codes/1. capstone/4. Result/4. TestHuman/TYPE_B_%04d.jpg'%(i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

sys.stdout.write('\n')
###################################