import sys
import argparse
import os
import random
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image

from cycleGAN_model import Generator
from datasets2 import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type = int, default = 1, help = 'size of the batches')
parser.add_argument('--dataroot2', type = str, default = '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer')
parser.add_argument('--dataroot', type = str, default = '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer')
parser.add_argument('--input_nc', type = int, default = 3, help = 'number of channels of input data')
parser.add_argument('--output_nc', type = int, default = 3, help = 'number of channels of output data')
parser.add_argument('--size', type = int, default = 256, help = 'size of the data(squared assumed')
parser.add_argument('--cuda', default = True, action = 'store_true', help = 'use GPU computation')
parser.add_argument('--n_cpu', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
parser.add_argument('--generator_h2cat', type = str, default = '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/network_cat/cat_netG_A2B.pth')
parser.add_argument('--generator_h2dog', type = str, default = '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/network_dog/dog_netG_A2B.pth')
parser.add_argument('--generator_h2pig', type = str, default = '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/network_pig/pig_netG_A2B.pth')
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

# Networks  ============================================================================================================

cat_netG_B2A = Generator(options.output_nc, options.input_nc)
dog_netG_B2A = Generator(options.output_nc, options.input_nc)
pig_netG_B2A = Generator(options.output_nc, options.input_nc)

if options.cuda:
    cat_netG_B2A.cuda()
    dog_netG_B2A.cuda()
    pig_netG_B2A.cuda()

# Load state dicts
cat_netG_B2A.load_state_dict(torch.load(options.generator_h2cat))
dog_netG_B2A.load_state_dict(torch.load(options.generator_h2dog))
pig_netG_B2A.load_state_dict(torch.load(options.generator_h2pig))

# Set model's test mode
cat_netG_B2A.eval()
dog_netG_B2A.eval()
pig_netG_B2A.eval()

# Inputs and Targets memory allocation
Tensor = torch.cuda.FloatTensor if options.cuda else torch.Tensor
input_A = Tensor(options.batchSize, options.input_nc, options.size, options.size)
input_B = Tensor(options.batchSize, options.input_nc, options.size, options.size)
# DatasetLoader
transforms_ = transforms.Compose([
    transforms.Resize((int(options.size), int(options.size)), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

dataloader = DataLoader(ImageDataset(options.dataroot, options.dataroot, transforms = transforms_, mode = 'test'),
                        batch_size=options.batchSize, shuffle=False, num_workers=options.n_cpu)


########################################################################################################################
# Test

# Create save folder(Transfer)
if not os.path.exists('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/cat'):
    os.makedirs('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/cat')
if not os.path.exists('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/dog'):
    os.makedirs('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/dog')
if not os.path.exists('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/pig'):
    os.makedirs('/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/pig')


for i, batch in enumerate(dataloader):
    # set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))
    real_A = real_A.cuda()
    real_B = real_B.cuda()

    # Generate output
    fake_cat = 0.5*(cat_netG_B2A(real_B).data+1.0)
    fake_dog = 0.5*(dog_netG_B2A(real_B).data+1.0)
    fake_pig = 0.5*(pig_netG_B2A(real_B).data+1.0)

    save_image(fake_cat, '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/cat/human_cat_%04d.jpg'%(i+1))
    save_image(fake_dog, '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/dog/human_dog_%04d.jpg'%(i+1))
    save_image(fake_pig, '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/pig/human_pig_%04d.jpg'%(i+1))
    save_image(real_B, '/home/cheeze/PycharmProjects/KJW/capstone_project/human2animal/transfer_network/image_transfer/test_transfer/human/human_%04d.jpg'%(i+1))
    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
