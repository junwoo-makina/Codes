import argparse
import os
import random
import glob
from PIL import ImageOps
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import GAN_model as model
# import custom package
import utils
import visualize_tools

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='CelebA', help='what is dataset?')
parser.add_argument('--dataroot', default='/home/cheeze/PycharmProjects/KJW/research/dcgan+autoencoder/img', help='path to dataset')
parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--outf', default='./output', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=True, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--imageSize', type=int, default=80, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nz', type=int, default=100, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')

parser.add_argument('--seed', type=int, help='manual seed')

options = parser.parse_args()
print(options)

sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


class DL(torch.utils.data.Dataset):
    def __init__(self, path, transform, type):
        random.seed = 1
        self.transform = transform
        self.type = type
        assert os.path.exists(path)
        self.base_path = path

        total_file_paths = []
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR','*'))
        #total_file_paths =  total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'FERET', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'FERET_normal', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR_normal', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'CMU_PIE', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'CMU_PIE_normal', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'Yale', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'YaleB', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'Yale_normal', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'YaleB_normal', '*'))
        total_file_paths = total_file_paths + cur_file_paths

        random.shuffle(total_file_paths)
        num_of_valset = int(len(total_file_paths)/2)

        self.train_file_paths = total_file_paths[:num_of_valset]
        self.test_file_paths = total_file_paths[num_of_valset:]

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = ImageOps.equalize(img)
                return img.resize((80,80))

    def __len__(self):
        if self.type == 'train':
            return len(self.train_file_paths)
        if self.type == 'test':
            return len(self.test_file_paths)

    def __getitem__(self, item):
        if self.type == 'train':
            path = self.train_file_paths[item]
        elif self.type == 'test':
            path = self.test_file_paths[item]

        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img



# save directory make   ================================================================================================
try:
    os.makedirs(options.outf)
except OSError:
    pass

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


#=======================================================================================================================
# Data and Parameters
#=======================================================================================================================

# Dataset call and load   ================================================================================================
transform = transforms.Compose([
    transforms.Scale(80),
    transforms.ToTensor(),
    transforms.Normalize(mean =(0.5, 0.5, 0.5),
                         std =(0.5, 0.5, 0.5))
])

dataloader = torch.utils.data.DataLoader(
    DL(options.dataroot, transform, 'train'),
    batch_size=options.batchSize, shuffle=True, num_workers= 0)
# normalize to -1~1
ngpu = int(options.ngpu)
nz = int(options.nz)
ngf = int(options.ngf)
ndf = int(options.ndf)


#=======================================================================================================================
# Models
#=======================================================================================================================

# Generator ============================================================================================================
netG = model._netG(ngpu, in_channels=nz)
netG.apply(utils.weights_init)
if options.netG != '':
    netG.load_state_dict(torch.load(options.netG))
print(netG)

# Discriminator ========================================================================================================
netD = model._netD(ngpu)
netD.apply(utils.weights_init)
if options.netD != '':
    netD.load_state_dict(torch.load(options.netD))
print(netD)

#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
criterion_D = nn.BCELoss()
criterion_G = nn.BCELoss()


# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
#optimizerD = optim.Adam(netD.parameters(), betas=(0.5, 0.999), lr=2e-4)
#optimizerG = optim.Adam(netG.parameters(), betas=(0.5, 0.999), lr=1e-3)

optimizerD = optim.SGD(netD.parameters(), lr=0.0001, momentum=0.9)
optimizerG = optim.SGD(netG.parameters(), lr=0.0001, momentum=0.9)


# container generate
input = torch.FloatTensor(options.batchSize, 1, options.imageSize, options.imageSize)
noise = torch.FloatTensor(options.batchSize, nz, 1, 1)

label = torch.FloatTensor(options.batchSize)


if options.cuda:
    netD.cuda()
    netG.cuda()
    criterion_D.cuda()
    criterion_G.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()


# make to variables ====================================================================================================
input = Variable(input)
label = Variable(label,requires_grad = False)
noise = Variable(noise)

win_dict = visualize_tools.win_dict()
line_win_dict =visualize_tools.win_dict()
# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ###########################
        # train with real data  ========================================================================================

        optimizerD.zero_grad()
        #batch_size = data.size(0)
        #data = data.reshape(batch_size,-1)
        #labels = torch.ones(batch_size, 1)
        #fake_labels = torch.zeros(batch_size, 1)
        real_cpu = data
        batch_size = real_cpu.size(0)

        input.data.resize_(real_cpu.size()).copy_(real_cpu)

        label.data.resize_(batch_size,1,1,1).fill_(1)
        real_label = label.clone()
        label.data.resize_(batch_size,1,1,1).fill_(0)
        fake_label = label.clone()

        outputD = netD(input)
        errD_real = criterion_D(outputD, real_label)
        #errD_real.backward()
        D_x = outputD.data.mean()   # for visualize

        # generate noise    ============================================================================================
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)

        #train with fake data   ========================================================================================
        fake = netG(noise)
        outputD = netD(fake.detach())
        errD_fake = criterion_D(outputD, fake_label)
        #errD_fake.backward()
        D_G_z1 = outputD.data.mean()

        errD = errD_fake+errD_real

        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network and Q network
        ###########################
        if(i%2==0):
            optimizerG.zero_grad()

            #fake = Variable(fake.data, requires_grad = True)
            outputD = netD(fake)
            errG = criterion_G(outputD, real_label)
            errG.backward()
            D_G_z2 = outputD.data.mean()
            optimizerG.step()

        #visualize
        print('[%d/%d][%d/%d] Loss_D: %.4f(1:%.4f 2:%.4f) Loss_G: %.4f     D(x): %.4f D(G(z)): %.4f | %.4f'
              % (epoch, options.iteration, i, len(dataloader),
                 errD.data[0],errD_real.mean(),errD_fake.mean(), errG.data[0],  D_x, D_G_z1, D_G_z2))
        if True:
            testImage = fake.data[0]
            win_dict = visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Face GAN"])
            line_win_dict = visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [errD.data.mean(), errG.data.mean(), D_x,
                                                                       D_G_z1],
                                                                      ['lossD', 'lossG', 'real is?', 'fake is?'], epoch, i,
                                                                      len(dataloader))
        if i%100 == 0:
            testImage = fake[0].view(-1, 1, 80, 80)
            save_image(testImage, os.path.join(sample_dir, 'testGenerate{}.jpg'.format(epoch+1)))
        if errG.data.mean() > 2.2:
            break


    # do checkpointing
    if epoch % 10 == 0 :
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (options.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (options.outf, epoch))



# Je Yeol. Lee \[T]/
# Jun Woo, Kim (o,.o)7