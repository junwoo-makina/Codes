import torch.utils.data as ud
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from PIL import Image
from PIL import ImageOps

import utils
import visualize_tools
import numpy as np
import os
import glob
import torch.utils.data


# ======================================================================================================================
# ____________________________________________________Options___________________________________________________________
# ======================================================================================================================


parser = argparse.ArgumentParser()

# Options for path
parser.add_argument('--dataroot', default = '/home/cheeze/PycharmProjects/KJW/research/face_autoencoder/input', help = 'path to dataset')
parser.add_argument('--outf', default = './pretrain_model', help = 'folder to output images and model checkpoints')
parser.add_argument('--cuda', default = 'True', action = 'store_true', help = 'enables cuda')
parser.add_argument('--display', default = False, help = 'display options. default:False, NOT IMPLEMENTED')
parser.add_argument('--ngpu', type = int, default = 1, help = 'number of GPUs to use')
parser.add_argument('--workers', type = int, default = 1, help = 'number of data loading workers')
parser.add_argument('--iteration', type = int, default = 10000, help = 'number of epochs to train for')

# Options for saving and testing
parser.add_argument('--batchSize', type = int, default = 10, help = 'input batch size')
parser.add_argument('--imageSize', type = int, default = 120, help = 'the height / width of the input image to network')
parser.add_argument('--model', type = str, default = 'pretrained_AF', help = 'Model name')
parser.add_argument('--nc', type = int, default = 3, help = 'number of input channel')
parser.add_argument('--nz', type = int, default = 2, help = 'number of input channel(z)')
parser.add_argument('--ngf', type = int, default = 64, help = 'number of generator filters')
parser.add_argument('--ndf', type = int, default = 64, help = 'number of discriminator filters')
parser.add_argument('--lr', type = float, default = 0.0002, help = 'learning rate')
parser.add_argument('--beta1', type = float, default = 0.5, help = 'beta1 for Adam')

parser.add_argument('--seed', type = int, help = 'manual seed')

options = parser.parse_args()
print(options)


# Set the saving folder
sample_dir = 'samples'
path = '/home/cheeze/PycharmProjects/KJW/research/face_autoencoder'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# How to calculate Variational Inference?
def Variational_loss(input, target, mu, logvar, epoch):
    recon_loss = MSE_loss(input, target)
    KLD_loss = -0.5*torch.sum(1+logvar-mu.pow(2) - logvar.exp())
    result = recon_loss + KLD_loss
    return result

class AR_DL(torch.utils.data.Dataset):
    def __init__(self, path, transform, type):
        self.transform = transform
        self.type = type
        assert os.path.exists(path)
        self.base_path = path

        total_file_paths = []

        cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR_session1', '*'))
        total_file_paths = total_file_paths + cur_file_paths

        total_file_paths = sorted(total_file_paths)
        self.file_paths = sorted(total_file_paths[:])


    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = ImageOps.equalize(img)
                return img.resize((120, 100))

    def __len__(self):
        if self.type == 'AR':
            return len(self.file_paths)

    def __getitem__(self, item):
        #if (item%10==9) or (item%10==8) or (item%10)==7 or (item%10)==6 or (item%10)==5 or (item%10)==4 or (item%10)==3:
        #    item = item-(item%10)
        if self.type == 'AR':
            path = self.file_paths[item]

        img = self.pil_loader(path)

        if self.transform is not None:
            img = self.transform(img)
            if path[65:67] =='AR':
                sample_size = 7
            elif path[65:67] =='Fe':
                sample_size = 2
        return img, sample_size


class Feret_DL(torch.utils.data.Dataset):
    def __init__(self, path, transform, type):
        self.transform = transform
        self.type = type
        assert os.path.exists(path)
        self.base_path = path

        total_file_paths = []

        cur_file_paths = glob.glob(os.path.join(self.base_path, 'Feret_train', '*'))
        total_file_paths = total_file_paths + cur_file_paths

        total_file_paths = sorted(total_file_paths)
        self.file_paths = sorted(total_file_paths[:])


    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = ImageOps.equalize(img)
                return img.resize((120, 100))

    def __len__(self):
        if self.type == 'Feret':
            return len(self.file_paths)

    def __getitem__(self, item):
        if self.type == 'Feret':
            path = self.file_paths[item]

        img = self.pil_loader(path)

        if self.transform is not None:
            img = self.transform(img)
            if path[65:67] =='AR':
                sample_size = 7
            elif path[65:67] =='Fe':
                sample_size = 2
        return img, sample_size




# ======================================================================================================================
# ________________________________________________________Model_________________________________________________________
# ======================================================================================================================
class encoder(nn.Module):
    def __init__(self,  z_size=2, channel=3, num_filters=64, type='AE'):
        super().__init__()
        self.type = type
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, num_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, z_size, 3, 1, 1, bias=False),
        )
        if self.type == 'VAE':
            self.fc_mu = nn.Conv2d(z_size, z_size, 1)
            self.fc_sig = nn.Conv2d(z_size, z_size, 1)
        # init weights
        self.weight_init()

    def forward(self, x):
        if self.type == 'AE':
            #AE
            z = self.encoder(x)
            return z
        elif self.type == 'VAE':
            # VAE
            z_ = self.encoder(x)
            mu = self.fc_mu(z_)
            logvar = self.fc_sig(z_)
            return mu, logvar
    def weight_init(self):
        self.encoder.apply(weight_init)

class decoder(nn.Module):
    def __init__(self, z_size=2,channel=3, num_filters=64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, num_filters * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 1, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(num_filters, channel, 5, 5, 0, bias=False),
            nn.Tanh()
        )
        self.decoder_subpixel = nn.Sequential(


        )
        # init weights
        self.weight_init()

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def weight_init(self):
        self.decoder.apply(weight_init)

# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)




# ======================================================================================================================
# ______________________________________Image pre-processing and cuda settings__________________________________________
# ======================================================================================================================

# Make sunglasses on (120,100) face images
def make_sunglass(img):
    image_ = img

    # make sunglass
    image_[:, :, 26:56, 8:40] = -1  # whole, height, width
    image_[:, :, 26:56, 64:96] = -1
    image_[:, :, 30:40, 40:64] = -1
    return image_

# Make random noises on (120,100) face images
def make_random_squre(image):
    image_ = image
    x = random.randrange(0, 120)
    y = random.randrange(0, 100)
    w = random.randrange(5, 119)
    h = random.randrange(5, 99)
    if y + h < 100 and x + w < 120:
        image_[:, :, x:x + w, y:y + h] = -1
        return image_
    else:
        return make_random_squre(image_)


# Set cuda avaliable

torch.backends.cudnn.benchmark = True
cudnn.benchmark = True
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")





# ======================================================================================================================
# _______________________________________________Data and Parameters____________________________________________________
# ======================================================================================================================
ngpu = int(options.ngpu)
nz = int(options.nz)
autoencoder_type = 'AE'
encoder = encoder(options.nz, options.nc, 64, autoencoder_type)
encoder.apply(utils.weights_init)
# if options.netG != '':
#   encoder.load_state_dict(torch.load(options.netG))
print(encoder)

decoder = decoder(options.nz, options.nc)
decoder.apply(utils.weights_init)
# if options.netD != '':
#   decoder.load_state_dict(torch.load(options.netD))
print(decoder)



# =======================================================================================================================
# ___________________________________________________Training___________________________________________________________
# =======================================================================================================================

# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss(size_average=False)
L1_loss = nn.L1Loss(size_average=False)

# setup optimizer   ====================================================================================================
optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.099), lr=0.0001)
optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.099), lr=0.0001)

#optimizerD = optim.SGD(decoder.parameters(), lr = 0.0001, momentum = 0.9)
#optimizerE = optim.SGD(encoder.parameters(), lr = 0.0001, momentum = 0.9)

# container generate
input = torch.FloatTensor(options.batchSize, 1, 120, 100)
sample_size = torch.FloatTensor(options.batchSize, 1)

if options.cuda:
    encoder.cuda()
    decoder.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()
    L1_loss.cuda()
    input = input.cuda()
    sample_size = sample_size.cuda()

# make to variables
input = Variable(input)
sample_size = Variable(sample_size)


def train():
    # Set image processing (for training)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        #transforms.CenterCrop(55),
        transforms.Scale((120, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    unorm = visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


    dataloader_AR = torch.utils.data.DataLoader(
        AR_DL(options.dataroot, transform, 'AR'),
        batch_size=10, shuffle=False)
    dataloader_Feret = torch.utils.data.DataLoader(
        Feret_DL(options.dataroot, transform, 'Feret'),
        batch_size=2, shuffle=False
    )

    win_dict_AR = visualize_tools.win_dict()
    win_dict_Feret = visualize_tools.win_dict()
    win_dict_AR2 = visualize_tools.win_dict()

    line_win_dict_AR = visualize_tools.win_dict()
    line_win_dict_Feret = visualize_tools.win_dict()
    ep = 0

    for epoch in range(options.iteration):
        train_err = 0

        for i, (data, sample_size) in enumerate(dataloader_AR,0):
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            real_cpu = data
            sample_size = Variable(sample_size).cuda().float()
            original_data = Variable(real_cpu).cuda()
            input.data.resize_(original_data.size()).copy_((original_data))

            instance_xi = original_data[0]
            total_err = 0
            if autoencoder_type == 'AE':
                z = encoder(input)
                x_recon = decoder(z)
            for j in range(0,10):
                err_mse = (MSE_loss(x_recon[j], instance_xi))/10
                total_err = total_err + err_mse
            total_err.backward(retain_graph = True)
            train_err += float(total_err.data.mean())

            optimizerE.step()
            optimizerD.step()

            # Visualization
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, options.iteration, i, len(dataloader_AR), total_err.data.mean()))

            testImage = torch.cat((unorm(input.data[0]), unorm(input.data[1]), unorm(input.data[2]),
                                   unorm(input.data[3]), unorm(input.data[4]), unorm(input.data[5]),
                                   unorm(input.data[6]), unorm(input.data[7])), 2)
            testImage2 = torch.cat((unorm(original_data.data[0]), unorm(x_recon.data[0]), unorm(x_recon.data[1]),
                                    unorm(x_recon.data[2]), unorm(x_recon.data[3]), unorm(x_recon.data[4]),
                                    unorm(x_recon.data[5]), unorm(x_recon.data[6]), unorm(x_recon.data[7])), 2)
            win_dict_AR = visualize_tools.draw_images_to_windict(win_dict_AR, [testImage], ["GAE"])
            win_dict_AR2 = visualize_tools.draw_images_to_windict(win_dict_AR2, [testImage2], ["GAE"])
            line_win_dict_AR = visualize_tools.draw_lines_to_windict(line_win_dict_AR,
                                                                     [total_err.data.mean(), 0],
                                                                     ['loss_recon_AR', 'zero'],
                                                                     epoch, i, len(dataloader_AR))

            i = i+1
            train_err = (train_err/i)

        val_err = 0
        for i, (data, sample_size) in enumerate(dataloader_Feret, 0):
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            real_cpu = data
            sample_size = Variable(sample_size).cuda().float()
            original_data = Variable(real_cpu).cuda()
            input.data.resize_(real_cpu.size()).copy_(make_random_squre(real_cpu))

            instance_xi = original_data[0]
            total_err = 0
            if autoencoder_type =='AE':
                z = encoder(input)
                x_recon = decoder(z)
            for k in range(0,2):
                err_mse = (MSE_loss(x_recon[k], instance_xi))/2
                total_err = total_err + err_mse
            total_err.backward(retain_graph = True)
            optimizerE.step()
            optimizerD.step()
            train_err += float(total_err.data.mean())

            # Visualization
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, options.iteration, i, len(dataloader_Feret), total_err.data.mean()))

            testImage = torch.cat((unorm(original_data.data[0]), unorm(x_recon.data[0]), unorm(input.data[0])),2)
            win_dict_Feret = visualize_tools.draw_images_to_windict(win_dict_Feret, [testImage], ["GAE"])
            line_win_dict_Feret = visualize_tools.draw_lines_to_windict(line_win_dict_Feret,
                                                                        [total_err.data.mean(), 0],
                                                                        ['loss_recon_Feret', 'zero'],
                                                                        epoch, i, len(dataloader_Feret))

        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), '%s/pretrain_AE_encoder_epoch_%d.pth' % (options.outf, ep + epoch))
            torch.save(decoder.state_dict(), '%s/pretrain_AE_decoder_epoch_%d.pth' % (options.outf, ep + epoch))
            output_test = testImage.view(-1, 1, 120, 300)
            save_image(output_test, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

if __name__ == "__main__" :
    train()