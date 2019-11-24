import numpy as np
import argparse
import os
import torch
import torchvision
import torch.nn as nn
import torch.backends as cudnn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import random
import glob
from PIL import Image
import Null_LDA_for_FSDD
import KNN_for_NLDA_Python
from PIL import ImageOps

import torch.utils.data
import visualize_tools
import LJY_utils

from PIL import ImageDraw

'''
# Introduction

    1. Dataloader set
    
    2. Image Pre-Processing
    
    3. Set the Encoder-Decoder Networks
    
    4. Networks Parameter Initialization(Xavier)
    
    5. Define AE Loss function
    
    6. Train AutoEncoder
        6.1 Options
        6.2 Save directory set
        6.3 Seed set
        6.4 Cuda set
        6.5 Data and Parameters
        6.6 Training settings(criterion set, optimizer set, container generate, make to Variables)
    
    7. Test NLDA
'''

sample_dir = 'samples'
path = '/home/cheeze/PycharmProjects/NLDA_code/data'
path2 = 'home/cheeze/PycharmProjects/NLDA_code/data'

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# Set the Dataloader(train, test)

class DL(torch.utils.data.Dataset):
    def __init__(self, path, transform, type):
        random.seed = 1
        self.transform = transform
        self.type = type
        assert os.path.exists(path)
        self.base_path = path

        total_file_paths = []
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR_session1_train', '*'))
        total_file_paths =  total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR_session2', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'Feret_train', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'FERET_normal', '*'))
        #total_file_paths = total_file_paths + cur_file_paths

        random.shuffle(total_file_paths)
        num_of_valset = int(len(total_file_paths)/10)

        self.train_file_paths = total_file_paths[num_of_valset:]
        self.test_file_paths = total_file_paths[:num_of_valset]

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = ImageOps.equalize(img)
                return img.resize((120,100))

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


# Image pre-processing
def default_image(img):
    image_=img
    return image_

def make_sunglass(img):
    image_ = img

    # make sunglass
    image_[:, :, 6:31, 8:34] = -1
    image_[:, :, 6:31, 49:74] = -1
    image_[:, :, 8:15, 34:49] = -1
    return image_

def make_sunglass2(img):
    image_ = img

    # make sunglass
    image_[:, :, 26:56, 8:40] = -1    # whole, height, width
    image_[:, :, 26:56, 64:96] = -1
    image_[:, :, 30:40, 40:64] = -1
    return image_


def make_sunglass3(img):
    image_ = img

    # make sunglass
    image_[:, :, 6:31, 5:29] = -1
    image_[:, :, 6:31, 35:59] = -1
    image_[:, :, 8:15, 29:35] = -1
    return image_

# Set the Encoder-Decoder Networks

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
            nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, z_size, 4, 2, 1, bias=False),
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
            nn.ConvTranspose2d(z_size, num_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 2, 1, bias=False),
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
            nn.ConvTranspose2d(num_filters, channel, 4, 2, 1, bias=False),
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

# Network Parameter Initialization

def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


# Define the Loss-function

def Variational_loss(input, target, mu, logvar, epoch):
    recon_loss = MSE_loss(input, target)
    KLD_loss = -0.5 * torch.sum(1+logvar-mu.pow(2) - logvar.exp())
    result=recon_loss+ KLD_loss
    if(result >35000 and epoch > 350):
        print("%4f %4f", recon_loss, KLD_loss)
    return recon_loss + KLD_loss


def train_ae():
    class encoder(nn.Module):
        def __init__(self, z_size=2, channel=3, num_filters=64, type='AE'):
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
                # AE
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
        def __init__(self, z_size=2, channel=3, num_filters=64):
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

    # Options

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celebA', help='what is dataset?')
    parser.add_argument('--dataroot',
                        default='/home/cheeze/PycharmProjects/NLDA_code/data',
                        help='path to dataset')
    parser.add_argument('--pretrainedModelName', default='autoencoder', help="path of Encoder networks.")
    parser.add_argument('--pretrainedEpoch', type=int, default=0, help="path of Decoder networks.")
    parser.add_argument('--outf', default='./pretrained_model', help="folder to output images and model checkpoints")

    parser.add_argument('--cuda', default='True', action='store_true', help='enables cuda')
    parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--iteration', type=int, default=10000, help='number of epochs to train for')

    # these options are saved for testing
    parser.add_argument('--batchSize', type=int, default=200, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=120, help='the height / width of the input image to network')
    parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
    parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
    parser.add_argument('--nz', type=int, default=500, help='number of input channel.')
    parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
    parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')

    parser.add_argument('--seed', type=int, help='manual seed')

    options = parser.parse_args()
    print(options)


    # Save directory make

    try:
        os.makedirs(options.outf)
    except OSError:
        pass

    # Seed set

    if options.seed is None:
        options.seed = random.randint(1, 10000)
    print("Random Seed: ", options.seed)
    random.seed(options.seed)
    torch.manual_seed(options.seed)

    # Cuda set

    if options.cuda:
        torch.cuda.manual_seed(options.seed)

    torch.backends.cudnn.benchmark = True
    cudnn.benchmark = True
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Data and Parameters

    ngpu = int(options.ngpu)
    nz = int(options.nz)
    autoencoder_type = 'AE'
    encoder = encoder(options.nz, options.nc, 64, autoencoder_type)
    encoder.apply(LJY_utils.weights_init)
    # if options.netG != '':
    #   encoder.load_state_dict(torch.load(options.netG))
    print(encoder)

    decoder = decoder(options.nz, options.nc)
    decoder.apply(LJY_utils.weights_init)
    # if options.netD != '':
    #   decoder.load_state_dict(torch.load(options.netD))
    print(decoder)

    # Training settings(criterion set, optimizer set, container generate, make to Variables)

    BCE_loss = nn.BCELoss()
    MSE_loss = nn.MSELoss(size_average=False)
    L1_loss = nn.L1Loss(size_average=False)

    optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=2e-3)
    optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-3)

    input = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)
    if options.cuda:
        encoder.cuda()
        decoder.cuda()
        MSE_loss.cuda()
        BCE_loss.cuda()
        L1_loss.cuda()
        input = input.cuda()

    input = Variable(input)



    ## Start Training

    transform = transforms.Compose([
        transforms.Scale((120,100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    unorm = visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    dataloader = torch.utils.data.DataLoader(
        DL(options.dataroot, transform, 'train'),
        batch_size=options.batchSize, shuffle=True, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(
        DL(options.dataroot, transform, 'test'),
        batch_size=options.batchSize, shuffle=True)

    win_dict = visualize_tools.win_dict()
    line_win_dict = visualize_tools.win_dict()
    line_win_dict_train_val = visualize_tools.win_dict()
    print(autoencoder_type)
    print("Training Start!")

    ep = 0
    if ep != 0:
        # encoder.load_state_dict(torch.load(os.path.join(options.outf, "face_noFERET_AE_encoder_epoch_%d.pth") % ep))
        # decoder.load_state_dict(torch.load(os.path.join(options.outf, "face_noFERET_AE_decoder_epoch_%d.pth") % ep))
        encoder.load_state_dict(torch.load(os.path.join(options.outf, "face_AE_pretrain_start_0.pth") % ep))
        decoder.load_state_dict(torch.load(os.path.join(options.outf, "face_AE_decoder_epoch_%.pth") % ep))

    for epoch in range(options.iteration):
        train_err = 0
        for i, data in enumerate(dataloader, 0):
            # autoencoder training  ====================================================================================
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            real_cpu = data
            batch_size = real_cpu.size(0)

            original_data = Variable(real_cpu).cuda()
            input.data.resize_(real_cpu.size()).copy_(make_sunglass2(real_cpu))
            #input.data.resize_(real_cpu.size()).copy_(real_cpu)

            if autoencoder_type == 'AE':
                z = encoder(input)
                x_recon = decoder(z)
                err_mse = MSE_loss(x_recon, original_data.detach())
            elif autoencoder_type == 'VAE':
                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
                x_recon = decoder(z)
                err_mse = Variational_loss(x_recon, original_data.detach(), mu, logvar)




            err_mse.backward(retain_graph=True)
            train_err += float(err_mse.data.mean())

            optimizerE.step()
            optimizerD.step()
            #visualize
            print('[%d/%d][%d/%d] Loss: %.4f'% (epoch, options.iteration, i, len(dataloader), err_mse.data.mean()))

            # We need a val_image set, not a test image set. So, move them into the dataloader_val for 3 lines
            testImage = torch.cat((unorm(original_data.data[0]),unorm(input.data[0]), unorm(x_recon.data[0])), 2)
            win_dict = visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

            line_win_dict = visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [err_mse.data.mean(),0],
                                                                     ['loss_recon_x','zero'],
                                                                     epoch, i, len(dataloader))

            #output_test=testImage.view(-1,1,64,192)
            #save_image(output_test, os.path.join(sample_dir,'sampled-{}.png'.format(epoch+1)))
        i=i+1
        train_err = (train_err/i)
        val_err = 0

        for i, data in enumerate(dataloader_val, 0):
            # autoencoder training  ====================================================================================
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            real_cpu = data
            batch_size = real_cpu.size(0)

            original_data = Variable(real_cpu).cuda()
            input.data.resize_(real_cpu.size()).copy_(make_sunglass2(real_cpu))
            # input.data.resize_(real_cpu.size()).copy_(real_cpu)

            if autoencoder_type == 'AE':
                z = encoder(input)
                x_recon = decoder(z)
                err_mse = MSE_loss(x_recon, original_data.detach())
            elif autoencoder_type == 'VAE':
                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
                x_recon = decoder(z)
                err_mse = Variational_loss(x_recon, original_data.detach(), mu, logvar, epoch)

            val_err += float(err_mse.data.mean())


            #testImage = torch.cat((unorm(original_data.data[0]), unorm(input.data[0]), unorm(x_recon.data[0])), 2)
            #win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])
            ValidImage = unorm(x_recon.data[0])
            #save_image(ValidImage, os.path.join(sample_dir,'val_sample-{}.png'.format(i+1)))
            '''
            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [err_mse.data.mean(), 0],
                                                                      ['loss_recon_x', 'zero'],
                                                                      epoch, i, len(dataloader_val))
            '''
            output_test = ValidImage.view(-1, 1, 120, 100)
            save_image(output_test, os.path.join(sample_dir, '{}.png'.format(epoch + 1)))

        i=i+1
        val_err = (val_err / i)
        line_win_dict_train_val = visualize_tools.draw_lines_to_windict(line_win_dict_train_val,
                                                                  [train_err, val_err, 0],
                                                                  ['train_err_per_epoch', 'val_err_per_epoch','zero'],
                                                                            0, epoch, options.iteration)



        # do checkpointing
        if epoch % 100 == 0:
            torch.save(encoder.state_dict(), '%s/pretrain_AE_encoder_epoch_%d.pth' % (options.outf, ep+epoch))
            torch.save(decoder.state_dict(), '%s/pretrain_AE_decoder_epoch_%d.pth' % (options.outf, ep+epoch))


def test_NLDA():

    class DL(torch.utils.data.Dataset):
        def __init__(self, path, transform, type):
            random.seed = 1
            self.transform = transform
            self.type = type
            assert os.path.exists(path)
            self.base_path = path

            total_file_paths = []
            cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR', '*'))
            total_file_paths = total_file_paths + cur_file_paths
            #cur_file_paths = glob.glob(os.path.join(self.base_path, 'FERET_normal', '*'))
            #total_file_paths = total_file_paths + cur_file_paths

            #random.shuffle(total_file_paths)
            #num_of_valset = 792

            self.train_file_paths = total_file_paths[:]
            self.test_file_paths = total_file_paths[451:]

        def pil_loader(self, path):
            with open(path, 'rb') as f:
                with Image.open(f) as img:
                    img = ImageOps.equalize(img)
                    return img.resize((80, 80))

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

    path2 = '/home/cheeze/PycharmProjects/NLDA_code/data'


    transform = transforms.Compose([
        transforms.Scale(80),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    dataloader_train = torch.utils.data.DataLoader(DL(path2, transform, 'train'), batch_size= 1, shuffle=False,
                                                   drop_last=False)

    win_dict = visualize_tools.win_dict()
    line_win_dict = visualize_tools.win_dict()
    line_win_dict_train_val = visualize_tools.win_dict()

    default = torch.FloatTensor(1, 3, 80, 80)
    default = default.cuda()

    input = torch.FloatTensor(1, 3, 80, 80)
    input = input.cuda()
    input = Variable(input)

    unorm = visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    total_train_vector = np.zeros((1, 6401))
    total_test_vector = np.zeros((1, 6401))

    for i, data in enumerate(dataloader_train, 0):
        ###### 200
        real_cpu = data
        original_data = Variable(real_cpu).cuda()
        default.data.resize_(real_cpu.size()).copy_(default_image(real_cpu))
        input.data.resize_(real_cpu.size()).copy_(make_sunglass(real_cpu))

        testImage = torch.cat((unorm(original_data.data[0]), unorm(input.data[0]), unorm(default.data[0])), 2)

        win_dict = visualize_tools.draw_images_to_windict(win_dict, [testImage], ["testImage"])
        width = len(input.data[0][0][0])
        height = len(input.data[0][0])
        input_vector = input.data[0][0].reshape(1, width * height)
        input_vector2 = default.data[0][0].reshape(1, width * height)

        label = i // 2
        label_info = np.array([label])

        input_vector2 = input_vector2.cpu().numpy()
        input_vector = input_vector.cpu().numpy()

        input_train_vector2 = np.hstack((input_vector2[0], label_info))
        input_train_vector = np.hstack((input_vector[0], label_info))
        #if (i <= 199):
        total_train_vector = np.vstack((total_train_vector, input_train_vector2))

        #if (i > 199):
        total_test_vector = np.vstack((total_test_vector, input_train_vector))

    total_train_vector = total_train_vector[1:]
    total_test_vector = total_test_vector[1:]
    NLDA_Class = Null_LDA_for_FSDD.Null_LDA('')
    NLDA_Class.NULL_LDA(total_train_vector)

    ###### 792

    std_tr = NLDA_Class.std_tr
    mean_tr = NLDA_Class.mean_tr
    w_final = NLDA_Class.w_dis_com

    KNNs = KNN_for_NLDA_Python.Face_kNN_Classification(w_final, total_train_vector, total_test_vector, mean_tr, std_tr)
    KNNs.KNN_INIT(1, 1)

if __name__ == "__main__" :
    train_ae()
    #test_NLDA()
    #test_img()





