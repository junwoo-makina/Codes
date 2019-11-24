import torch.utils.data as ud
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import utils
import visualize_tools
import numpy as np
#import Null_LDA_for_FSDD
#import KNN_for_NLDA_Python
from scipy.misc import imsave
import scipy.io as sio

from PIL import Image
from PIL import ImageOps
import os
import glob
import torch.utils.data

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='celebA', help='what is dataset?')
parser.add_argument('--dataroot', default='/home/cheeze/PycharmProjects/KJW/research/face_autoencoder/input', help='path to dataset')
parser.add_argument('--pretrainedModelName', default='autoencoder', help="path of Encoder networks.")
parser.add_argument('--pretrainedEpoch', type=int, default=0, help="path of Decoder networks.")
parser.add_argument('--outf', default='./pretrain_sunglass', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', default='True', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=10000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=120, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=3, help='number of input channel.')
parser.add_argument('--nz', type=int, default=2, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')

parser.add_argument('--seed', type=int, help='manual seed')

options = parser.parse_args()
print(options)

sample_dir='samples'
path = '/home/cheeze/PycharmProjects/NLDA_code/data'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

def Variational_loss(input, target, mu, logvar, epoch):
    recon_loss = MSE_loss(input, target)
    KLD_loss = -0.5 * torch.sum(1+logvar-mu.pow(2) - logvar.exp())
    result=recon_loss+ KLD_loss
    if(result >35000 and epoch > 350):
        print("%4f %4f", recon_loss, KLD_loss)
    return recon_loss + KLD_loss

class CelebA_DL(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.transform = transform

        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()

        cur_file_paths = glob.glob(self.base_path + '/*.*')
        cur_file_paths.sort()
        self.file_paths = cur_file_paths

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('L')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        path = self.file_paths[item]
        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class DL(torch.utils.data.Dataset):
    def __init__(self, path, transform, type):
        random.seed = 1
        super().__init__()
        self.transform = transform
        self.type = type
        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()
        total_file_paths = []

        #these '# codes'are for elimination of glasses.
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'glasses', '*'))
        #total_file_paths = total_file_paths + cur_file_paths

        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'CMU_PIE_normal', '*'))
        #total_file_paths = total_file_paths+cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'YaleB_normal', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'Yale_normal', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR_normal', '*'))
        #total_file_paths = total_file_paths + cur_file_paths


        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'CMU_PIE', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'YaleB', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'Yale', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR', '*'))
        #total_file_paths = total_file_paths + cur_file_paths

        #for occlusion


        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR_session2_test', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'Feret_test', '*'))
        total_file_paths = total_file_paths + cur_file_paths

        #random.shuffle(total_file_paths)

        num_of_valset=int(len(total_file_paths)/10)
        self.val_file_paths=sorted(total_file_paths[:num_of_valset])
        self.file_paths=sorted(total_file_paths[:])

        # for testset
        #self.val_file_paths = sorted(total_file_paths)

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = ImageOps.equalize(img)
                img = ImageOps.grayscale(img)
                return img.resize((120,100))

    def __len__(self):
        if self.type == 'train':
            return len(self.file_paths)
        elif self.type == 'test':
            return len(self.val_file_paths)

    def __getitem__(self, item):
        if self.type == 'train':
            path = self.file_paths[item]
        elif self.type == 'test':
            path = self.val_file_paths[item]

        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img


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

def make_sunglass(img):
    image_ = img

    # make sunglass
    image_[:, :, 26:56, 8:40] = -1    # whole, height, width
    image_[:, :, 26:56, 64:96] = -1
    image_[:, :, 30:40, 40:64] = -1
    return image_


def make_random_squre(image):
    image_ = image
    x = random.randrange(0, 120)
    y = random.randrange(0, 100)
    w = random.randrange(5, 119)
    h = random.randrange(5, 99)
    if y+h < 100 and x+w < 120:
        image_[:, :, x:x + w, y:y + h] = -1
        return image_
    else:
        return make_random_squre(image_)





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
ngpu = int(options.ngpu)
nz = int(options.nz)
autoencoder_type = 'AE'
encoder = encoder(options.nz, options.nc, 64, autoencoder_type)
encoder.apply(utils.weights_init)
#if options.netG != '':
 #   encoder.load_state_dict(torch.load(options.netG))
print(encoder)

decoder = decoder(options.nz, options.nc)
decoder.apply(utils.weights_init)
#if options.netD != '':
 #   decoder.load_state_dict(torch.load(options.netD))
print(decoder)
#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss(size_average=False)
L1_loss = nn.L1Loss(size_average=False)


# setup optimizer   ====================================================================================================
optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=2e-3)
optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-3)


# container generate
input = torch.FloatTensor(1, 3, 120, 100)

if options.cuda:
    encoder.cuda()
    decoder.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()
    L1_loss.cuda()
    input = input.cuda()


# make to variables ====================================================================================================
input = Variable(input)



def test_img():
    win_dict = visualize_tools.win_dict()
    print("Testing Start!")

    utils.make_dir("/home/cheeze/PycharmProjects/pytorch//bro/Pytorch/Autoencoder/result_test")

    ep = 7000


    encoder.load_state_dict(torch.load(os.path.join('/home/cheeze/PycharmProjects/KJW/research/face_autoencoder/pretrain_model', "pretrain_AE_encoder_epoch_80.pth")))
    decoder.load_state_dict(torch.load(os.path.join('/home/cheeze/PycharmProjects/KJW/research/face_autoencoder/pretrain_model', "pretrain_AE_decoder_epoch_80.pth")))

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Scale((120,100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    unorm = visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    dataloader = torch.utils.data.DataLoader(DL(options.dataroot, transform,'train'),
                                             batch_size=1, shuffle=False)

    total_train_vector = np.zeros((1,12001))
    total_test_vector = np.zeros((1,12001))

    for i, data in enumerate(dataloader, 0):
        real_cpu = data
        original_data = Variable(real_cpu).cuda()

        input.data.resize_(real_cpu.size()).copy_(make_sunglass(real_cpu))
        #input.data.resize_(real_cpu.size()).copy_(real_cpu)

        z = encoder(input)
        x_recon = decoder(z)
        err_mse = MSE_loss(x_recon, original_data.detach())

        testImage = torch.cat((unorm(original_data.data[0]), unorm(x_recon.data[0]), unorm(input.data[0])), 2)
        toimg = transforms.ToPILImage()
        toimg(testImage.cpu()).save("/home/cheeze/PycharmProjects/KJW/research/face_autoencoder/samples/%d.jpg" % i)
        #output_test = testImage.view(-1, 1, 120, 300)
        #save_image(output_test, os.path.join(sample_dir,'sampled-{}.jpg'.format(i+1)))

        data = x_recon.detach().cpu().numpy()
        data = np.resize(data, (1,12000))

        label = i//2
        label = label
        label_info = np.array([label])

        input_train_vector = np.hstack((data[0], label_info))
        total_train_vector = np.vstack((total_train_vector, input_train_vector))

    total_train_vector = total_train_vector[1:]
    sio.savemat('/home/cheeze/PycharmProjects/KJW/research/face_autoencoder/Feret_recon2.mat', {"Feret_recon2":total_train_vector})

if __name__ == "__main__" :
    #train()
    #test()
    test_img()






