
import torch.utils.data as ud
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
from torchvision.utils import save_image
from torch.autograd import Variable
import utils
import visualize_tools
import numpy as np
import torchvision.models as models
#from sklearn.preprocessing import OneHotEncoder
from PIL import Image
from PIL import ImageOps
import os
import glob

#======================================================================================================================#
# Options
#======================================================================================================================#
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='Custom', help='what is dataset?')
parser.add_argument('--dataroot', default='/home/cheeze/PycharmProjects/KJW/capstone_project/capstone_project', help='path to dataset')
parser.add_argument('--pretrainedModelName', default='custom_model', help="path of Encoder networks.")
parser.add_argument('--pretrainedEpoch', type=int, default=0, help="path of Decoder networks.")
parser.add_argument('--outf', default='./result_test', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', default='True', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=3, help='number of input channel.')
parser.add_argument('--nz', type=int, default=1, help='number of latent channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')

parser.add_argument('--seed', type=int, help='manual seed')

options = parser.parse_args()
print(options)



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

        # Including each folders
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'BearHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'best_cat', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'best_dog', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'best_pig', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'DeerHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'DogHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'DuckHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'EagleHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'ElephantHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'HumanHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'LionHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'MonkeyHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'MouseHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'PandaHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'PigeonHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'PigHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'RabbitHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'SheepHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'TigerHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'WolfHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths


        random.shuffle(total_file_paths)

        num_of_valset=int(len(total_file_paths)/10)
        self.val_file_paths=sorted(total_file_paths[:num_of_valset])
        self.file_paths=sorted(total_file_paths[num_of_valset:])

        print("")
        # for testset
        #self.val_file_paths = sorted(total_file_paths)

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
                img = ImageOps.equalize(img)
                return img

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
        label = get_label(path)[:-4]
        one_hot_label = one_hot_encoding(label)
        #one_hot_label = torch.FloatTensor(one_hot_label)
        if self.transform is not None:
            img = self.transform(img)
        return (img, one_hot_label)


def get_label(path):
    return str(path.split('/')[-2])

# One hot-encoding
class_vector = ['Bear','Cat','Chicken', 'Cow', 'Deer', 'Dog', 'Duck', 'Eagle', 'Elephant', 'Lion', 'Monkey', 'Mouse', 'Panda',
                'Pigeon', 'Pig', 'Rabbit', 'Sheep', 'Tiger', 'Wolf']
Five_class_vector = ['Cat', 'Dog', 'Pig']

def one_hot_encoding(label):
    for i in range(0, 3):
        if Five_class_vector[i] == label:
            return i