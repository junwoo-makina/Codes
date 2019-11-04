'''
Reference : https://ieeexplore.ieee.org/document/4147155
Implement : pytorch, numpy, visdom

Created by JunWoo Kim,
Date : 2019.4.24
'''

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
parser.add_argument('--dataroot', default='/home/cheeze/PycharmProjects/KJW/capstone_project/Image', help='path to dataset')
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
#======================================================================================================================#








#======================================================================================================================#
# Data Loader
#======================================================================================================================#

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
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'CatHead', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'ChickenHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'CowHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'DeerHead', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'DogHead', '*'))
        total_file_paths = total_file_paths + cur_file_paths
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
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'PigHead', '*'))
        total_file_paths = total_file_paths + cur_file_paths
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


# Get Label Method
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


def one_hot_encoding2(label):
    one_hot_vector =[0,0,0]
    for i in range(0,3):
        if Five_class_vector[i] == label:
            one_hot_vector[i] = 1
            #one_hot_vector.append(1)
            return one_hot_vector
        #elif class_vector[i] is not label:
        #    one_hot_vector.append(0)
    #return one_hot_vector

#======================================================================================================================#








#======================================================================================================================#
# Model Build
#======================================================================================================================#
class CNN(nn.Module):
    def __init__(self, z_size=1, channel=3, num_filters = 64):
        super().__init__()
        self.resnet = models.resnet50()
        self.fc1 = nn.Linear(1000, 3)

    def forward(self, x):
        z = self.resnet(x)
        #z = z.view(-1, 16)
        output = self.fc1(z)
        return output

    def weight_init(self):
        self.resnet(required_grad = False)
        self.fc1.apply(weight_init)


def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
#======================================================================================================================#






#======================================================================================================================#
# Use Resnet model
#======================================================================================================================#
resnet50 = models.resnet50()
print(resnet50)
#======================================================================================================================#






#======================================================================================================================#
# Seed, Directories, CUDA setting
#======================================================================================================================#

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



# CUDA Setting
if options.cuda:
    torch.cuda.manual_seed(options.seed)

torch.backends.cudnn.benchmark = True
cudnn.benchmark = True
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#======================================================================================================================#





#======================================================================================================================#
# Data and Parameters
#======================================================================================================================#

ngpu = int(options.ngpu)
nz = int(options.nz)
cnn_module = CNN(options.nz, options.nc, options.imageSize)
#cnn_module.apply(utils.weights_init)

print(cnn_module)

#======================================================================================================================#






#======================================================================================================================#
# Training Settings
#======================================================================================================================#

# Criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss(size_average = False)
L1_loss = nn.L1Loss(size_average = False)

criterion_NN = nn.CrossEntropyLoss()


# Set up optimizer
Optimizer_CNN = optim.SGD(cnn_module.parameters(), lr = 0.001, momentum = 0.9)
Optimizer_Resnet = optim.SGD(resnet50.parameters(), lr= 0.001, momentum = 0.9)


# Container generate
input = torch.FloatTensor(options.batchSize, options.nc, options.imageSize, options.imageSize)
label = torch.FloatTensor(options.batchSize)

if options.cuda:
    cnn_module.cuda()
    criterion_NN.cuda()
    input, label = input.cuda(), label.cuda()
    resnet50.cuda()


# Make to variables
input_image = Variable(input)
input_label = Variable(label)

# Visualize setting
win_dict = visualize_tools.win_dict()
line_win_dict = visualize_tools.win_dict()
line_win_dict_val = visualize_tools.win_dict()
#======================================================================================================================#






#======================================================================================================================#
# Data Call and load
#======================================================================================================================#
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop((200,200)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

unorm = visualize_tools.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

dataloader = torch.utils.data.DataLoader(
    DL(options.dataroot, transform, 'train'), batch_size=options.batchSize, shuffle=True, num_workers=0
    )
dataloader_val = torch.utils.data.DataLoader(
    DL(options.dataroot, transform, 'test'), batch_size=5, shuffle=True, num_workers=0
)
#======================================================================================================================#







#======================================================================================================================#
# Training Start !
#======================================================================================================================#
print("Training Start!")
ep = 0
if ep !=0 :
    cnn_module.load_state_dict(torch.load(os.path.join(options.outf, "%d_epoch.pth")%ep))
    print("Load pretrained network parameter")
for epoch in range(options.iteration):
    train_err = 0
    for i, (input, label) in enumerate(dataloader, 0):
        Optimizer_CNN.zero_grad()

        # Seperate variables
        image_cpu = input
        label_cpu = label

        original_image = Variable(image_cpu).cuda()
        input_image.data.resize_(image_cpu.size()).copy_(image_cpu)
        input_label.data.resize_(label_cpu.size()).copy_(label_cpu)


        # Execute convolutional neural-network
        result_z = cnn_module(input_image)
        err_mse = criterion_NN(result_z, input_label.long())

        err_mse.backward(retain_graph=True)
        train_err += float(err_mse.data.mean())



        # Back-propagation
        Optimizer_CNN.step()
        print('[%d/%d][%d/%d]  Loss: %.4f'%(epoch, options.iteration, i, len(dataloader), err_mse.data.mean()))



        # Visualize train images

        test_Image = torch.cat((unorm(original_image.data[0]), unorm(input_image.data[0])), 1)
        win_dict = visualize_tools.draw_images_to_windict(win_dict, [test_Image], ["Autoencoder"])
        line_win_dict = visualize_tools.draw_lines_to_windict(line_win_dict,
                                                              [err_mse.data.mean(), 0],
                                                              ['Train_Loss', 'Zero'],
                                                              epoch, i, len(dataloader))
        #print(result_z[1:5])
        #print(input_label[1:5])

    torch.save(cnn_module.state_dict(), '/home/cheeze/PycharmProjects/KJW/capstone_project/classify_synthesis/animal_classify_%d_epoch.pth'%epoch)



    for i, (data, label) in enumerate(dataloader_val, 0):
        var_err = 0
        # Seperate variables
        image_cpu = data
        label_cpu = label

        original_image = Variable(image_cpu).cuda()
        input_image.data.resize_(image_cpu.size()).copy_(image_cpu)
        input_label.data.resize_(label_cpu.size()).copy_(label_cpu)

        # Execute convolutional neural-network
        result_z = cnn_module(input_image)
        err_mse = criterion_NN(result_z, input_label.long())
        var_err += float(err_mse.data.mean())

        line_win_dict_val = visualize_tools.draw_lines_to_windict(line_win_dict_val,
                                                               [train_err, var_err*10, 0],
                                                               ['Train_Loss', 'Validation_Loss', 'zero'],
                                                            epoch,i, options.iteration)




























































