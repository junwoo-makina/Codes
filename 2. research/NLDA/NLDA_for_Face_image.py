import numpy as np
import os
import torch
import torchvision
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


from PIL import ImageDraw



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample_dir = 'samples'
path = '/home/cheeze/PycharmProjects/NLDA_code/img'

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

        cur_file_paths = glob.glob(os.path.join(self.base_path, 'FERET_normal', '*'))
        total_file_paths = total_file_paths + cur_file_paths

        self.train_file_paths = total_file_paths[:]
        self.test_file_paths = total_file_paths[201:]

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


transform = transforms.Compose([
                transforms.Scale(80),
                transforms.ToTensor(),
                transforms.Normalize(mean =(0.5, 0.5, 0.5),
                                    std= (0.5, 0.5, 0.5))
])

dataloader_train = torch.utils.data.DataLoader(DL(path, transform, 'train'), batch_size=1, shuffle=False, drop_last=False)
dataloader_test = torch.utils.data.DataLoader(DL(path,transform,'test'), batch_size=1, shuffle=False, drop_last=False)

win_dict = visualize_tools.win_dict()
line_win_dict = visualize_tools.win_dict()
line_win_dict_train_val = visualize_tools.win_dict()


default = torch.FloatTensor(1,3,80,80)
default = default.cuda()


input = torch.FloatTensor(1,3,80,80)
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

    win_dict = visualize_tools.draw_images_to_windict(win_dict, [testImage],["testImage"])
    width = len(input.data[0][0][0])
    height = len(input.data[0][0])
    input_vector = input.data[0].reshape(1, width*height)
    input_vector2 = default.data[0].reshape(1, width*height)


    label = i//2
    label_info = np.array([label])


    input_vector2 = input_vector2.cpu().numpy()
    input_vector = input_vector.cpu().numpy()

    input_train_vector2 = np.hstack((input_vector2[0], label_info))
    input_train_vector = np.hstack((input_vector[0],label_info))
    if(i <= 199):
        total_train_vector = np.vstack((total_train_vector, input_train_vector2))

    if(i>199):
        total_test_vector = np.vstack((total_test_vector, input_train_vector))


total_train_vector = total_train_vector[1:]
total_test_vector = total_test_vector[1:]
NLDA_Class = Null_LDA_for_FSDD.Null_LDA('')
NLDA_Class.NULL_LDA(total_train_vector, label_info)


###### 792

std_tr = NLDA_Class.std_tr
mean_tr = NLDA_Class.mean_tr
w_final = NLDA_Class.w_dis_com

KNNs =KNN_for_NLDA_Python.Face_kNN_Classification(w_final, total_train_vector, total_test_vector, mean_tr,std_tr)
KNNs.KNN_INIT(1, 1)





































