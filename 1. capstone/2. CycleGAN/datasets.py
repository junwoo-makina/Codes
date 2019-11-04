import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageOps
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root_human, root_animal, transforms=None, unaligned=False, mode='train'):
        self.transform = transforms
        self.unaligned = unaligned
        self.human_path = root_human
        self.animal_path = root_animal

        total_human_path = []
        cur_human_path = glob.glob(os.path.join(self.human_path, 'img_align_celeba', '*'))
        total_human_path = total_human_path + cur_human_path


        total_animal_path = []
        cur_animal_path = glob.glob(os.path.join(self.animal_path, 'BearHead', '*'))
        total_animal_path = total_animal_path + cur_animal_path




        random.shuffle(total_animal_path)
        random.shuffle(total_human_path)

        self.files_human = total_human_path[:2000]
        self.files_animal =total_animal_path

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
                img = ImageOps.equalize(img)
                return img



    def __getitem__(self, item):
        #item_A = self.transform(Image.open(self.files_human[item%len(self.files_human)]))
        item_A = self.pil_loader(self.files_human[item%len(self.files_human)])
        item_B = self.pil_loader(self.files_animal[item%len(self.files_animal)])

#        if self.unaligned:
#            item_B = self.transform(Image.open(self.files_animal[random.randint(0, len(self.files_animal)-1)]))
#        else:
#            item_B = self.transform(Image.open(self.files_animal[item%len(self.files_animal)]))
        if self.transform is not None:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)
        return {'A': item_A, 'B': item_B}


    def __len__(self):
        return max(len(self.files_animal), len(self.files_human))