import os
import glob
import torch.utils.data
import numpy as np
from PIL import Image
import utils

class multimodal_dataloader(torch.utils.data.Dataset):
    def __init__(self, RGB_path, Depth_path, Semantic_path, transform):
        super().__init__()

        assert os.path.exists(RGB_path)
        self.RGB_base_path = RGB_path

        assert os.path.exists(Depth_path)
        self.Depth_base_path = Depth_path

        #assert os.path_exists(Semantic_path)
        #self.Semantic_base_path = Semantic_path

        self.transform = transform

        RGB_left_paths = glob.glob(self.RGB_base_path + '/left/*.*')
        RGB_left_paths.sort()
        self.RGB_left_paths = RGB_left_paths

        RGB_right_paths = glob.glob(self.RGB_base_path + '/right/*.*')
        RGB_right_paths.sort()
        self.RGB_right_paths = RGB_right_paths

        Depth_lidar_paths = glob.glob(self.Depth_base_path + '/lidar/*.*')
        Depth_lidar_paths.sort()
        self.Depth_lidar_paths = Depth_lidar_paths

        Depth_disparity_paths = glob.glob(self.Depth_base_path + '/disparity/*.*')
        Depth_disparity_paths.sort()
        self.Depth_disparity_paths = Depth_disparity_paths

        '''
        Semantic_paths = glob.glob(self.Semantic_base_path + '/*.*')
        Semantic_paths.sort()
        self.Semantic_paths = Semantic_paths
        '''

    def __len__(self):
        return len(self.RGB_left_paths)

    def __getitem__(self, item):
        RGB_right_path = self.RGB_right_paths[item]
        img = self.pil_loader(RGB_right_path, 'RGB')
        RGB_right = self.transform(img)

        RGB_left_path = self.RGB_left_paths[item]
        img = self.pil_loader(RGB_left_path, 'RGB')
        RGB_left = self.transform(img)

        Depth_disparity_path = self.Depth_disparity_paths[item]
        img = self.pil_loader(Depth_disparity_path, 'L')
        D_disparity = self.transform(img)
        #D_mask = self.sparse_mask(D)

        Depth_lidar_path = self.Depth_disparity_paths[item]
        img = self.pil_loader(Depth_lidar_path, 'L')
        D_lidar = self.transform(img)
        #D_mask = self.sparse_mask(D)

        '''
        # Semantic Preprocessing =======================================================================================
        # load semantic array(binary file from piecewisecrf (from https://github.com/Vaan5/piecewisecrf))
        with open(self.Semantic_paths[item] , 'rb') as array_file:
            ndim = np.fromfile(array_file, dtype=np.uint32, count=1)[0]
            shape = []
            for d in range(ndim):
                shape.append(np.fromfile(array_file, dtype=np.uint32, count=1)[0])
            array_data = np.fromfile(array_file, dtype=np.int16)
        S_ground, S_object, S_building, S_vegetation, S_sky = self.Semantic_to_binary_mask(np.reshape(array_data, shape))
        '''
        # s_ground, s_object, s_building, s_vegetation, s_sky
        return RGB_left, RGB_right, D_lidar, D_disparity

    def pil_loader(self, path, convert = 'RGB'):
    # open path as file to avoid Resource Warning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.conver(convert)

    def Semantic_to_binary_mask(self, semantic_array):
        ground = [2, 3]
        object = [4, 6, 7, 8, 9, 10]
        building = [1]
        vegetation = [5]
        sky = [0]
        semantic_list = [ground, object, building, vegetation, sky]
        result_list = [np.zeros(semantic_array.shape) for _ in range[5]]

        for i, semantic in enumerate(semantic_list):
            for type in semantic:
                mask = (semantic_array ==type)
                masked = semantic_array *mask
                masked[masked != 0] = 1
                result_list[i] = result_list[i] + masked

        return result_list[0], result_list[1], result_list[2], result_list[3], result_list[4]

    def sparse_mask(self, sparse_matrix):
        return sparse_matrix != sparse_matrix.min()

class fold_multimodal_dataloader(torch.utils.data.Dataset):
    def __init__(self, fold_number, fold_path, transform, type = 'train'):
        super().__init__()
        self.type = type
        train_path, val_path = utils.fold_loader(fold_number, fold_path)
        if self.type == 'train':
            self.RGB_paths = train_path[1]
            self.Depth_paths = train_path[0]
        #self.Semantic_pase_path = Semantic_path
        elif self.type == 'validation':
            self.RGB_paths = val_path[1]
            self.Depth_paths = val_path[0]

        self.transform = transform

        # todo segmentation

    def __len__(self):
        return len(self.RGB_paths)

    def __getitem__(self, item):
        # RGB imgae Preprocessing
        RGB_path = self.RGB_paths[item]
        img = self.pil_loader(RGB_path, 'RGB')
        RGB = self.transform(img)

        R = RGB[0].view((1, RGB.shape[1], RGB.shape[2]))
        G = RGB[0].view((1, RGB.shape[1], RGB.shape[2]))
        B = RGB[0].view((1, RGB.shape[1], RGB.shape[2]))

        # todo image tensor(3 w h) RGB => (1 w h) * 3 R,G,B

        Depth_path = self.Depth_paths[item]
        img = self.pil_loader(Depth_path, 'L')
        D = self.transform(img)
        D_mask = self.sparse_mask(D)

        '''
        # Semantic Preprocessing =======================================================================================
        # load semantic array(binary file from piecewisecrf (from https://github.com/Vaan5/piecewisecrf))
        with open(self.Semantic_paths[item] , 'rb') as array_file:
            ndim = np.fromfile(array_file, dtype=np.uint32, count=1)[0]
            shape = []
            for d in range(ndim):
                shape.append(np.fromfile(array_file, dtype=np.uint32, count=1)[0])
            array_data = np.fromfile(array_file, dtype=np.int16)
        S_ground, S_object, S_building, S_vegetation, S_sky = self.Semantic_to_binary_mask(np.reshape(array_data, shape))
        '''
        return R, G, B, D, D_mask # s_ground, s_object, s_building, s_vegetation, s_sky

    def pil_loader(self, path, convert = 'RGB'):
    #open path as file to avoid Resource Warning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert(convert)

    def Semantic_toBinary_mask(self, semantic_array):
        ground = [2, 3]
        object = [4, 6, 7, 8, 9, 10]
        building = [1]
        vegetation = [5]
        sky = [0]
        semantic_list = [ground, object, building, vegetation, sky]
        result_list = [np.zeros(semantic_array.shape) for _ in range(5)]

        for i, semantic in enumerate(semantic_list):
            for type in semantic:
                mask = (semantic_array == type)
                masked = semantic_array * mask
                masked[masked != 0] = 1
                result_list[i] = result_list[i] + masked

        return result_list[0], result_list[1], result_list[2], result_list[3], result_list[4]

    def sparse_mask(self, sparse_matrix):
        mask = sparse_matrix != sparse_matrix.min()
        return mask




































