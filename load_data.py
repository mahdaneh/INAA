import torchvision

import os
import torch

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pretrainedmodels
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# TODO: specify the return type
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path: str):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def imagnetlabel_2_10cls (dir):
    selected_classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    selected_classes.sort()
    all_classes = open('data/imagenet_classes.txt', 'r').readlines()
    find_index = lambda all_classes, given_class: [i for i, cls in enumerate(all_classes) if
                                                   (cls.strip() == given_class)]

    indx_2_10cls = { find_index(all_classes, cls_name)[0]:i for i, cls_name in enumerate(selected_classes)}


    # print ( {cls_name: i for i, cls_name in enumerate(selected_classes)})
    # print (indx_2_10cls)

    return indx_2_10cls

class load_local_data(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, transform, convert_2_10cls=False):
        self.transform = transform
        self.data = torch.load(image_path)
        print (self.data.shape)
        self.labels = torch.load(label_path)
        if convert_2_10cls:
            index_2_10cls = imagnetlabel_2_10cls('data/imagenette')

            labels = [ index_2_10cls[X.item()] for X in self.labels]

            self.labels = labels
            print (np.unique(self.labels))


    def __getitem__(self, index):

        img, target = self.data[index], (self.labels[index])


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.asarray(img*255, dtype='uint8') if np.max(img)<=1 else np.asarray(img, dtype='uint8')
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        print (len(self.data))
        return len(self.data)

class ImageFolder_imagenette(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            root,loader=default_loader,
            transform= None):

        super(ImageFolder_imagenette, self).__init__(root,  loader, extensions=IMG_EXTENSIONS,
                                          transform=transform)

        self.imgs = self.samples


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        selected_classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        selected_classes.sort()
        all_classes = open('data/imagenet_classes.txt', 'r').readlines()
        find_index = lambda all_classes, given_class: [i for i, cls in enumerate(all_classes) if (cls.strip() == given_class)]

        class_to_idx = {cls_name: find_index(all_classes, cls_name)[0] for cls_name in selected_classes}
        # class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return selected_classes, class_to_idx



import math
import torchvision.transforms as transforms
from PIL import Image
from munch import munchify

class TransformImage_nomean(object):

    def __init__(self, opts, scale=0.875, random_crop=False,
                 random_hflip=False, random_vflip=False,
                 preserve_aspect_ratio=True):
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range


        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        tfs = []
        if preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale))))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)
            tfs.append(transforms.Resize((height, width)))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        tfs.append(transforms.ToTensor())
        tfs.append(pretrainedmodels.utils.ToSpaceBGR(self.input_space=='BGR'))
        tfs.append(pretrainedmodels.utils.ToRange255(max(self.input_range)==255))


        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor

import pickle
import numpy as np
class adversaries (Dataset):
    def __init__(self, filepath, transform, epsilon):
        data = pickle.load(open(filepath, 'rb'))
        self.labels = data['true label']
        self.images = data['advs%.4f' % epsilon]
        self.transform = transform

    def __getitem__(self, index):
        img = np.asarray(self.images[index])

        lbl = (self.labels[index])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl

    def __len__(self):
        return len(self.images)



