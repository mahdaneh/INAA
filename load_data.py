import torchvision

import os

from PIL import Image
from torch.utils.data import DataLoader

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








