from PIL import Image
import torch
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, images_bag_id: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.images_bag_id = images_bag_id
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.images_class[item]
        bag_id = self.images_bag_id[item]
        
        return img, label, bag_id, self.images_path[item]

    @staticmethod
    def collate_fn(batch):
        images, labels, bag_ids, paths = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, bag_ids, paths
