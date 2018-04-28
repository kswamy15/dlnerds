import os, sys
import pandas as pd
from collections import defaultdict
import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader

class myImageFolder(data.Dataset):
    """It is ImageFolder class that has been made to load images that is not arranged in class folders
    """
    def __init__(self, label_file, root, permitted_filenames=None, transform=None, target_transform=None):
        self.root = root
        labels, classes, class_to_idx, class_freq = self._init_classes(label_file)
        self.labels = labels
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.class_freq = class_freq
        self.imgs = self.make_dataset(permitted_filenames)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        
    def _init_classes(self, label_file):
        df = pd.read_csv(label_file)
        labels = {k: v.split() for k,v in df.values}

        class_freq = defaultdict(int)
        for tags in labels.values():
            for tag in tags:
                class_freq[tag] += 1

        classes = list(class_freq.keys())
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return labels, classes, class_to_idx, class_freq

    def make_dataset(self, permitted_filenames):
        images = []
        for filename in os.listdir(self.root):
            if permitted_filenames is None or filename in permitted_filenames:
                labels = set(self.labels[filename.split('.')[0]])
                target = [int(class_name in labels) for class_name in self.classes]
                images.append((filename, torch.FloatTensor(target)))

        return images

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
