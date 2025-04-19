import numpy as np
import torch
import torchvision
from torchvision import transforms
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def load_cifar10_cat_dataset(data_dir):
    train_transform, test_transform = _data_transforms_cifar10()
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    cifar10_classes = test_dataset.classes

    train_cat_indices = [idx for idx, label in enumerate(train_dataset.targets) if label == 3]  # 选择训练集中标签为 3（猫）的样本索引
    test_cat_indices = [idx for idx, label in enumerate(test_dataset.targets) if label == 3]  # 选择测试集中标签为 3（猫）的样本索引

    train_cat_dataset = torch.utils.data.Subset(train_dataset, train_cat_indices)
    test_cat_dataset = torch.utils.data.Subset(test_dataset, test_cat_indices)

    return train_cat_dataset, test_cat_dataset,cifar10_classes

