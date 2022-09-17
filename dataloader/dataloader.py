import torch
import torchvision
from torchvision import transforms
import numpy as np
from configs.dataset_config import cfg as dataset_cfg
from configs.train_config import cfg as train_cfg


def get_transforms():
    transform = transforms.Compose([
        transforms.Resize(train_cfg.image_size),
        transforms.CenterCrop(train_cfg.image_size),
        transforms.ToTensor(),
        ])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return transform


def get_dataloaders(dataset='mnist'):
    """
    Initializes train, test datasets and gets their dataloaders.
    :return: train and test dataloaders
    """
    if dataset is 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                   transform=get_transforms())
        train_ids = np.where(train_dataset.train_labels.numpy() == 8)[0]
        train_dataset.data = train_dataset.data[train_ids]
        train_dataset.targets = train_dataset.targets[train_ids]
        # train_dataset.train_data = train_dataset.train_data[train_ids]
        # train_dataset.train_labels = train_dataset.train_labels[train_ids]
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                                  transform=get_transforms())
        test_ids = np.where(test_dataset.test_labels.numpy() == 8)[0]
        test_dataset.data = test_dataset.data[test_ids]
        test_dataset.targets = test_dataset.targets[test_ids]
        # test_dataset.test_data = test_dataset.test_data[test_ids]
        # test_dataset.test_labels = test_dataset.test_labels[test_ids]
    else:
        train_dataset = torchvision.datasets.LSUN(root='./data',
                                                  classes=['bedroom_train'],
                                                  transform=get_transforms())
        test_dataset = torchvision.datasets.LSUN(root='./data',
                                                 classes=['bedroom_val'],
                                                 transform=get_transforms())

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size, drop_last=True, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=train_cfg.batch_size)
    return train_dl, test_dl
