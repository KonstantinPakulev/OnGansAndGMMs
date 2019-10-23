import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def Celebaloader(DATA_PATH='./data', PICTURE_SIZE=64, BATCH_SIZE=128):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(PICTURE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CelebA(root=DATA_PATH, split='train',
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True)
    testset = torchvision.datasets.CelebA(root=DATA_PATH, split='test',
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False)

    return trainset, trainloader, testset, testloader
