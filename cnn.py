import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torchvision.models.alexnet import AlexNet

# torchvision.disable_beta_transforms_warning()
# from torchvision.transforms.v2 import Lambda, Compose, ToTensor
from torchvision.transforms import Lambda, Compose, ToTensor

expand = Lambda(lambda x: x.expand(3, -1, -1))
transforms = Compose([
    ToTensor(),
    expand,
])

dataset = MNIST(root="/home/gridsan/jpmcd/data", transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
)

model = torch.nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.Flatten(-3),
    nn.Linear(36864, 128),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(128, 10),
)

model.to("cuda")

loss_fcn = nn.CrossEntropyLoss()
# loss_fcn.to("cuda")

opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for x, y in tqdm(dataloader):
    x = x.to("cuda")
    y = y.to("cuda")
    logits = model(x)
    loss = loss_fcn(logits, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
