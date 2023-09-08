from train import load_config

import os
import time
import json
import random
import itertools

from tqdm import tqdm
from glob import glob
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset
from torchvision.utils import make_grid

#Dataset parameter setup
config = load_config('config.json')

root = config['root_data_dir']
n_cpu = 3
batch_size = config['batch_size']
width = config['width']
height = config['height']

#Dataset 
class HorseZebraDataset(Dataset):
	def __init__(self, root_dir, train, transforms):
		super(HorseZebraDataset, self).__init__()
		self.root_dir = root
		self.train = train
		self.transforms = transforms

		path_a = self.root_dir + ('trainA' if train else 'testa')
		path_b = self.root_dir + ('trainB' if train else 'testB')

		self.imgs_a = sorted(glob(path_a + '/*.jpg'))
		self.imgs_b = sorted(glob(path_b + '/*.jpg'))	

	def __getitem__(self, index):
		a_len = len(self.imgs_a)
		b_len = len(self.imgs_b)

		a = self.imgs_a[index % a_len]
		b = self.imgs_b[index % b_len]

		a = Image.open(a)
		b = Image.open(b)

		a = a.convert('RGB')
		b = b.convert('RGB')

		if self.transforms:
			a = self.transforms(a)
			b = self.transforms(b)

		return a, b

	def __len__(self):
		return max(len(self.imgs_a), len(self.imgs_b))

#Transforms
transforms = T.Compose([
	T.Resize((width, height)),
	T.ToTensor()
])

train_dataset = HorseZebraDataset(root_dir=root, train=True, transforms=transforms)
test_dataset = HorseZebraDataset(root_dir=root, train=False, transforms=transforms)

#DataLoader
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True,num_workers=n_cpu)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
