from model import Generator, Discriminator

from data import (
	train_data_loader,
	test_data_loader
)

from train import (
	gen_AB,
	gen_BA,
	disc_A,
	disc_B,
	criterion_gan,
	criterion_cycle,
	criterion_identity,
	optim_gen,
	optim_disc_A,
	optim_disc_B,
	train_model
)

import json
import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 
import torch.optim as optim

import torchvision
from torchvision.utils import make_grid
from torchvision import transforms as T, datasets as dset

def model_weight(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.2)

		
if __name__ == "__main__":
	gen_AB.apply(model_weight)
	gen_BA.apply(model_weight)
	disc_A.apply(model_weight)
	disc_B.apply(model_weight)

	train_model(train_data_loader, gen_AB, gen_BA, disc_A, disc_B, criterion_gan, criterion_cycle, criterion_identity, optim_gen, optim_disc_A, optim_disc_B)
