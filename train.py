from model import Generator, Discriminator

import json
import os
import random
import time
import itertools
from tqdm import tqdm
import numpy as np

import torch, torchvision
import torch.nn as nn
import torch.optim as optim


def load_config(config_file):
	with open(config_file,"r") as f:
		config = json.load(f)
	return config

def weight_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.2)

config = load_config('config.json')

lr = config['learning_rate']
epochs = config['epochs']
batch_size = config['batch_size']
betas = config['betas']
seed = 44
lambda_ = config['lambda']
n_residual_blocks = config['n_residual_blocks']
n_cpu = 3
width = height = 128
channels = 3
input_shape = (channels, width, height)
device = config['device']

#Training

gen_AB = Generator(channels, n_residual_blocks).to(device)
gen_BA = Generator(channels, n_residual_blocks).to(device)

disc_A = Discriminator(input_shape).to(device)
disc_B = Discriminator(input_shape).to(device)

# gen_AB.apply(weight_init_normal)
# gen_BA.apply(weight_init_normal)
# disc_A.apply(weight_init_normal)
# disc_B.apply(weight_init_normal)

#Loss Functions
criterion_gan = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

#Optimizers
optim_gen = optim.Adam(itertools.chain(gen_AB.parameters(), gen_BA.parameters()), lr=lr, betas=betas)

optim_disc_A = optim.Adam(disc_A.parameters(), lr=lr, betas=betas)
optim_disc_B = optim.Adam(disc_B.parameters(), lr=lr, betas=betas)


cyclic_coefficient_loss = config['cyclic_loss_coefficient']
identity_loss_coefficient = config['identity_loss_coefficient']
loop = config['loop']



def train_model(train_dataset_loader, gen_AB, gen_BA, disc_A, disc_B, criterion_gan, criterion_cycle, criterion_identity, optim_gen, optim_disc_A, optim_disc_B):
	#Bookkeeping Losses
	current_epoch = 0
	gen_losses = []
	disc_losses = []
	adv_losses = []
	cycle_losses = []
	identity_losses = []
	image_list = []
	iterations = 0

	for epoch in range(current_epoch, epochs):
		for idx, data in enumerate(tqdm(train_dataset_loader)):
			real_A = data[0].to(device)
			real_B = data[0].to(device)

			real = torch.Tensor(np.ones((real_A.size(0), *disc_A.output_shape))).to(device)
			fake = torch.Tensor(np.zeros((real_A.size(0), *disc_A.output_shape))).to(device)

			gen_AB.train()
			gen_BA.train()

			optim_gen.zero_grad()

			fake_B = gen_AB(real_A)
			gen_AB_loss = criterion_gan(disc_B(fake_B), real)

			fake_A = gen_BA(real_B)
			gen_BA_loss = criterion_gan(disc_A(fake_A), real)

			adverserial_loss = (gen_AB_loss + gen_BA_loss)/2

			identity_A = gen_BA(real_A)
			identity_A_loss = criterion_identity(identity_A, real_A)

			identity_B = gen_AB(real_B)
			identity_B_loss = criterion_identity(identity_B, real_B)

			identity_loss = (identity_A_loss + identity_B_loss) / 2

			cycle_A = gen_BA(fake_B)
			cycle_A_loss = criterion_cycle(cycle_A, real_A)

			cycle_B = gen_AB(fake_A)
			cycle_B_loss = criterion_cycle(cycle_B, real_B)

			cycle_loss = (cycle_B_loss + cycle_A_loss) / 2

			gen_loss = (adverserial_loss + 
						identity_loss_coefficient*identity_loss 
						+ cyclic_coefficient_loss*cycle_loss)

			gen_loss.backward()
			optim_gen.step()

			optim_disc_A.zero_grad()
			
			real_A_loss = criterion_gan(disc_A(real_A), real)
			fake_A_loss = criterion_gan(disc_A(fake_A.detach()), fake)
			disc_A_loss = (real_A_loss + fake_A_loss) / 2

			disc_A_loss.backward()
			optim_disc_A.step()

			optim_disc_B.zero_grad()
			
			real_B_loss = criterion_gan(disc_B(real_B), real)
			fake_B_loss = criterion_gan(disc_B(fake_B.detach()), fake)
			disc_B_loss = (real_B_loss + fake_B_loss) / 2

			disc_B_loss.backward()
			optim_disc_B.step()

			disc_loss = (disc_B_loss + disc_A_loss) / 2

			if (idx + 1) % loop == 0:
				print(f'Working')

			iterations += 1
			gen_losses.append(gen_loss.item())
			identity_losses.append(identity_loss.item())
			cycle_losses.append(cycle_loss.item())
			adv_losses.append(adverserial_loss.item())
			disc_losses.append(disc_loss.item())
