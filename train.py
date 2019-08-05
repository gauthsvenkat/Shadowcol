import torch
import argparse
import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
from utils import PairLoader, SiameseNet
from torch.utils.data import DataLoader
import os

parser = argparse.ArgumentParser(description='Train SiameseNet')
parser.add_argument('--save_location', '-sl', type=str, default='model/{}-epoch-{}.pth')
parser.add_argument('epochs', '-e', type=int, default=50)
parser.add_argument('--load_model','-lm', type=str, default=None)
parser.add_argument('--save_every', '-se', type=int, default=5)

model = SiameseNet().cuda()
pairdata = PairLoader()
datagen = DataLoader(pairdata)
bce_loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(args.epochs):
	model.train()
	epoch_loss = 0.0
	for i, batch in enumerate(datagen):

		imgs, label = [batch[0][0].cuda(), batch[0][1].cuda()], batch[1].cuda()

		optimizer.zero_grad()

		output = model(imgs)
		loss = bce_loss(output, label)
		loss.backward()
		optimizer.step()

		epoch_loss+=loss.item()

	print('Epoch-{}, Loss-{}'.format(epoch, loss.item()/i+1))

	if (epoch+1)%args.save_every == 0:
		if not os.path.exists('model/'):
			os.mkdir('model/')

		torch.save(model.state_dict(),args.save_location.format('model', epoch))
		torch.save(optimizer.state_dict(), args.save_location.format('optimizer', epoch))
		




