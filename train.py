import torch
import argparse
import torch.optim as optim
import torch.nn as nn
from utils import Pairloader, SiameseNet, _tqdm as tqdm
from torch.utils.data import DataLoader
import os

parser = argparse.ArgumentParser(description='Train SiameseNet')
parser.add_argument('--save_location', '-sl', type=str, default='model/{}-epoch-{}.pth')
parser.add_argument('--epochs', '-e', type=int, default=50)
parser.add_argument('--save_every', '-se', type=int, default=5)
parser.add_argument('--device', '-d', type=str, default=None)
args = parser.parse_args()

if not args.device:
	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SiameseNet().to(device=args.device)
datagen = DataLoader(Pairloader(), shuffle=True)
bce_loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(args.epochs):
	model.train()
	epoch_loss = 0.0

	with tqdm(datagen) as t:
		for i, batch in enumerate(t):

			t.set_description('EPOCH: %i'%epoch)

			imgs, label = [batch[0][0].to(device=args.device), batch[0][1].to(device=args.device)], batch[1].to(device=args.device)

			optimizer.zero_grad()
			output = model(imgs)
			loss = bce_loss(output, label)
			loss.backward()
			optimizer.step()

			epoch_loss+=loss.item()
			t.set_postfix(loss=epoch_loss/(i+1))

	print('Loss-{}'.format(loss.item()/(i+1)))

	if (epoch+1)%args.save_every == 0:
		if not os.path.exists('model/'):
			os.mkdir('model/')

		torch.save(model.state_dict(),args.save_location.format('model', epoch))
		#torch.save(optimizer.state_dict(), args.save_location.format('optimizer', epoch))
