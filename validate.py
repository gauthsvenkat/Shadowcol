import torch
import argparse
from utils import Pairloader, SiameseNet
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Validate SiameseNet')
parser.add_argument('--model_location', '-l', type=str, default='model/{}-epoch-{}.pth')
parser.add_argument('--epoch', '-e', type=int, default=None)
args = parser.parse_args()

model = SiameseNet().cuda()
model.load_state_dict(torch.load(args.model_location.format('model',args.epoch)))
model.eval()

pairdata = Pairloader(split='valid')
datagen = DataLoader(pairdata)

for i, batch in enumerate(datagen):

	imgs, file_names = [batch[0][0].cuda(), batch[0][1].cuda()], batch[1]

	output = model(imgs)

	print(file_names[0], " and ", file_names[1], ":- ", output.item())