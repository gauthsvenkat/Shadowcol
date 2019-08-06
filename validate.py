import torch
import argparse
from utils import Pairloader, SiameseNet
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Validate SiameseNet')
parser.add_argument('--model_location', '-l', type=str, default='model/{}-epoch-{}.pth')
parser.add_argument('--epoch', '-e', type=int, default=None)
parser.add_argument('--device','-d', type=str, default=None)
args = parser.parse_args()

if not args.device:
	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SiameseNet().to(device=args.device)
model.load_state_dict(torch.load(args.model_location.format('model',args.epoch), map_location=args.device))
model.eval()

pairdata = Pairloader(split='valid')
datagen = DataLoader(pairdata)

for i, batch in enumerate(datagen):

	imgs, file_names = [batch[0][0].to(device=args.device), batch[0][1].to(device=args.device)], batch[1]

	output = model(imgs)

	print(file_names[0], " and ", file_names[1], ":- ", output.item())