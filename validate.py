import torch
import argparse
from utils import Pairloader, SiameseNet
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Validate SiameseNet')
parser.add_argument('--model_location', '-l', type=str, default='model/model-epoch-{}.pth')
parser.add_argument('--epoch', '-e', type=int, default=None)
parser.add_argument('--device','-d', type=str, default=None)
args = parser.parse_args()

if not args.device:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SiameseNet(mode='validate', weights_path=args.model_location.format(args.epoch), device=args.device)

datagen = DataLoader(Pairloader(split='valid'))

for i, batch in enumerate(datagen):

    data1, data2, file_names = batch[0][0].to(device=args.device), batch[0][1].to(device=args.device), batch[1]

    output = model(data1, data2)

    print(file_names[0], " and ", file_names[1], ":- ", output.item())
