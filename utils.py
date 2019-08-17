import librosa
import numpy as numpy
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
import os
import itertools
from tqdm import tqdm

class Pairloader(Dataset):
	def __init__(self, root_dir='data', split='train'):

		self.root_dir = root_dir #set root directory housing the data. (Folder should contain train/ and valid/)
		self.split = split #train or valid
		self.audio_files = glob(os.path.join(root_dir,split,'*.wav')) #get all the wav files
		self.all_combinations = list(itertools.combinations(self.audio_files, 2)) #get all combinations of 2 files from the dataset
		self.SR = 16000 #set sampling rate to 16000

	def __len__(self):
		return len(self.all_combinations)

	def __getitem__(self, idx):
		
		audio_file1, audio_file2 = self.all_combinations[idx] #randomly choose two files from audio_files

		audio1, audio2 = [librosa.load(file, sr=self.SR)[0] for file in [audio_file1, audio_file2]] #load audios
		audio1_trimmed, audio2_trimmed = [librosa.effects.trim(audio, top_db=7)[0] for audio in [audio1, audio2]] #trim trailing silence
		audio1_center, audio2_center = [librosa.util.pad_center(audio[:4000], 4000) for audio in [audio1_trimmed, audio2_trimmed]] #center pad the audio to 4000

		assert len(audio1_center) == len(audio2_center), "Audio lengths are not the same !" #make sure you're sane

		audio1_mfcc, audio2_mfcc = [librosa.feature.mfcc(y=audio, sr=self.SR) for audio in [audio1_center, audio2_center]] #compute the mfccs for both audios

		assert audio1_mfcc.shape == audio2_mfcc.shape, "MFCC shapes are not the same !" #make sure you're sane

		if self.split == 'train':
			return [audio1_mfcc[None], audio2_mfcc[None]], torch.tensor([1], dtype=torch.float) if audio_file1.split('_')[0] == audio_file2.split('_')[0] else torch.tensor([0], dtype=torch.float) #return both mfccs and labels if split is train
		elif self.split == 'valid':
			return [audio1_mfcc[None], audio2_mfcc[None]], [audio_file1, audio_file2] #else return only mfccs


class SiameseNet(nn.Module):

	def __init__(self):
		super(SiameseNet, self).__init__()

		self.maxpool = nn.MaxPool2d(2)

		self.conv1 = nn.Conv2d(1, 64, 4)
		self.conv2 = nn.Conv2d(64, 128, 3)
		self.conv3 = nn.Conv2d(128, 128, 2)
		
		self.linear1 = nn.Linear(896, 512)
		self.linear2 = nn.Linear(512, 1)

	def forward(self, data):
		res = []

		for i in [0,1]:
			x = data[i]

			x = self.conv1(x)
			x = F.relu(x)

			x = self.conv2(x)
			x = F.relu(x)

			x = self.conv3(x)
			x = F.relu(x)
			x = self.maxpool(x)

			x = x.view(x.shape[0], -1)

			x = self.linear1(x)
			x = F.relu(x)

			res.append(x)

		res = torch.abs(res[1] - res[0])
		res = self.linear2(res)
		res = torch.sigmoid(res)

		return res


class _tqdm(tqdm):
	def format_num(self, n):
		f = '{:.5f}'.format(n)
		return f
