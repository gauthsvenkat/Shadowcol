import librosa
import numpy as np
import torch
import os
import argparse
from utils import SiameseNet
from glob import glob

parser = argparse.ArgumentParser(description='Test SiameseNet')
parser.add_argument('--model_location', '-l', type=str, default='model/model-epoch-{}.pth')
parser.add_argument('--epoch', '-e', type=int, default=None)
parser.add_argument('--device', '-d', type=str, default='cpu')
parser.add_argument('--ref', '-r', type=str, default='references/')
parser.add_argument('--test_dir', type=str, default='data/test/*.wav')
args = parser.parse_args()

RATE = 16000

def preprocess(audio=None):
    audio_trimmed = librosa.effects.trim(audio, top_db=7)[0]
    audio_center = librosa.util.pad_center(audio_trimmed[:4000], 4000)
    audio_mfcc = librosa.feature.mfcc(y=audio_center, sr=RATE)
    audio_tensor = torch.tensor(audio_mfcc[None,None])
    
    return audio_tensor.to(device=args.device)

refs = {
        'up':preprocess(librosa.load(os.path.join(args.ref,'up.wav'), sr=RATE)[0]),
        'down':preprocess(librosa.load(os.path.join(args.ref,'down.wav'), sr=RATE)[0]),
        'left':preprocess(librosa.load(os.path.join(args.ref,'left.wav'), sr=RATE)[0]),
        'right':preprocess(librosa.load(os.path.join(args.ref,'right.wav'), sr=RATE)[0]),
        'action':preprocess(librosa.load(os.path.join(args.ref,'action.wav'), sr=RATE)[0]),
        'stop':preprocess(librosa.load(os.path.join(args.ref,'stop.wav'), sr=RATE)[0]),
        'sil':preprocess(librosa.load(os.path.join(args.ref,'sil.wav'), sr=RATE)[0])
        }

print('Loading model')
model = SiameseNet(mode='inference', weights_path=args.model_location.format(args.epoch), refs_dict=refs, device=args.device)

audio_paths = glob(args.test_dir)

correct = 0

for audio_path in audio_paths:
    audio_tensor = preprocess(librosa.load(audio_path, sr=RATE)[0])

    scores = model(audio_tensor)

    predicted_action = list(refs)[np.argmax(scores)]

    if predicted_action in audio_path: correct+=1

    print('{} is predicted as {} with {} similarity'.format(audio_path, predicted_action.upper(), np.max(scores)))

print('Total accuracy =', correct/len(audio_paths))