from pyKey import press
import librosa
import numpy as np
import torch
import pyaudio
from time import time
import os
import argparse
from utils import SiameseNet

parser = argparse.ArgumentParser(description='Live test SiameseNet')
parser.add_argument('--model_location', '-l', type=str, default='model/model-epoch-{}.pth')
parser.add_argument('--epoch', '-e', type=int, default=None)
parser.add_argument('--device', '-d', type=str, default=None)
parser.add_argument('--ref', '-r', type=str, default='references/')
parser.add_argument('--verbose', '-v', action='store_true')
args = parser.parse_args()
if not args.device:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4000

def preprocess(audio=None):
    audio_trimmed = librosa.effects.trim(audio, top_db=7)[0]
    audio_center = librosa.util.pad_center(audio_trimmed[:4000], 4000)
    audio_mfcc = librosa.feature.mfcc(y=audio_center, sr=RATE)
    audio_tensor = torch.tensor(audio_mfcc[None,None])
    
    return audio_tensor.to(device=args.device)

refs = {
        'up':preprocess(librosa.load(os.path.join(args.ref,'up.wav'), sr=RATE)[0]),
        'down':preprocess(librosa.load(os.path.join(args.ref,'down.wav'), sr=RATE)[0]),
        'sil':preprocess(librosa.load(os.path.join(args.ref,'sil.wav'), sr=RATE)[0]),
        'quit':preprocess(librosa.load(os.path.join(args.ref,'quit.wav'), sr=RATE)[0])
        }

print('Loading model')
model = SiameseNet(mode='inference', weights_path=args.model_location.format(args.epoch), refs_dict=refs, device=args.device)

previous = np.zeros((CHUNK, 1))

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

while True:
    start = time()
    
    data = stream.read(CHUNK)
    data_int = np.frombuffer(data, dtype='<i2').reshape(-1, CHANNELS) / (2**15)
    data_tensor = preprocess(np.squeeze(data_int))
    
    scores = model(data_tensor)

    if np.argmax(scores) == 0:
        press('UP')
    elif np.argmax(scores) == 1:
    	press('DOWN')
    elif np.argmax(scores) == 3:
        break

    if args.verbose:
        print('Up : ',scores[0],' Down : ', scores[1], ' Silence : ', scores[2], ' time : ', time()-start)
