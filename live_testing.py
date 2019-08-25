from pyKey import press, pressKey, releaseKey
import librosa
import numpy as np
import torch
import pyaudio
from time import time, sleep
import os
import argparse
from utils import SiameseNet

parser = argparse.ArgumentParser(description='Live test SiameseNet')
parser.add_argument('--model_location', '-l', type=str, default='model/model-epoch-{}.pth')
parser.add_argument('--epoch', '-e', type=int, default=None)
parser.add_argument('--device', '-d', type=str, default='cpu')
parser.add_argument('--ref', '-r', type=str, default='references/')
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

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
        'left':preprocess(librosa.load(os.path.join(args.ref,'left.wav'), sr=RATE)[0]),
        'right':preprocess(librosa.load(os.path.join(args.ref,'right.wav'), sr=RATE)[0]),
        'action':preprocess(librosa.load(os.path.join(args.ref,'action.wav'), sr=RATE)[0]),
        'stop':preprocess(librosa.load(os.path.join(args.ref,'stop.wav'), sr=RATE)[0]),
        'sil':preprocess(librosa.load(os.path.join(args.ref,'sil.wav'), sr=RATE)[0])
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
    pressed_key = None

    start = time()
    
    data = stream.read(CHUNK)
    data_int = np.frombuffer(data, dtype='<i2').reshape(-1, CHANNELS) / (2**15)
    data_tensor = preprocess(np.squeeze(data_int))
    
    scores = model(data_tensor)

    if np.argmax(scores) == 0: 
        press('UP'); sleep(0.4)

    elif np.argmax(scores) == 1:
        press('DOWN'); sleep(0.4)

    elif np.argmax(scores) == 2: #Release any key that is pressed and press left
        if pressed_key is not None: releaseKey(pressed_key)
        pressed_key = 'LEFT'
        pressKey(pressed_key); sleep(0.4)

    elif np.argmax(scores) == 3: #Release any key that is pressed and press right
        if pressed_key is not None: releaseKey(pressed_key)
        pressed_key = 'RIGHT'
        pressKey(pressed_key); sleep(0.4)

    elif np.argmax(scores) == 4: #action key
        press('LCTRL'); sleep(0.4)

    elif np.argmax(scores) == 5: #stop command that will release any key that is pressed
        if pressed_key is not None: releaseKey(pressed_key)
        pressed_key = None
        sleep(0.4)

    if args.verbose:
        print(' Up : ',scores[0], end='')
        print(' Down : ',scores[1], end='')
        print(' Left : ',scores[2], end='')
        print(' Right : ',scores[3], end='')
        print(' Action : ',scores[4], end='')
        print(' Stop : ',scores[5], end='')
