from glob import glob
import librosa

files = glob('train/*.wav')

for file in files:
	y, sr = librosa.load(file, sr=16000)
	print(len(y[:6400]), sr)