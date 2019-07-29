import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse

parser = argparse.ArgumentParser(description='Visualize mic input')
parser.add_argument('-s', '--seconds', default = 20, type=int)
parser.add_argument('-w', action='store_true')

args = parser.parse_args()

FORMAT = pyaudio.paInt16
yrange = [-1, 1]
dtype = '<i2' 
CHANNELS = 1
RATE = 44100
CHUNK = 1024 * 4
RECORD_SECONDS = args.seconds
if args.w:
	WAVE_OUTPUT_FILENAME = "file.wav"
previous = np.zeros((CHUNK, 1))
plt.figure(figsize=(15,6))

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
					rate=RATE, input=True,
					frames_per_buffer=CHUNK)
print ("recording...")

if args.w:
	frames = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	plt.clf()
	ax1 = plt.subplot(211)
	ax2 = plt.subplot(212)
	
	data = stream.read(CHUNK)

	if args.w:
		frames.append(data)

	data_int = np.frombuffer(data, dtype=dtype).reshape(-1, CHANNELS) / (2**15)
	
	full = np.concatenate((previous, data_int))
	mfcc = librosa.feature.mfcc(y=np.squeeze(full), sr=RATE)

	plt.subplot(211)
	ax1.set_ylim(yrange)
	plt.plot(full)

	plt.subplot(212)
	librosa.display.specshow(mfcc, x_axis='time')
	plt.colorbar()

	plt.pause(0.01)
	
	previous = data_int


print ("finished recording")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

if args.w:
	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(audio.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()


