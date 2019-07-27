import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt

FORMAT = pyaudio.paInt16

yrange = [-32768, 32767]
dtype = '<i2' 

CHANNELS = 1
RATE = 44100
CHUNK = 1024 * 4
RECORD_SECONDS = 20
WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
					rate=RATE, input=True,
					frames_per_buffer=CHUNK)
print ("recording...")
frames = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	plt.clf()
	plt.axes().set_ylim(yrange)
	data = stream.read(CHUNK)
	data_int = np.frombuffer(data, dtype=dtype).reshape(-1, CHANNELS)
	plt.plot(data_int)
	plt.pause(0.01)

plt.show()


print ("finished recording")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

