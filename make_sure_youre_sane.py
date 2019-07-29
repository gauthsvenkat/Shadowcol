import numpy as np
import wave
import librosa
import struct

FILE_NAME = "file.wav"

floating = librosa.core.load(FILE_NAME, 44100)

wf = wave.open(FILE_NAME, 'rb')

(nchannels, sampwidth, framerate, nframes, comptype, compname) = wf.getparams()
print('- input file config -');
print('nchannels', nchannels);
print('sampwidth', sampwidth);
print('framerate', framerate);
print('nframes', nframes);
print('comptype', comptype);
print('compname', compname);


frames = wf.readframes(nframes * nchannels)
# h = short (signed 16 bit)
out = struct.unpack_from ("%dh" % nframes * nchannels, frames)
mono = np.array(out)

# 2 ^ 16 = 65536 -> [-32768, 32767]
# librosa simply scale by dividing each entry by 32768
scaled = mono / 32768

print('\n- conversion from PCM to floating point time series -');
print('librosa', np.min(floating[0]), np.max(floating[0]))
print('PCM', np.min(mono), np.max(mono))
print('scaled PCM', np.min(scaled), np.max(scaled))
