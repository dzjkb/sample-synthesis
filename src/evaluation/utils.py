import os
from os.path import dirname

import matplotlib.pyplot as plt
import librosa.display
import soundfile


def save_wav(path, x, sr):
    os.makedirs(dirname(path), exist_ok=True)
    soundfile.write(path, x, sr) 


def draw_wave(x, sr, figsize=(14, 5)):
    plt.figure(figsize=figsize)
    librosa.display.waveplot(x, sr=sr)


def draw_spect(x, sr, figsize=(14, 5)):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
