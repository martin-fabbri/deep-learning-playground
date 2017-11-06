import IPython.display as ipd
import numpy as np
import scipy.special
import librosa
import librosa.display
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import random

data, sampling_rate = librosa.load('./data/train/2022.wav')

plt.figure(figsize=(12, 4))
# librosa.display.waveplot(data, sr=sampling_rate)
librosa.display.waveplot(data, sr=sampling_rate)

data_dir = "C:\\Users\\martin\\workspace\\github\\deep-learning-playground\\urband-sound"



plt.show()