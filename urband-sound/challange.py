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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

data_dir = "C:\\Users\\martin\\workspace\\github\\deep-learning-playground\\urband-sound"

def parser(row):
    # function to load and extract features
    file_name = os.path.join(os.path.abspath(data_dir),
                             "train", str(row.ID) + ".wav")
    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type="kaiser_fast")

        # we extract mfcc feature from data
        mfccs = np.mean(
            librosa.feature.mfcc(
                y=X,
                sr=sample_rate,
                n_mfcc=40).T,
            axis=0)
    except Exception as e:
        print("Error encounter while parsing file: ", file_name)
        return None, None

    feature = mfccs
    label = row.Class

    return [feature, label]

temp = train.apply(parser, axis=1)
temp.columns = ["feature", "label"]

from sklearn.preprocessing import LabelEncoder

X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y))

