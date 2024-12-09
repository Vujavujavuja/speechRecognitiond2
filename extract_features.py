import librosa
import numpy as np

def extract_mfcc(path, n_mfcc=13):
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc