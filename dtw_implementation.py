import os
import numpy as np
import pickle
from extract_features import extract_mfcc, compute_averaged_mfcc


def dtw_dist(sound1, sound2):
    if sound1 is None or sound2 is None:
        raise ValueError("One of the inputs to dtw_dist is None.")
    if len(sound1.shape) < 2 or len(sound2.shape) < 2:
        raise ValueError(f"Input shapes are invalid: sound1 {sound1.shape}, sound2 {sound2.shape}")

    n, m = sound1.shape[1], sound2.shape[1]
    dtw = np.zeros((n + 1, m + 1)) + np.inf
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(sound1[:, i - 1] - sound2[:, j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1]
            )
    return dtw[n, m]


if __name__ == '__main__':
    mfccs = {}
    dataset_path = 'dataset'

    for file in os.listdir(dataset_path):
        if file.endswith('.wav'):
            word_label = file.split('-')[0]
            if word_label not in mfccs:
                mfccs[word_label] = []

            mfcc = extract_mfcc(os.path.join(dataset_path, file))
            if mfcc is not None:
                mfccs[word_label].append(mfcc)
                print(f"MFCC extracted for {file} (label: {word_label})")
            else:
                print(f"Failed to extract MFCC for {file}")

    with open('mfccs.pkl', 'wb') as f:
        pickle.dump(mfccs, f)
