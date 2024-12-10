import librosa
import numpy as np


def extract_mfcc(path, n_mfcc=13):
    try:
        y, sr = librosa.load(path)

        y_trimmed, _ = librosa.effects.trim(y, top_db=20)  # Adjust `top_db` if necessary

        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)

        delta = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)
        mfcc_combined = np.vstack([mfcc, delta, delta_delta])

        mfcc_combined = (mfcc_combined - np.mean(mfcc_combined, axis=1, keepdims=True)) / np.std(
            mfcc_combined, axis=1, keepdims=True
        )
        return mfcc_combined

    except Exception as e:
        print(f"Error extracting mfcc for {path}: {e}")
        return None

def compute_averaged_mfcc(mfccs):
    averaged_mfccs = {}

    for word, mfcc_list in mfccs.items():
        valid_mfccs = [mfcc for mfcc in mfcc_list if mfcc is not None and len(mfcc.shape) == 2]
        if not valid_mfccs:
            print(f"No valid mfccs for word '{word}'")
            continue

        min_frames = min(mfcc.shape[1] for mfcc in valid_mfccs)
        truncated_mfccs = [mfcc[:, :min_frames] for mfcc in valid_mfccs]

        averaged_mfccs[word] = np.mean(truncated_mfccs, axis=0)

    return averaged_mfccs
