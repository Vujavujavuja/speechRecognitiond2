import os
import pickle

from extract_features import extract_mfcc
from dtw_implementation import dtw_dist


def classify(test_file, mfccs, n_mfcc=13, threshold=150):
    test_mfcc = extract_mfcc(test_file, n_mfcc=n_mfcc)
    if test_mfcc is None or len(test_mfcc.shape) != 2:
        raise ValueError(f"Test file {test_file} produced invalid MFCC shape.")

    distances = {}
    for word, ref_mfcc_list in mfccs.items():
        distances[word] = min(dtw_dist(test_mfcc, ref_mfcc) for ref_mfcc in ref_mfcc_list)

    min_distance = min(distances.values())
    prediction = min(distances, key=distances.get)

    max_diff = max(distances.values()) - min_distance

    if min_distance > threshold:
        return "Unknown", distances
    elif max_diff < 10:
        return "Unknown", distances
    return prediction, distances


if __name__ == '__main__':
    mfccs_file = 'mfccs.pkl'
    if not os.path.exists(mfccs_file):
        print("mfccs.pkl not found!")
        exit(1)

    with open(mfccs_file, 'rb') as f:
        mfccs = pickle.load(f)

    test_file = 'record_out (5).wav'
    if not os.path.exists(test_file):
        print(f"Test file {test_file} does not exist.")
    else:
        prediction, distances = classify(test_file, mfccs)
        print(f"Predicted word: {prediction}")
        print("Distances to each word:")
        for word, dist in distances.items():
            print(f"  {word}: {dist:.2f}")
