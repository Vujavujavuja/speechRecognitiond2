# Speech Recognition, part 2: isolated words with MFCC and DTW

A small isolated-word recogniser built without a neural network. It compares a new recording against reference recordings of each word using dynamic time warping, and picks the closest one.

## How it works

1. **Features** (`extract_features.py`): each recording is trimmed of silence and turned into 13 MFCCs plus their first and second derivatives, normalised per coefficient.
2. **Reference set** (`dtw_implementation.py`): features are extracted for every file in `dataset/` and saved to `mfccs.pkl`. The word label is the part of the file name before the first `-`, so `open-01.wav` is a recording of "open".
3. **Classification** (`test.py`): the test recording is compared with every reference using a from-scratch DTW implementation. The word with the smallest distance wins. If the best distance is above a threshold, or all words are about equally far, the answer is `Unknown`.

`plot_and_visualize.py` draws the MFCC heatmap for each recording.

## Running it

```bash
pip install -r requirements.txt
python load_dataset.py        # unpacks dataset*.zip into dataset/
python dtw_implementation.py  # builds mfccs.pkl
python test.py                # classifies the file named in test.py
```

The dataset is not included. Record a few `.wav` files per word, name them `word-N.wav`, and put them in `dataset/`.

Part 1, signal inspection, is in [speechRecognitiond1](https://github.com/Vujavujavuja/speechRecognitiond1).
