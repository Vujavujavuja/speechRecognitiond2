import plotly.graph_objects as go
import numpy as np
import os

from extract_features import extract_mfcc


def plot_mfcc(mfcc, title="MFCC Features"):
    fig = go.Figure(data=go.Heatmap(
        z=mfcc,
        x=np.arange(mfcc.shape[1]),
        y=np.arange(mfcc.shape[0]),
        colorscale='Viridis'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time Frames",
        yaxis_title="MFCC Coefficients"
    )
    fig.show()


if __name__ == "__main__":
    path = 'dataset'
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                mfcc = extract_mfcc(os.path.join(root, file))
                plot_mfcc(mfcc, title=file)