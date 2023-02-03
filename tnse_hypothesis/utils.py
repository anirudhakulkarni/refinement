'''
Test hypothesis by plotting TSNE plots before calibration and after calibration
'''

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_tsne(features, labels, n_comps, title):
    '''
    Plot TSNE plot
    '''
    tsne = TSNE(n_components=n_comps, random_state=0)
    tsne_obj = tsne.fit_transform(features)
    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0], 'Y': tsne_obj[:, 1], 'label': labels})
    fig = plt.figure(figsize=(8, 8))
    
