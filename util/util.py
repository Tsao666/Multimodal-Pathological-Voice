import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

dataPath = "/outerDisk1/rawDatasets/multimodalPathologicalVoice/Training Dataset/"
publicTestPath = '/outerDisk1/rawDatasets/multimodalPathologicalVoice/Public Testing Dataset'
privateTestPath = '/outerDisk1/rawDatasets/multimodalPathologicalVoice/Private Testing Dataset'
sampleRate = 44100
featureCols = ['Sex', 'Age', 'Narrow pitch range', 'Decreased volume',
               'Fatigue', 'Dryness', 'Lumping', 'heartburn',
               'Choking', 'Eye dryness', 'PND', 'Smoking', 'PPD', 'Drinking',
               'frequency', 'Diurnal pattern', 'Onset of dysphonia ', 'Noise at work',
               'Occupational vocal demand', 'Diabetes', 'Hypertension', 'CAD',
               'Head and Neck Cancer', 'Head injury', 'CVA',
               'Voice handicap index - 10']


def fillBatch(df, batchSize):
    dataSize = len(df)
    fillSize = batchSize - dataSize % batchSize
    return pd.concat([df, df.sample(n=fillSize)], ignore_index=True)
    

def getDsFromDf(df, batchSize=32, returnNp=False, testMode=False):
    data = dict()
    data['audio'] = np.stack(df['beatsFeature'])
    data['structured'] = np.stack(df[featureCols].apply(lambda x: np.hstack(x), axis=1))
    if testMode:
        return data
    label = np.stack(df['label'])
    if returnNp:
        return data, label
    ds = tf.data.Dataset.from_tensor_slices((data, label))
    ds = ds.shuffle(len(df), reshuffle_each_iteration=True)
    ds = ds.batch(batchSize, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
    

def showHist(hist):
    print(hist.columns)

    for col in hist.columns:
        plt.plot(hist[col], label=col)
        
    plt.title("training history")
    plt.ylabel("value")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()