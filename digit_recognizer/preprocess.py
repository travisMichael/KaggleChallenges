import pandas as pd

trainDf = pd.read_csv("../datasets/raw/train.csv")

# Normalize the data on a 0-1 scale
trainDf.loc[:, trainDf.columns != 'label'] *= 1/255

trainDf.to_csv("../datasets/preprocessed/train_preprocessed.csv", index=False)