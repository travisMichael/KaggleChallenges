# https://www.kaggle.com/c/digit-recognizer
import pandas as pd
from digit_recognizer.util import *
from models.scratch.NeuralNetwork import NeuralNetwork
from collections import deque

scores_window = deque(maxlen=500)
trainDf = pd.read_csv("../datasets/preprocessed/train_preprocessed.csv")

train = trainDf.to_numpy()

X = train[:, 1:]
Y = train[:, 0:1]

# calculate number of training examples
n = X.shape[0]

STEPS = 100
ALPHA = 0.001
LABELS = 11

model = NeuralNetwork(learning_rate=ALPHA)


for i in range(STEPS):
    cost = 0.0
    s_1 = 0.0
    s_2 = 0.0
    c = 0.0
    for j in range(n):
        if j % 1000 == 0:
            print("processing", c, i, j)
            c = 0.0
        x = X[j]
        y = np.zeros(11)
        y[int(Y[j][0])] = 1

        output = model.forward(x)
        model.backward(y)

        cost += np.sum(np.square(y - output))
        c += np.sum(np.square(y - output))

    for j in range(n):
        x = X[j]
        y = int(Y[j][0])

        output = model.forward(x)
        actual = np.argmax(output)
        if actual == y:
            scores_window.append(1)
        else:
            scores_window.append(0)


    print("average for train set", np.mean(scores_window))
    print(cost, s_1, s_2)


print("Hello")