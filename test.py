import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help="path to the Universal Dependencies data directory", default="/Users/kazuyabe/Data/UD_Japanese-GSD")
parser.add_argument("-m", "--model", help="name for saving the model", default="./tmp/model.h5")
args = parser.parse_args()

from Dependency_Parser import Dependency_Parser

batch_size = 32

import pickle as p
X_tr = p.load(open("data/X_train.p", "rb"))
X_test = p.load(open("data/X_test.p", "rb"))
X_val = p.load(open("data/X_val.p", "rb"))
y_tr = p.load(open("data/y_train.p", "rb"))
y_test = p.load(open("data/y_test.p", "rb"))
y_val = p.load(open("data/y_val.p", "rb"))
tagToIndex = p.load(open("data/tagToIndex.p", "rb"))


# import numpy as np
# X_tr, X_val = np.array(X_tr[:121*batch_size]), np.array(X_tr[-13*batch_size:])
# y_tr, y_val = y_tr[:121*batch_size], y_tr[-13*batch_size:]
# y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
# y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)

import numpy as np
# X_test = np.array(X_test)
# y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

parser = Dependency_Parser(epochs=10)
hist = parser.fit(data=True, X_train=X_tr, y_train=y_tr, validation_data=(X_val, y_val))
p.dump(hist.history, open("./tmp/history.p", "wb"))
score = parser.evaluate(X_test, y_test)
print(score)