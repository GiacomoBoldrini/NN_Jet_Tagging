import math as mt
import numpy as np
import pandas as pd
import ROOT
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
import keras
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import plot_model
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve, auc
from Utils.generic_utils import mkdir_p

def target_split(dataset, target):
    y = dataset[:,target]
    X = np.delete(dataset, target, 1)
    
    return X, y


save_path = "./FFNN"
mkdir_p(save_path)

quark_var = np.load("./data_general/quark_var.npy")
gluon_var = np.load("./data_general/gluon_var.npy")

#shape = min(quark_var.shape[0], gluon_var.shape[0])
shape = 16000

quark_var = quark_var[:shape]
gluon_var = gluon_var[:shape]
dataset_q = np.c_[quark_var, (np.zeros(quark_var.shape[0]))]
dataset_g = np.c_[gluon_var, (np.ones(gluon_var.shape[0]))]

print(dataset_q.shape)
print(dataset_g.shape)

final_dataset = np.concatenate((dataset_q, dataset_g), axis = 0)
np.random.shuffle(final_dataset)

X, y = target_split(final_dataset, -1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

lb = LabelBinarizer()
y = lb.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(200, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=y_train.shape[1], activation='sigmoid'))

early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=8, verbose=1, mode='auto', baseline=None)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

tf.keras.utils.plot_model(model, to_file=save_path+'/model_FF.png')

history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test), callbacks= [early_stop])

print(">>>Plotting metrics")

fig = plt.figure(figsize=(12,7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train','Test'])
plt.savefig(save_path+"/FF_loss.pdf")

fig = plt.figure(figsize=(12,7))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.legend(['train', 'test'])
plt.savefig(save_path+"/FF_acc.pdf")

y_pred_proba = model.predict(x_test)
plt.figure(figsize=(12,7))
def plot_roc_curve(y_test, pred):
     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b',
             label='0 pred power', alpha=.8)
     fp , tp, th = roc_curve(y_test, pred)
     roc = roc_auc_score(y_test, pred)
     plt.plot(fp, tp, 'r', label='ROC binary categorizzation (AUC = %0.3f)' %(roc))
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.title('ROC NN')
     plt.legend(loc="lower right")


plt.figure(figsize=(13,8))
plot_roc_curve(y_test, y_pred_proba)
plt.savefig(save_path+"/FF_Roc.pdf")

y_pred_proba = model.predict(x_test)
y_pred_sig = []
y_pred_bkg = []
for i in range(0,len(y_pred_proba)):
    if(y_test[i] == 1):
        y_pred_bkg.append(y_pred_proba[i][0])
    else:
        y_pred_sig.append(y_pred_proba[i][0])

plt.figure(figsize=(20,10))
n, bins, _ = plt.hist(y_pred_sig, 100, histtype='step', fill=False, linewidth=2, label = "quarks")
n1, bins1, _ = plt.hist(y_pred_bkg, 100, histtype='step', fill=False, linewidth=2, label = "gluons")
#plt.yscale('log', nonposy='clip')
plt.legend(loc = "upper center",  borderpad=1, fontsize=25)
plt.xlabel("Prediction Value", size=15)
plt.ylabel("Counts", size=15)
plt.savefig(save_path+"/FF_pred_dist.pdf")

