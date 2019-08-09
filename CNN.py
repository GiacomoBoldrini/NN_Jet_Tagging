import ROOT
import numpy as np
import matplotlib.pyplot as plt
import random

from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from keras.layers import MaxPooling2D, BatchNormalization, Activation
from keras.utils import plot_model
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
import tensorflow as tf 

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from Utils.generic_utils import mkdir_p

def Creating_train_test(X, y, size = 0.2):
    
    X = list(X)
    y = list(y)
    X_test = []
    y_test = []
    
    test_size = int(size*len(X))
    counting = 0
    while(counting != test_size):
        try:
            r = random.randint(0, len(X))
            x_i = X.pop(r)
            y_i = y.pop(r)
            X_test.append(x_i)
            y_test.append(y_i)
            
            counting += 1
        except:
            continue
    
    X_train = np.array(X)
    y_train = np.array(y)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_train, X_test, y_train, y_test

print(">>>Creating saving folder...")
save_path = "./CNN_outputs"
mkdir_p(save_path)

print(">>> Loading Arrays")

quarks_im = np.load("./data_general/quarks_3d_im.npy")
gluons_im = np.load("./data_general/gluons_3d_im.npy")

print(">>>Creating train test")

#shape = min(quarks_im.shape[0], gluons_im.shape[0])
shape = 15000
quarks_im = quarks_im[:shape]
gluons_im = gluons_im[:shape]
y_gluons = [1]*gluons_im.shape[0]
y_quarks = [0]*quarks_im.shape[0]

y = y_gluons + y_quarks
y = np.array(y)

X = np.concatenate((gluons_im, quarks_im), axis=0)

print(X.shape, y.shape)

x_train, x_test, y_train, y_test = Creating_train_test(X, y, size = 0.1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#reshaping for keras input
x_train = x_train.reshape((x_train.shape[0], x_train.shape[2], x_train.shape[3], 3))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[2], x_test.shape[3], 3))

print(x_test.shape, x_train.shape)
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
dropoutRate = 0.1

image_shape = (img_rows, img_cols, 3)
####
inputImage = Input(shape=(image_shape))
#x = Conv2D(5, kernel_size=(5,5), data_format="channels_last", strides=(1, 1), padding="same")(inputImage)
x = Conv2D(5, kernel_size=(5,5), data_format="channels_last", strides=(1, 1), padding="same")(inputImage)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D( pool_size = (5,5))(x)
x = Dropout(dropoutRate)(x)
#
x = Conv2D(3, kernel_size=(3,3), data_format="channels_last", strides=(1, 1), padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D( pool_size = (3,3))(x)
x = Dropout(dropoutRate)(x)
#Here the output as a 1xN vector of convolution
x = Flatten()(x)
#Here enters DNN
x = Dense(100, activation='relu')(x)
x = Dropout(dropoutRate)(x)
x = Dense(100, activation='relu')(x)
x = Dropout(dropoutRate)(x)
x = Dense(50, activation='relu')(x)
x = Dropout(dropoutRate)(x)
#
output = Dense(1, activation='sigmoid')(x)
####
model = Model(inputs=inputImage, outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
batch_size = 64
n_epochs = 50

history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose = 1,
                validation_data=(x_test, y_test), shuffle = True,
                callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),
                TerminateOnNaN()])

print(">>>Plotting metrics")

plt.figure(figsize=(15,10))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy', size=18)
plt.ylabel('accuracy', size=18)
plt.xlabel('epoch', size=18)
plt.legend(['train', 'test'], loc='lower right', prop={'size': 16})
plt.savefig(save_path+"/CNN_acc.pdf")

plt.figure(figsize=(15,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss', size=18)
plt.ylabel('Loss value', size=18)
plt.xlabel('epoch', size=18)
plt.legend(['train', 'test'], loc='lower right', prop={'size': 16})
plt.savefig(save_path+"/CNN_loss.pdf")

y_pred = model.predict(x_test)
y_pred = [round(i[0]) for i in y_pred]

from sklearn.metrics import roc_auc_score, roc_curve, auc

y_pred = model.predict(x_test)
def plot_roc_curve(y_test, pred):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b',
    label='0 pred power', alpha=.8)
    fp , tp, th = roc_curve(y_test, pred)
    roc = roc_auc_score(y_test, pred)
    plt.plot(fp, tp, 'r', label='ROC binary categorizzation (AUC = %0.3f)' %(roc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right", prop={'size': 16})
    

plt.figure(figsize=(13,8))
plot_roc_curve(y_test, y_pred)
plt.savefig(save_path+"/CNN_ROC.pdf")

y_pred_proba = model.predict(x_test)
y_pred_sig = []
y_pred_bkg = []
for i in range(0,len(y_pred_proba)):
    if(y_test[i] == 1):
        y_pred_bkg.append(y_pred_proba[i][0])
    else:
         y_pred_sig.append(y_pred_proba[i][0])

plt.figure(figsize=(20,10))
n, bins, _ = plt.hist(y_pred_sig, 100, histtype='step', fill=False, linewidth=2, label = "Quarks")
n1, bins1, _ = plt.hist(y_pred_bkg, 100, histtype='step', fill=False, linewidth=2, label = "Gluons")
#plt.yscale('log', nonposy='clip')
plt.legend(loc = "upper center",  borderpad=1, fontsize=25)
plt.xlabel("Prediction Value", size=15)
plt.ylabel("Counts", size=15)
plt.savefig(save_path+"/CNN_prob_dist.pdf")


