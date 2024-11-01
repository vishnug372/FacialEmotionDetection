
import cv2
import dlib
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

import urllib.request

from sklearn import metrics
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from tqdm import tqdm,tqdm_pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import re
import keras

from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers

warnings.filterwarnings("ignore")
def model_to_string(model):
    import re
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    sms = "\n".join(stringlist)
    sms = re.sub('_\d\d\d','', sms)
    sms = re.sub('_\d\d','', sms)
    sms = re.sub('_\d','', sms)
    return sms


!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Emotion%20Detection/fer2013_5.csv"

!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Emotion%20Detection/shape_predictor_68_face_landmarks.dat"

!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Emotion%20Detection/pureX.npy"

!wget -q --show-progress -O ./dataX.npy "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Emotion%20Detection/dataX_edited.npy"

!wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Emotion%20Detection/dataY.npy"



def plot_confusion_matrix(y_true,y_predicted):
  cm = metrics.confusion_matrix(y_true, y_predicted)
  print ("Plotting the Confusion Matrix")
  labels = list(label_map.values())
  df_cm = pd.DataFrame(cm,index = labels,columns = labels)
  fig = plt.figure()
  res = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
  plt.yticks([0.5,1.5,2.5,3.5,4.5], labels,va='center')
  plt.title('Confusion Matrix - TestData')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

  plt.show()
  plt.close()

def plot_graphs(history, best):

  plt.figure(figsize=[10,4])
  plt.subplot(121)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy across training\n best accuracy of %.02f'%best[1])
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')

  plt.subplot(122)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss across training\n best loss of %.02f'%best[0])
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()


label_map = {"0":"ANGRY","1":"HAPPY","2":"SAD","3":"SURPRISE","4":"NEUTRAL"}


predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def get_landmarks(image):



  rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]


  landmarks = [(p.x, p.y) for p in predictor(image, rects[0]).parts()]
  return image,landmarks


def image_landmarks(image,face_landmarks):

  radius = -4
  circle_thickness = 1
  image_copy = image.copy()
  for (x, y) in face_landmarks:
    cv2.circle(image_copy, (x, y), circle_thickness, (255,0,0), radius)

  plt.imshow(image_copy, interpolation='nearest')
  plt.show()


def landmarks_edist(face_landmarks):
    e_dist = []
    for i,j  in itertools.combinations(range(68), 2):
      e_dist.append(distance.euclidean(face_landmarks[i],face_landmarks[j]))
    return e_dist

def compare_learning(mlp, lm, cnn, vgg): 
  
  plt.plot(vgg.history['val_accuracy'],)
  plt.plot(cnn.history['val_accuracy'])
  plt.plot(mlp.history['val_accuracy'],)
  plt.plot(lm.history['val_accuracy'])
  plt.ylabel('validitation accuracy')
  plt.xlabel('epoch')
  plt.legend(['cnn_transfer', 'cnn_scratch', 'mlp_pixels', 'mlp_landmarks'], bbox_to_anchor=[1,1])
  plt.xticks(range(0, epochs+1, 5), range(0, epochs+1, 5))
  plt.show()



tmp_model = Sequential()

tmp_model.add(Dense(7, input_shape=(5,), activation = 'relu')) 

tmp_model.add(Dense(7, activation = 'relu'))

tmp_model.add(Dense(4, activation = 'softmax'))

tmp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


perceptron = Sequential()


perceptron.add(Dense(1024, input_shape=(2278,), activation = 'relu',kernel_initializer='glorot_normal')) 


perceptron.add(Dense(512, activation = 'relu',kernel_initializer='glorot_normal'))


perceptron.add(Dense(5, activation = 'softmax'))


perceptron.compile(loss='categorical_crossentropy', optimizer= SGD(learning_rate=0.001), metrics=['accuracy'])


epochs = 20

batch_size = 64

test_ratio = .1

n_labels = 5

dataX_pixels = np.load('pureX.npy')
dataY_labels = np.load('dataY.npy')

y_onehot = to_categorical(dataY_labels, len(set(dataY_labels)))

pixel_scaler = StandardScaler()
pixel_scaler.fit(X_train)
X_train = pixel_scaler.transform(X_train)
X_test = pixel_scaler.transform(X_test)


mlp_model = Sequential()

mlp_model.add(Dense(1024, input_shape=(2304,), activation = 'relu',kernel_initializer='glorot_normal')) 

mlp_model.add(Dense(512, activation = 'relu',kernel_initializer='glorot_normal'))

mlp_model.add(Dense(300, activation = 'relu',kernel_initializer='glorot_normal'))


mlp_model.add(Dense(5, activation = 'softmax'))

mlp_model.compile(loss=categorical_crossentropy, optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_mlp_model.keras', verbose=1, monitor='val_accuracy', save_best_only=True)

mlp_history = mlp_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            callbacks=[checkpoint], validation_data=(X_test, y_test), shuffle=True)

y_pred_mlp = mlp_model.predict(X_test)
y_pred_mlp_classes = np.argmax(y_pred_mlp, axis=1)
y_true = np.argmax(y_test,axis=1)
plot_confusion_matrix(y_true, y_pred_mlp_classes)

dataX_lm = np.load('./dataX.npy')

X_train_lm, X_test_lm, y_train_lm, y_test_lm = train_test_split(dataX_lm, y_onehot, test_size=0.1, random_state=42)

lm_scaler = StandardScaler()
lm_scaler.fit(X_train_lm)
X_train_lm = lm_scaler.transform(X_train_lm)
X_test_lm = lm_scaler.transform(X_test_lm)

lm_model = Sequential()

lm_model.add(Dense(1024, input_shape=(2278,), activation = 'relu',kernel_initializer='glorot_normal')) 
lm_model.add(Dense(700, activation = 'relu',kernel_initializer='glorot_normal')) 


lm_model.add(Dense(512, activation = 'relu',kernel_initializer='glorot_normal'))

lm_model.add(Dense(300, activation = 'relu',kernel_initializer='glorot_normal'))

lm_model.add(Dense(5, activation = 'softmax'))

lm_model.compile(loss=categorical_crossentropy, optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_lm_model.keras', verbose=1, monitor='val_loss',save_best_only=True,  mode='auto')

lm_history = lm_model.fit(X_train_lm, y_train_lm, batch_size=batch_size, epochs=epochs,
                          verbose=1, callbacks=[checkpoint], validation_data=(X_test_lm, y_test_lm), shuffle=True)

lm_performance = lm_model.evaluate(X_test_lm, y_test_lm, batch_size=64)

plot_graphs(lm_history, lm_performance);

width, height = 48, 48

print(X_train.shape)

X_train_cnn = X_train.reshape(len(X_train),height,width)
X_test_cnn = X_test.reshape(len(X_test),height,width)

print(X_train_cnn.shape)
print(X_test_cnn.shape)

X_train_cnn = np.expand_dims(X_train_cnn,3)
X_test_cnn = np.expand_dims(X_test_cnn,3)

print(X_train_cnn.shape)


cnn_model = Sequential()
cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
cnn_model.add(Dropout(0.8))


cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation='relu'))
cnn_model.add(Dense(n_labels, activation='softmax'))

checkpoint = ModelCheckpoint('best_cnn_model.keras', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])

cnn_history = cnn_model.fit(X_train_cnn, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            callbacks=[checkpoint], validation_data=(X_test_cnn, y_test), shuffle=True)

cnn_performance = cnn_model.evaluate(X_test_cnn, y_test, batch_size=64)

plot_graphs(cnn_history, cnn_performance);
