from tensorflow.keras import layers, models, Model, Sequential
from keras.models import Model, load_model,Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation,BatchNormalization,Conv1D,MaxPooling1D,Add,ZeroPadding1D,AveragePooling1D,GlobalAvgPool1D
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from data_processing import n1,m1,y_train,score_r,score_c
from sklearn.model_selection import train_test_split
input_size=n1.shape[1]  #Spectral Dimension
model = Sequential()

#卷积层、池化层
model.add(Conv1D(filters=64, kernel_size=5, padding='same', input_shape=(input_size,1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu',name="cov011"))
# model.add(MaxPooling2D(pool_size=(2,2)))

#Flatten层、全连接层
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(3, activation='softmax'))

y_train=np.array(y_train)
n1= np.expand_dims(n1.astype(float), axis=2)
x_train, x_test, y_train, y_test=train_test_split(n1,y_train,train_size=0.8,random_state=24)#划分数据集
model.compile(loss='categorical_crossentropy',optimizer='adam')
y_train=np_utils.to_categorical(y_train,3)#convert to one-hot
model.fit(x_train,y_train,batch_size=128,epochs=600)
model.save('Lenet.h5')#save model
