from tensorflow.keras import layers, models, Model, Sequential
from keras.models import Model, load_model,Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation,BatchNormalization,Conv1D,MaxPooling1D,Add,ZeroPadding1D,AveragePooling1D,GlobalAvgPool1D
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from data_processing import n1,m1,y_train,score_r,score_c
from sklearn.model_selection import train_test_split
input_size=n1.shape[1]  #Spectral Dimension
def AlexNet(input_shape=(input_size,1),output_shape=3):
    # AlexNet
    model = Sequential()
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 所建模型后输出为48特征层
    model.add(Conv1D(
                     filters=48,
                     kernel_size=11,
                     strides=1,
                     padding='same',
                     input_shape=input_shape,
                     activation='relu'
                    )
             )
    # model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    # model.add(MaxPooling1D(
    #                        pool_size=(3,3),
    #                        strides=(2,2),
    #                        padding='valid'
    #                       )
    #          )
    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv1D(
            filters=128,
            kernel_size=5,
            strides=1,
            padding='same',
            activation='relu'
        )
    )

    # model.add(BatchNormalization())
    model.add(
        Conv1D(
            filters=64,
            kernel_size=5,
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv1D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'
        )
    )

    model.add(
        Conv1D(
            filters=128,
            kernel_size=5,
            strides=1,
            padding='same',
            activation='relu',
            name="cov011"
        )
    )

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    return model




y_train=np.array(y_train)
n1= np.expand_dims(n1.astype(float), axis=2)
x_train, x_test, y_train, y_test=train_test_split(n1,y_train,train_size=0.8,random_state=24)#划分数据集
model = AlexNet()
model.compile(loss='categorical_crossentropy',optimizer='adam')
y_train=np_utils.to_categorical(y_train,3)#convert to one-hot
model.fit(x_train,y_train,batch_size=128,epochs=300)
model.save('Alex.h5')#save model
