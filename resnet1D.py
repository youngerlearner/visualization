import keras
from keras.models import Model, load_model,Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation,BatchNormalization,Conv1D,MaxPooling1D,Add,ZeroPadding1D,AveragePooling1D,GlobalAvgPool1D
from keras.initializers import glorot_uniform
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from data_processing import n1,m1,y_train,score_r,score_c
from sklearn.model_selection import train_test_split


#Build Convolutional Blocks
def cov_block(X,stage,block):
    #Naming Convolutional Blocks
    cov_name="res"+str(stage)+block+"branch"
    X_TEMP=X
    X=Conv1D(64,3,activation="relu",name=cov_name+"2a",padding='same',strides=1,kernel_initializer=glorot_uniform(seed=0))(X)
    X=Conv1D(64,1,activation="relu",name=cov_name+"2b",padding="same",kernel_initializer=glorot_uniform(seed=0))(X)
    X=Conv1D(64,3,name=cov_name+"2c",padding='same',kernel_initializer=glorot_uniform(seed=0))(X)
    X_TEMP=Conv1D(64,1,name=cov_name+"1",padding='same',strides=1,kernel_initializer=glorot_uniform(seed=0))(X_TEMP)
    X=Add()([X,X_TEMP])
    X=Activation("relu")(X)
    return X

#Constructing a ResNet Network Structure Using Convolutional Blocks
def resnet(flag):   #flag==1 for regression，flag==0 for classfication
    X_input=Input(shape=(input_size,1))
    X=Conv1D(64,3,activation="relu",padding="same",name="cov1")(X_input)

    X=cov_block(X,stage=2,block="a")
    X=cov_block(X,stage=3,block="b")
    X=cov_block(X,stage=4,block="c")
    X=cov_block(X,stage=5,block="d")

    X=Conv1D(128,3,activation="relu",padding="same",name="cov011")(X)
    X=Flatten(name='fla')(X)

    if flag==0:#classfication
        X=Dense(units=classnum,activation="softmax",name="fc")(X)
    else:#regression
        X=Dense(units=1,name="fc")(X)

    model=Model(inputs=X_input,outputs=X,name="resnet")
    return model

def model_type(x_train,y_train,x_test,flag):
    model=resnet(flag)
    #Model Compilation
    if flag==0:#classfication
        y_train=np_utils.to_categorical(y_train,classnum)#convert to one-hot
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam")
    else:#regression
        model.compile(
            loss="mse",
            optimizer="adam")
    train=model.fit(x_train,y_train,batch_size=128,epochs=300)
    model.save('model.h5')#save model
    prediction=model.predict(x_test)

    if flag==0:#Converting Hot Code to Classification Results
        result=np.zeros(np.array(prediction).shape[0])
        for i in range(np.array(prediction).shape[0]):
            result[i]=np.argmax(prediction[i])
    else:
        result=prediction
    return train,result


if __name__ == '__main__':
    input_size=n1.shape[1]  #Spectral Dimension
    classnum=int(np.max(y_train))+1 #Number of Nodes in the Last Layer of the Network

    y_train=np.array(y_train)
    n1= np.expand_dims(n1.astype(float), axis=2)
    x_train, x_test, y_train, y_test=train_test_split(n1,y_train,train_size=0.8,random_state=24)#划分数据集
    train,result=model_type(x_train,y_train,x_test,0)#0 for classfication，1 for regression

    # score_c(y_test,result)#accuracy
    score_r(y_test,result)#mse
