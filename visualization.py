# -*- coding: utf8 -*-
import numpy as np
import keras.backend as K
from matplotlib import pyplot as plt
import cv2
import keras
import tensorflow as tf
from data_processing import n1,m1,y_train,name1
from scipy import interpolate

n1= np.expand_dims(n1.astype(float), axis=2)#Convolutional Receiving of Three-Dimensional Data
print(np.array(n1).shape)
"""
    Visualization Functions
    X_test:Data to be visualized
    model_name:Saved Network Models
    last_conv:Names of Convolutional Layers to be Visualized
    kenel:Number of Convolutional Kernels
"""
def visualization(X_test,model_name,last_conv,kenel):
    tf.compat.v1.disable_eager_execution()
    K.clear_session()
    K.set_learning_phase(1)

    model = keras.models.load_model(model_name)
    x=X_test
    x = np.expand_dims(x,axis=0)#Expand Dimension to Enter Batch State(None,2301,1)
    pred = model.predict(x)
    class_idx = np.argmax(pred[0])#Extract Result Classes
    class_output = model.output[:,class_idx]
    #Extract the Last Layer of Convolution
    last_conv_layer = model.get_layer(last_conv)
    grads = K.gradients(class_output,last_conv_layer.output)[0]    #Calculate the Gradient of Model Output to the Last Convolution Layer Activation Output
    pooled_grads = K.mean(grads,axis=0)#Normalize the Gradient to Get the Average Gradient for Each Channel, 0 for No Averaging, 1 for Averaging.
    iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])#Establish the Relationship between Model Input, Gradient and Last Layer Convolution.
    pooled_grads_value, conv_layer_output_value = iterate([x])
    print(np.array(conv_layer_output_value).shape,np.array(pooled_grads_value).shape)#size of Feature Map and Gradient

    # Multiply the Gradient Values with the Features of the Last Layer's Channels to Represent the Importance.
    for i in range(kenel):#Number of Channels (Convolution Kernels) in the Last Layer Convolutional Layer.
        # plt.plot(m1[0],conv_layer_output_value.T[i],linestyle='-')#Check Feature Maps
        # plt.plot(m1,pooled_grads_value.T[i]/np.max(pooled_grads_value),linestyle='-')#Check gradients
        if np.max(abs(pooled_grads_value))==0:#If the Gradient is 0, Use the Weights of the Last Layer.
            t=model.get_layer('fc')#Name of the Last Layer.
            weight=t.get_weights()[0].T[class_idx]
            weight=np.reshape(weight,(int(np.array(weight).shape[0]/kenel),kenel))
            conv_layer_output_value[:,i]= weight[:,i]*conv_layer_output_value[:,i]
        else:
            conv_layer_output_value[:,i]= pooled_grads_value[:,i]*conv_layer_output_value[:,i]
    #
    heatmap = np.mean(conv_layer_output_value, axis=-1)#Calculate the Average Weights of Each Feature Across All Channels.
    fe=list(m1[0])
    '''
    #If the Network Exists Dimension Reduction Operations, Interpolation is Needed.
    x = np.linspace(1,431,np.array(heatmap).shape[0])
    tck = interpolate.splrep(x,heatmap)
    xx = np.linspace(min(x),max(x),431)
    heatmap = interpolate.splev(xx,tck,der=0)   
    '''
    heatmap /= np.max(abs(heatmap))#Normalization.
    np.savetxt('Lenet.txt',heatmap)
    plt.plot(fe,heatmap)
    # plt.show()
    return heatmap

def show_heatmeap(heatmap,X_test):#Convert the Weights into Heat Maps.
    img = np.uint8(255*X_test)
    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    img = cv2.applyColorMap(img,cv2.COLORMAP_JET)#From blue to red, the redder the more attention it gets.
    superimposed_img = cv2.addWeighted(img,0.6,heatmap,0.4,0)
    cv2.imwrite('heatmap.png',heatmap)#Weight Heat Map.
    cv2.imwrite('img.png',img)
    cv2.imwrite('Grad-cam.png',superimposed_img)#Overlay Heat Maps.



if __name__=='__main__':
    plt.xlim(750,1700)
    plt.xticks(np.arange(750,1700,100))
    plt.title("Alexnet")
    visualization(n1[0],'Lenet.h5','cov011',128)
    # visualization(n1[20],'model.h5','cov011',128)
    # visualization(n1[40],'model.h5','cov011',128)
    plt.show()
