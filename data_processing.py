import numpy as np
import matplotlib.pyplot as plt
import peakutils
from read_data import read_all,read_simple,draw
from collections import Counter

from sklearn import linear_model
from sklearn.metrics import r2_score

plt.rcParams['font.sans-serif']=['SimHei'] #Using Chinese Labels for Normal Display
plt.rcParams['axes.unicode_minus']=False #Using Negative Signs for Normal Display

# Subtracting the Background
def base_del(data):
    intensity_r=data
    intensity_m=np.zeros(data.shape)
    for i in range(data.shape[0]):
        base=peakutils.baseline(intensity_r[i], deg=8, max_it=5000, tol=1e-3)  #Background Fitting
        intensity_m[i]=intensity_r[i]-base   #Subtracting the Background
    return intensity_m

#Peak Normalization
def normalize_peer(data,wave_start,wave_end):
    intensity_r=data[:,wave_start:wave_end]
    intensity_m=data
    for i in range(data.shape[0]):
        intensity_m[i]=intensity_m[i]/max(intensity_r[i])
    return intensity_m

#Maximum peak normalization
def normalize_max(data,wave_start,wave_end):
    intensity_m=data
    for i in range(data.shape[0]):
        intensity_m[i]=intensity_m[i]/max(intensity_m[i])
    return intensity_m

#write file
def write(n1,m1,name):
    import numpy as np
    for i in range(n1.shape[0]):
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=6)   #Setting the Precision
        np.savetxt(name[i], np.column_stack((m1[i],n1[i])), fmt='%.10f')   #Rounding to 10 Decimal Places

"""
    Information Entropy
    n1:dataset
    start:The Starting Index of the Target Class in the Dataset
    end:The Ending Index of the Target Class in the Dataset
    rate:Expansion Factor
    y_train:label
"""
def ent(n1,start,end,rate,y_train):
    ent=[]#Information Entropy of the Individual Features
    for i in range(np.array(n1).shape[1]):
        #Region of the Target Class Spectrum
        max_value=np.max(n1[start:end,i])
        min_value=np.min(n1[start:end,i])
        #Region of Expansion of the Target Class
        ex_value=(max_value-min_value)*rate#Expansion Factor
        max_value=max_value+ex_value
        min_value=min_value-ex_value
        data=[]#Indexing of Samples in the Region
        for j in range(np.array(n1).shape[0]):
            if n1[j][i] <=max_value and n1[j][i] >=min_value:
                data.append(y_train[j])

        b=len(data)#Total Number of Samples
        c=len(set(y_train))#Total Number of Classes
        a=Counter(data)#Storage of Categories and Corresponding Sample Numbers
        entropy=0
        max_entropy=np.log2(c)#Maximum Cross Entropy for C-Class Classification
        class_num=len(dict(a))#Number of Categories in the Overlapping Region
        for (k,v) in dict(a).items():
            p=v/b
            if p==0:
                logp=0
            else:
                logp=np.log2(p)
            entropy-=p*logp
        entropy*=class_num/c
        ent.append(1-entropy/max_entropy)
    return np.array(ent)


def linearity(n1,y_train):
    cof=[]
    inter=[]
    r2=[]
    for i in range(np.array(n1).shape[1]):
        reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        reg.fit(np.expand_dims(y_train, axis=1),np.expand_dims(n1[:,i].astype(float), axis=1))
        y_pred=y_train*reg.coef_[0][0]+reg.intercept_[0]
        inter.append(reg.intercept_[0])
        cof.append(reg.coef_[0][0])
        r2.append(r2_score(n1[:,i], y_pred))#r2
        r2.append(1/(1-r2_score(n1[:,i], y_pred)))#weighted_R2
    return r2,cof,inter


#Mean Squared Error for Regression
def score_r(y_test,result):
    sum=0
    for i in range(np.array(y_test).shape[0]):
        sum+=(y_test[i]-result[i])*(y_test[i]-result[i])
    score=np.sqrt((sum/np.array(y_test).shape[0]))
    print(score)
#Classification Accuracy
def score_c(y_test,result):
    sum=0
    for i in range(np.array(y_test).shape[0]):
        if y_test[i]==result[i]:
            sum+=1
    score=sum/np.array(y_test).shape[0]
    print(score)


s1='347'  #file name
n1,m1,name1=read_all(s1,0,2301)
y_train=np.zeros((n1.shape[0]))
#label
m=0
for i in range(n1.shape[0]):
    if i%15==0:
        m+=1
    y_train[i]=m-1

n1=normalize_peer(n1,1500,1520)#Resaling
print(y_train)

if __name__=='__main__':
    draw(n1,m1[0],name1)
