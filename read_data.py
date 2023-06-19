import os
import numpy as np
"""
    Function:Read text 
    filename:Text path
    wave_start:Initial wavelength of spectrum
    wave_end:End wavelength of spectrum
"""
def read(filename,wave_start,wave_end):
    file = open(filename)
    data_lines = file.readlines()
    file.close()
    orign_keys = []
    orign_values = []
    for data_line in data_lines[wave_start:wave_end]:#Truncation
        pair = data_line.split()
        key = float(pair[0]) #Save Wavelength
        value = float(pair[1])  #Save Intensity
        orign_keys.append(key)
        orign_values.append(value)
    return orign_keys, orign_values

"""
    Bulk Reading of Text Files in a Single Folder
    filenam:File Name
    wave_start:Starting Wavelength
    wave_end:Ending Wavelength
"""
def read_simple(file_name,wave_start,wave_end):
    finame=os.listdir(file_name)
    num=0
    n1=[]
    m1=[]
    name=[]
    for i in finame:
        file1=os.path.join(file_name,i) #Path to Generate All Texts
        print(file1,num)
        name.append(i)
        m,n=read(file1,wave_start,wave_end)
        n1.append(n)
        m1.append(m)
        num=num+1
    return np.array(n1),np.array(m1),name

"""
    Bulk Read Texts From Double Folders
    filenam:File Name
    wave_start:Starting Wavelength
    wave_end:Ending Wavelength
"""
def read_all(file_name,wave_start,wave_end):
    finame=os.listdir(file_name)
    num=0
    n1=[]
    m1=[]
    name=[]
    for i in finame:
        filename=os.listdir(os.path.join(file_name,i))#Paths of All Folders
        for j in filename:
            file1=os.path.join(file_name,i,j) #Paths of All Texts
            name.append(j) #Save Spectrum Names
            print(file1,num)
            m,n=read(file1,wave_start,wave_end)
            n1.append(n)
            m1.append(m)
            num=num+1
    return np.array(n1),np.array(m1),name
"""
    Drawing Graphs
    data:Spectral Matrix with Each Row Representing a Spectrum
    wave_start:Starting Wavelength
    wave_end:Ending Wavelength
    finame:Spectral Labels
"""
def draw(data,m1,finame):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['SimHei'] #Using Chinese Labels for Normal Display
    plt.rcParams['axes.unicode_minus']=False #Using Negative Signs for Normal Display
    plt.figure(1)
    plt.xlabel("nm")
    plt.ylabel("abs")
    for i in range(data.shape[0]):
        plt.plot(m1,data[i],linewidth=0.4,label=finame[i].replace(".txt",''))
        plt.legend()
    plt.show()
