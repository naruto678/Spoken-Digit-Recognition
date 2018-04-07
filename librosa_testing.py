#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:36:57 2018

@author: arnab
"""
 

import numpy as np
import librosa as lb

#y,sr=lb.load("0_jackson_19.wav")
#chroma=lb.feature.mfcc(y,sr,n_mfcc=40)
#lb.display.waveplot(y,sr)
#chroma_mean=np.mean(chroma,axis=1)
#chroma_std=np.std(chroma,axis=1)
#chroma=(chroma-chroma_mean)/chroma_std

def vectorize_sequences(sequences, dimension=10):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        
        results[i,sequence] = 1
    return results        


def make_data(str):

    str=["{}_{}_{}.wav".format(i,str,j) for i in range(10) for j in range(50)]
    x_train=np.zeros((550,28,28))
    y_train=np.zeros((550,1))
    for i in range(len(str)):
        y,sr=lb.load(str[i])
        chroma=lb.feature.mfcc(y,sr)
        chroma=(chroma-np.mean(chroma,axis=0))/np.std(chroma)
        
        y_train[i][0]=int(str[i][0])
        chroma=chroma.flatten()
        if len(chroma)<784:
            chroma=np.concatenate((chroma,np.zeros(784-len(chroma))))
            assert(len(chroma)==784)
        chroma=chroma[:784].reshape((28,28))
        
        
        x_train[i,:,:]=chroma
    return x_train,y_train


jackson_data,jackson_label=make_data("jackson")
nicolas_data,nicolas_label=make_data("nicolas")
theo_data,theo_label=make_data("theo")
## made the data now the keras model

def make_model():
    from keras import models
    from keras import layers
    model=models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(layers.AveragePooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.AveragePooling2D((1,1)))
    model.add(layers.Dropout(rate=0.4))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


from sklearn.model_selection import train_test_split

x_train=np.concatenate((jackson_data,nicolas_data,theo_data),axis=0)
x_train=x_train.reshape((1650,28,28,1))    
y_train=np.concatenate((jackson_label,nicolas_label,theo_label),axis=0)
 partial_x_train,val_x_train,partial_y_train,val_y_train=train_test_split(x_train,y_train,test_size=0.12,random_state=42)
#

val_y_train=vectorize_sequences(val_y_train.astype('int')).astype('float32')

partial_y_train=vectorize_sequences(partial_y_train.astype('int')).astype('float32')





speech_model=make_model()

#val_x_train=np.concatenate((jackson_data[500:],nicolas_data[500:],theo_data[500:]),axis=0).reshape((150,28,28,1))
#val_y_train=np.concatenate((jackson_label[500:],nicolas_label[500:],theo_label[500:]),axis=0).reshape((150,1))


pred=speech_model.fit(partial_x_train,partial_y_train,epochs=25,batch_size=50,validation_data=(val_x_train,val_y_train))




speech_model.save('speech_model',speech_model)
speech_dict=pred.history
np.save('speech_loss',speech_dict['loss'])
np.save('speech_acc',speech_dict['acc'])
np.save('speech_val_acc',speech_dict['val_acc'])
np.save('speech_val_loss',speech_dict['val_loss'])


#### PLotting the results
import matplotlib.pyplot as plt
fig, ax = plt.subplots( nrows=1, ncols=1 ) 
plt.plot(range(25),pred.history['acc'],'b',label='training_accuracy')
plt.plot(range(25),pred.history['val_acc'],'r',label='validation_accuracy')
plt.legend()
plt.show()

plt.savefig('training and validation.png')


fig.savefig('image_1.png')  

    