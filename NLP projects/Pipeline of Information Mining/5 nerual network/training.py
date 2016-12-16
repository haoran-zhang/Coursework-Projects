#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import numpy as np
os.chdir('/home/haoran/桌面/vector')
file_lst=os.listdir(os.getcwd())
y_list=[]
x_list=[]
for item in file_lst:
    f=open(item,'r')
    for line in f:
        temp_l=line.split()
        y_list.append(int(temp_l[0]))
        temp_x=[]
        for i in range(1,len(temp_l)):
            temp_x.append(int(temp_l[i]))
        x_list.append(temp_x)
    f.close()

#y_train=np.array(y_list)[:int(len(y_list)*0.8)]
y_text=np.array(y_list)[int(len(y_list)*0.8):]
y_train=np.array(y_list)
#x_train = np.load('/home/haoran/桌面/x_train.npc.npy')
#x_test = np.load('/home/haoran/桌面/x_test.npc.npy')
num=int(len(x_list)*0.8)
x_train=x_list[:num]
x_test=x_list[num:]

#READY TO TRAIN : y_array , x_list
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import SGD
from keras.layers import Embedding
from keras.layers import LSTM

# #MODEL 1
# model=Sequential()
# model.add(Dense(input_dim=16,output_dim=100))
# model.add(Dense(output_dim=1000))
# model.add(Dense(output_dim=1000))
# model.add(Dense(output_dim=1000))
# model.add(Dense(output_dim=10))
#
#
# model.add(Activation("softmax"))
#
# model.compile(loss="sparse_categorical_crossentropy",optimizer='SGD',metrics=['accuracy'])
# sgd=SGD(clip)
# model.fit(x_list, y_train, nb_epoch=10)

#MODEL 2
model=Sequential()
model.add(Embedding(input_dim=133049,output_dim=256))
model.add(LSTM(output_dim=128,activation='sigmoid',inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(x_list,y_train,class_weight=np.array([0.1,0.1,1,1,1,0.8,1,0.6,0.4,0.8,1,0.6,0.4,1,1,1]),nb_epoch=5)

# score,acc=model.evaluate(x_test,y_text)
# print('------------------\n',score,'     ',acc)

#MODEL 3
# model = Sequential()
# model.add(Dense(input_dim=16, output_dim=512))
# model.add(Dropout(0.5))
# model.add(Dense(512))
# model.add(Dense(output_dim=1))
# model.add(Activation("softmax"))
# sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.985)
# model.compile(loss='binary_crossentropy', optimizer=sgd)
# model.fit(x_list,y_train,nb_epoch=10)


try:
    arr = model.get_weights()
    np.save('/home/haoran/桌面/arr.npy',arr)
except:
    print("nothing")
try:
    model.save_weights('/home/haoran/桌面/arr',overwrite=True)
except:
    print("fail")