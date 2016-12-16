#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers import Embedding
from keras.layers import LSTM

from keras.optimizers import SGD

#MODEL 1
# model=Sequential()
# model.add(Dense(input_dim=16,output_dim=100))
# model.add(Dense(output_dim=1000))
# model.add(Dense(output_dim=1000))
# model.add(Dense(output_dim=1000))
# model.add(Dense(output_dim=10))
# model.add(Activation("softmax"))
#
# model.compile(loss="sparse_categorical_crossentropy",optimizer='SGD',metrics=['accuracy'])
#model.fit(x_list, y_train, nb_epoch=10)

# #MODEL 2
model=Sequential()
model.add(Embedding(input_dim=133049,output_dim=256))
model.add(LSTM(output_dim=128,activation='sigmoid',inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# model.fit(x_list,y_train,nb_epoch=3)
#
# #MODEL 3
# model = Sequential()
# model.add(Dense(input_dim=16, output_dim=512))
# model.add(Dropout(0.5))
# model.add(Dense(512))
# model.add(Dense(output_dim=1))
# model.add(Activation("softmax"))
# sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.1)
# model.compile(loss='binary_crossentropy', optimizer=sgd)


model.load_weights('/home/haoran/桌面/arr')

os.chdir('/home/haoran/桌面/vector_test')
to_dir='/home/haoran/桌面/predict_result/'
file_lst=os.listdir(os.getcwd())
for item in file_lst:
    x_list=[]
    f=open(item,'r')
    for line in f:
        temp_x=[]
        temp_x=line.split()
        x_list.append(temp_x)
    f.close()
    ans=model.predict(x_list)

    f1=open(to_dir+item,'w')
    for i in range(ans.shape[0]):
        print(str(int(100*ans[i])))
        f1.writelines(str(int(6.5*ans[i]))+'\n')
    f1.close()



