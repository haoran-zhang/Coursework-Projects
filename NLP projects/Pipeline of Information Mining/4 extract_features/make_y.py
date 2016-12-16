#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
the format of files in pas_y:
12 13695 : the index of subtype, and the index of word
'''
import os
import pickle
from bs4 import BeautifulSoup
os.chdir('/home/haoran/桌面/pas')
file_lst=os.listdir(os.getcwd())
new_dir='/home/haoran/桌面/train_tag/'
y_dir='/home/haoran/桌面/pas_y/'

dic_of_type=pickle.load(open( '/home/haoran/桌面/dic_of_type', "rb" ) )
dic_of_word=pickle.load(open( '/home/haoran/桌面/dic_of_word', "rb" ) )

for item in file_lst:
    f=open(new_dir+item[:-4]+'.apf.xml','r')
    f1=open(y_dir+item[:-4],'w')
    soup=BeautifulSoup(f.read(),'lxml')
    a=soup.find_all('event')
    for i in range(len(a)):
        print('---------------',a[i]['type'],a[i]['subtype'],a[i].anchor)
        s=a[i].anchor.text.strip()
        try:
            idx_word=dic_of_word[s]
        except:
            'do nothing'
        f1.writelines(str(dic_of_type[str(a[i]['type'])+'_'+str(a[i]['subtype'])])+' '+str(idx_word)+'\n')
    f.close()
    f1.close()








