#!/usr/bin/python
# -*- encoding: utf-8 -*-
import pynlpir
import os
from bs4 import BeautifulSoup

os.chdir('/home/haoran/桌面/test/text')
file_lst=os.listdir(os.getcwd())
text_dir='/home/haoran/桌面/partition_test/'

cnt = 0
pynlpir.open()

for item in file_lst:
    cnt+=1
    if cnt:
        f=open(item,'r')
        text_lst=[]
        for line in f:
            if line[0]!='<':
                text_lst.append(line[:-1])
        f.close()
        text_s=''.join(text_lst)
        #print text_s.decode('utf-8')
        try:
            result=pynlpir.segment(text_s.decode('utf-8'),pos_tagging = False)
        except:
            print item
            continue
        f2=open(text_dir+item[:-4],'w')
        for item in result:
            print>>f2,item.encode('utf-8')
            if item == '。'.decode('utf-8'):
                print>>f2,'\n'
        f2.close()

pynlpir.close()





