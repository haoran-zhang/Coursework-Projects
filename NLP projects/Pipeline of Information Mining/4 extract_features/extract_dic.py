#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import pickle
os.chdir('/home/haoran/桌面/pas')
file_lst=os.listdir(os.getcwd())
new_dir='/home/haoran/桌面/'
dic_of_word={}
dic_of_pos={}
dic_of_parse={}

for item in file_lst:
    f=open(item,'r',encoding='gbk')
    for line in f:
        if line[0] != '\n':
            lst=line.split()
            dic_of_word.setdefault(lst[0],1)
            dic_of_pos.setdefault(lst[1],1)
            if len(lst)>3:
                dic_of_parse.setdefault(lst[3],1)
cnt=0
for keys in dic_of_word:
    cnt+=1
    dic_of_word[keys]+=cnt
cnt=0
for keys in dic_of_parse:
    cnt += 1
    dic_of_parse[keys] += cnt
cnt=0
for keys in dic_of_pos:
    cnt+=1
    dic_of_pos[keys]+=cnt

pickle.dump(dic_of_word,open(new_dir+'dic_of_word','wb'))
pickle.dump(dic_of_parse,open(new_dir+'dic_of_parse','wb'))
pickle.dump(dic_of_pos,open(new_dir+'dic_of_pos','wb'))