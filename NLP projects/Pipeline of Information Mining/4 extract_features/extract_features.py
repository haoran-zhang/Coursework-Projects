#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import pickle
y_dir='/home/haoran/桌面/pas_y/'

dic_of_type=pickle.load(open( '/home/haoran/桌面/dic_of_type', "rb" ) )  #subtype of trigger
dic_of_word=pickle.load(open( '/home/haoran/桌面/dic_of_word', "rb" ) )
dic_of_parse=pickle.load(open( '/home/haoran/桌面/dic_of_parse', "rb" ) )
dic_of_pos=pickle.load(open( '/home/haoran/桌面/dic_of_pos', "rb" ) )

def process(sent,par1,par2):
    '''
    processing on certain sentence
    '''
    result1=[]
    lst=sent.split('\n')
    if len(lst[-1])<=2:lst.pop()
    #consider every word
    for idx in range(len(lst)):
        lst_of_line=[]
        lst_of_line=lst[idx].split('\t')
        if len(lst_of_line)>=2:
            #print(lst_of_line)
            #feature 1 and 2, the number of total sentences and current sentense in the file
            result=[par1,par2]
            #feature 3 the idx of word
            result.append(dic_of_word[lst_of_line[0]])
            #feature 4 pos
            result.append(dic_of_pos[lst_of_line[1]])

            #feature 5 depth in tree
            depth=0
            curent=int(lst_of_line[2])
            while depth<len(lst):
                if curent==0 and lst_of_line[3]:
                    depth+=1
                    break
                elif curent==0:
                    depth=100       #if the word didn't find a dependency relation
                    break
                else:
                    curent=int((lst[curent-1]).split('\t')[2])
                    depth+=1
            result.append(depth)

            #feature 6: left 3 pos + left 1 word
            if idx>0:
                left1=lst[idx-1].split('\t')[1]
                num=dic_of_pos[left1]
                result.append(num)
                #left 1 word
                wrd=dic_of_word[lst[idx-1].split('\t')[0]]
                result.append(wrd)
                if idx>1:
                    left2 = lst[idx - 2].split('\t')[1]
                    num = dic_of_pos[left2]
                    result.append(num)
                    if idx>2:
                        left3 = lst[idx - 3].split('\t')[1]
                        num = dic_of_pos[left3]
                        result.append(num)
                    else: result.append(0)
                else:
                    result.append(0)
                    result.append(0)
            else:
                for i in range(4): result.append(0)
            #feature 7 :right 3 pos and 1 word
            if idx<len(lst)-1:
                r1=lst[idx+1].split('\t')[1]
                num=dic_of_pos[r1]
                result.append(num)
                #left 1 word
                wrd=dic_of_word[lst[idx+1].split('\t')[0]]
                result.append(wrd)
                if idx<len(lst)-2:
                    r2 = lst[idx +2].split('\t')[1]
                    num = dic_of_pos[r2]
                    result.append(num)
                    if idx<len(lst)-3:
                        r3 = lst[idx +3].split('\t')[1]
                        num = dic_of_pos[r3]
                        result.append(num)
                    else:
                        result.append(0)
                else:
                    result.append(0)
                    result.append(0)
            else:
                for i in range(4): result.append(0)

            #feature 8 : dependcy type
            if lst_of_line[3] :
                try:
                    result.append(dic_of_parse[lst_of_line[3]])
                except:
                    print(lst_of_line)
            else:
                result.append(100)   #if didn't find a dependency relation
            #feature 9 : head word and head pos
            if lst_of_line[3] :
                head=int(lst_of_line[2])
                if head!=0:
                    wrd=lst[head-1].split('\t')[0]
                    pos=lst[head-1].split('\t')[1]
                    result.append(dic_of_word[wrd])
                    result.append(dic_of_pos[pos])
                else:
                    result.append(0)
                    result.append(0)    #if the word is root
            else:
                result.append(100)
                result.append(100)  #if didn't find a dependency relation
            assert len(result)==16
            s=''
            for item in result:
                s+=' '+str(item)
            result1.append(s+'\n')
    return result1

def extract_features(f_item):
    '''
    processing on a file
    :param item:
    :return: 0 68:1 90:1...such load_svmlight() format
    '''
    result=[]
    sent = ''
    cnt_of_line = 0
    current_line=0
    with open(f_item, 'rt',encoding='gbk')as f:
        for line in f:
            if line[0] == '\n':
                cnt_of_line+=1
    with open(f_item, 'rt', encoding='gbk')as f:
        for line in f:
            if line[0] != '\n':
                sent += line
            else:
                current_line += 1
                #TODO : COMPARE THE Y AND CHANG THE TRAINING LINE
                f_y=open(y_dir+f_item[:-4],'r')
                y_lst_type=[]
                y_lst_word=[]
                for line in f_y:
                    y_lst_type.append(line.split()[0])
                    y_lst_word.append(line.split()[1])
                temp=process(sent,cnt_of_line,current_line)
                for item in temp:
                    wrd=item.split()[2]
                    if wrd in y_lst_word:
                        #TODO : IDENTIFY TRIGGER FIRST, THEN CLASSIFICATION
                        #idx=y_lst_word.index(wrd)
                        #item=y_lst_type[idx]+' '+item
                        item='1 '+item
                    else:
                        #print(item)
                        item='0 '+item
                        #print(item)
                    result+=item
                sent = ''
    return result

def extract_features2(f_item):
    '''
    for the purpose of calssify catagory of y
    :param item:
    :return: 0 68:1 90:1...such load_svmlight() format
    '''
    result=[]
    sent = ''
    cnt_of_line = 0
    current_line=0
    with open(f_item, 'rt',encoding='gbk')as f:
        for line in f:
            if line[0] == '\n':
                cnt_of_line+=1
    with open(f_item, 'rt', encoding='gbk')as f:
        for line in f:
            if line[0] != '\n':
                sent += line
            else:
                current_line += 1
                #TODO : COMPARE THE Y AND CHANG THE TRAINING LINE
                f_y=open(y_dir+f_item[:-4],'r')
                y_lst_type=[]
                y_lst_word=[]
                for line in f_y:
                    y_lst_type.append(line.split()[0])
                    y_lst_word.append(line.split()[1])
                temp=process(sent,cnt_of_line,current_line)
                for item in temp:
                    wrd=item.split()[2]
                    if wrd in y_lst_word:
                        #TODO : IDENTIFY TRIGGER FIRST, THEN CLASSIFICATION
                        idx=y_lst_word.index(wrd)
                        item=y_lst_type[idx]+' '+item
                        result.append(item)
                sent = ''
    return result


# f1=open(vector+'CBS20001006.1000.0874','w')
# result=extract_features('CBS20001006.1000.0874.pos')
# for item in result:
#     f1.writelines(item)
vector = '/home/haoran/桌面/vector/'
os.chdir('/home/haoran/桌面/pas')
f_y=open('/home/haoran/桌面/f_y','w')
file_lst=os.listdir(os.getcwd())
for f_item in file_lst:
    #result=extract_features(f_item)
    #f1=open(vector+f_item[:-4],'w')
    #for line in result:
    #    f1.writelines(line)
    #f1.close()
    result2=extract_features2(f_item)
    for line in result2:
        f_y.writelines(line)
f_y.close()
