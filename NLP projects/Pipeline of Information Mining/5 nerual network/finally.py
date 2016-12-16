#after trying, I choose to use dictionary here
import pickle
dic_y_cat={}
f=open('/home/haoran/桌面/f_y','r')
cnt=0
y_list={}
for line in f:
    lst_of_line=line.split()
    y=lst_of_line[0]
    if int(y) not in y_list:
        y_list.setdefault(int(y),1)
    else:
        y_list[int(y)]+=1
    x=lst_of_line[3]
    dic_y_cat.setdefault(int(x),int(y))
f.close()
m=0
k=''
for key in y_list:
    if y_list[key]>m:
        m=y_list[key]
        k=key


def y_cat (wrd_idx):
    if wrd_idx in dic_y_cat.keys():
        return dic_y_cat[wrd_idx]
    else:
        return key

dic_of_type=pickle.load(open( '/home/haoran/桌面/dic_of_type', "rb" ) )  #subtype of trigger
dic_of_word=pickle.load(open( '/home/haoran/桌面/dic_of_word', "rb" ) )
dic_of_parse=pickle.load(open( '/home/haoran/桌面/dic_of_parse', "rb" ) )
dic_of_pos=pickle.load(open( '/home/haoran/桌面/dic_of_pos', "rb" ) )

import os
from bs4 import BeautifulSoup
os.chdir('/home/haoran/桌面/predict_result')
file_lst=os.listdir(os.getcwd())
vector_test='/home/haoran/桌面/vector_test/'
partition_test='/home/haoran/桌面/partition_test2/'                        #in order to find start and end
tag_dir='/home/haoran/桌面/tag/'
to_dir='/home/haoran/桌面/result/'

for item in file_lst:
    wrd_lst=[]
    type_lst=[]
    index_lst=[]
    start_end=[]
    f_result=open(item,'r')
    f_vector=open(vector_test+item,'r')
    l_vector=f_vector.read().split('\n')
    f_vector.close()
    f_wrd=open(partition_test+item,'r')
    l_wrd=f_wrd.read().split('\n')
    cnt=0
    for line in f_result:
        if line[0]=='1':
            index_lst.append(cnt)
            wrd=int(l_vector[cnt].split()[2])
            y_type=y_cat(wrd)
            wrd_lst.append(wrd)
            type_lst.append(y_type)
            # TODO: FIND START AND END OF THE TRIGGER
            num=66
            start=0
            end=0
            for j in range(len(l_wrd)):
                if j!=index_lst[-1]:
                    num+=len(l_wrd[j])
                else:
                    start=num
                    num+=len(l_wrd[j])
                    end=num-1
                    start_end.append((start,end))
        cnt+=1

    print(wrd_lst,type_lst,start_end,item)
    assert len(wrd_lst)==len(type_lst)
    assert len(wrd_lst)==len(start_end)
    #TODO : WRITE IN XML
    f_xml=open(tag_dir+item+'.apf.xml','r')
    f_to=open(to_dir+item+'.apf.xml','w')
    print(wrd_lst,type_lst,start_end,'-------------')
    s=f_xml.read()
    soup=BeautifulSoup(s,'lxml')
    original_tag=soup.document
    for i in range(len(wrd_lst)):
        for key in dic_of_type.keys():
            if dic_of_type[key] == type_lst[i]:
                type_y=key.split('_')
        event=soup.new_tag("event",ID=item+'-EV'+str(i+1),TYPE=type_y[0],SUBTYPE=type_y[1])
        anchor=soup.new_tag("anchor")
        charseq=soup.new_tag("charseq",START=start_end[i][0],END=start_end[i][1])
        for key in dic_of_word.keys():
            if dic_of_word[key] == wrd_lst[i]:
                word = key
        charseq.append(soup.new_string(word))
        anchor.append(charseq)
        event.append(anchor)
        original_tag.append(event)

    f_to.write(soup.prettify())
    f_to.close()
    f_xml.close()




