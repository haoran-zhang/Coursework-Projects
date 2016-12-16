#!/usr/bin/python
# -*- encoding: utf-8 -*-
from nltk.parse import DependencyGraph, DependencyEvaluator
from transitionparse import TransitionParser, Configuration, Transition
import pickle
import os
gold_sent=DependencyGraph("""
中国	NR	0
鼓励	VV	0
民营	JJ	0
企业家	NN	0
投资	VV	0
国家	NN	0
基础	NN	0
建设	NN	0
""")
os.chdir('C:/Users/Haoran Zhang/Desktop/m3+')
file_lst=os.listdir(os.getcwd())
to_dir='C:/Users/Haoran Zhang/Desktop/pas_test2/'

def parse(f_item):
	dg_list=[]  
	sent=''
	cnt=0
	with open(f_item,'rt',encoding='utf-8')as f:
		for line in f:
			if line != '\n':
				sent+=line
			else:
				dg=DependencyGraph(sent,zero_based=True)
				dg_list.append(dg)
				sent=''
				cnt+=1
				print('the index of sentence'+str(cnt))
			if cnt==3:
				pass

	#only use them to predict, no training!!
	parser_std=TransitionParser('arc-standard')

	#loading trained features
	folder_dir='C:/Users/Haoran Zhang/Desktop/nlp/'
	parser_std._dictionary = pickle.load(open(folder_dir+'self._dictionary', 'rb'))
	parser_std._transition = pickle.load(open(folder_dir+'self._transition', 'rb'))
	parser_std._match_transition=pickle.load(open(folder_dir+'self.match_transition', 'rb'))

	result = parser_std.parse(dg_list,folder_dir+'temp.arcstd.model')  #result here is a list of dg
	#print(result[0].to_conll(4))
	try: 
	    f=open(to_dir+f_item,'w')
	    for item in result:
	        f.writelines(item.to_conll(4))
	        f.writelines('\n')
	finally:
	    f.close()


cnt=0
log=[]
for item_f in file_lst:
	cnt+=1
	print('-----------The current file '+item_f+' '+str(cnt))
	try:
		parse(item_f)
	except:
		log.append(item_f)
print(log)
	