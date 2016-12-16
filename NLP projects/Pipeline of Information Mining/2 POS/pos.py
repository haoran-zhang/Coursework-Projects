#def print_chinese(str):
 #   str=str[:-1].decode('utf-8') 
  #  print(str)
import numpy

lst=[]
with open('C:/Users/haoran/Desktop/assignment/trn.pos','rt')as f:
	for line in f:
		if not line in lst and line != '\n':
			lst.append(line) 
print (lst,len(lst))

pos_fq=[0 for i in range(len(lst))]
with open('C:/Users/haoran/Desktop/assignment/trn.pos','rt')as f:
	for line in f:
		for i in range(len(lst)):
			if line==lst[i]:
				pos_fq[i]+=1
print (pos_fq)
numpy.save("C:/Users/haoran/Desktop/assignment/pos_fq.npy",pos_fq)


lst2=[]
cnt=0
with open('C:/Users/haoran/Desktop/assignment/trn.wrd','rt')as f:
	for line in f:
		cnt+=1
		if not line in lst2 and line != '\n':
			lst2.append(line) 
print (len(lst2),cnt)

#for i in range(6):
 #   print_chinese(lst2[i])


total_freq=[[0 for i in range(len(lst2))] for j in range(len(lst))]

#with open('C:/Users/haoran/Desktop/assignment/trn.wrd','rt')as f:
#	for line in f:
#		if line != '\n':
#			idx=lst2.index(line)
f_pos=open('C:/Users/haoran/Desktop/assignment/trn.pos','rt')
pos_lst=list(f_pos)
f_pos.close()

f_wrd=open('C:/Users/haoran/Desktop/assignment/trn.wrd','rt')
wrd_lst=list(f_wrd)
f_wrd.close()

for i in range(len(wrd_lst)):
	if pos_lst[i] in lst:
		pos_idx=lst.index(pos_lst[i])
		wrd_idx=lst2.index(wrd_lst[i])
		total_freq[pos_idx][wrd_idx]+=1

#print(total_freq[0][0],total_freq[1][1],total_freq[1][2],len(total_freq),len(total_freq[0]))

#tune the NN TO BE A LITTLE LESS
for i in range(len(lst)):
	total_freq[i][1]*=0.8


numpy.save("C:/Users/haoran/Desktop/assignment/total_freq.npy",total_freq)
numpy.save("C:/Users/haoran/Desktop/assignment/pos_lst.npy",lst)
numpy.save("C:/Users/haoran/Desktop/assignment/wrd_lst.npy",lst2)

