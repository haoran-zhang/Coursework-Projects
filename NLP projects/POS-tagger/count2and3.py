#to count the y(x1,x2) and y(x1,x2,x3 from trn.pos)
import numpy
lst=numpy.load("C:/Users/haoran/Desktop/assignment/pos_lst.npy")
lst=list(lst)

lst.insert(0,'S1\n')
lst.insert(0,'S2\n')

f_pos=open('C:/Users/haoran/Desktop/assignment/trn.pos','rt')
pos_lst=list(f_pos)
f_pos.close()
#print(len(pos_lst),'is this equal to 211189')


#make count table
two=[[0 for i in range(len(lst))] for j in range(len(lst))]
three=[[[0 for i in range(len(lst))] for j in range(len(lst))] for k in range(len(lst))]

j=0
while(j<=len(pos_lst)-1):   #soft constrain
	tmp=[]
	while(pos_lst[j] != '\n'):
		tmp.append(pos_lst[j])
		j+=1
	j+=1

	#count y(x1,x2)
	tmp.insert(0,'S1\n')
	tmp.insert(0,'S2\n')
	for i in range(0,len(tmp)-2):
		row=lst.index(tmp[i])
		col=lst.index(tmp[i+1])
		two[row][col]+=1
		zray=lst.index(tmp[i+2])
		three[row][col][zray]+=1

	#two table last item
	row=lst.index(tmp[-2])
	col=lst.index(tmp[-1])
	two[row][col]+=1

for i in range(len(two)):
	for j in range(len(two[1])):
		if two[i][j]==0:
			print(i,j)

numpy.save("C:/Users/haoran/Desktop/assignment/two.npy",two)
numpy.save("C:/Users/haoran/Desktop/assignment/three.npy",three)
