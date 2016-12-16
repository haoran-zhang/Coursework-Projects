import numpy

#global variable
lst=numpy.load("C:/Users/haoran/Desktop/assignment/pos_lst.npy")
lst=list(lst)
lst.insert(0,'S1\n')
lst.insert(0,'S2\n')
print(lst)

lst2=numpy.load("C:/Users/haoran/Desktop/assignment/wrd_lst.npy")   
lst2=list(lst2)     #wrd list

pos_fq=numpy.load("C:/Users/haoran/Desktop/assignment/pos_fq.npy")
pos_fq=list(pos_fq)								   #pos frequence

total_freq=numpy.load("C:/Users/haoran/Desktop/assignment/total_freq.npy")
total_freq=list(total_freq)     #total frequence count the word-tag relationship

two=numpy.load("C:/Users/haoran/Desktop/assignment/two.npy")
three=numpy.load("C:/Users/haoran/Desktop/assignment/three.npy")



sum_two=0
sum_three=0
def sum2and3():
	global sum_two,sum_three
	for i in range(len(two)):
		sum_two+=sum(two[i])
	
	for i in range(len(three)):
		for j in range(len(three[0])):
			sum_three+=sum(three[i][j])
sum2and3()

def v_given_w_u(w,u,v):
	'''
	TAKE IN STRING (W,U,V) as number lst[w],lst[u],lst[v] means the pos
	'''
	row=w
	col=u
	zray=v
	wuv=1.0*three[row][col][zray]/sum_three
	wu=1.0*two[row][col]/sum_two
	#stupid backoff here
	if(wuv==0):
		uv=1.0*two[u][v]/sum_two
		u=1.0*pos_fq[u-2]/sum(pos_fq)
		return 0.4*uv/u

	if wu==0 : return 0
	else : return wuv/wu
 
v_w_u=[[[0 for i in range(len(lst))] for j in range(len(lst))] for k in range(len(lst))]

for i in range(len(lst)):
	for j in range(len(lst)):
		for k in range(len(lst)):
			v_w_u[i][j][k]=v_given_w_u(i,j,k)


x_v=[[0 for i in range(len(lst2)+5)] for j in range(len(lst))]

def x_given_v(x,v):
	'''
	this calculate the P(X|V),
	which, lst2[x] is the word, and lst[v] is the pos 
	'''
	row=v-2
	if x==-1:
		if row==0:
			return 0.4
		elif row==1:
			return 0.6
		elif row==3:
			return 0.6
		else:return 0
	elif x==-2:  #chengyu
		if row== 3:   #vv\n
			return 1
		else: return 0
	elif x==-3 :
		if row==0:
			return 1
		else : return 0
	elif x==-4:
		if row==0:
			return 1
		else:return 0
	elif x==-5:
		if row==7:
			return 1
		else:return 0

	else:
		col=x
	xv=1.0*total_freq[row][col]
	sum_col=0
	for j in range(len(total_freq)):
		sum_col+=total_freq[j][col]
	v=1.0*sum_col
	return xv/v


for i in range(len(lst)):
	for j in range(len(lst2)+5):
		x_v[i][j]=x_given_v(j-5,i)

numpy.save("C:/Users/haoran/Desktop/assignment/v_w_u.npy",v_w_u)
numpy.save("C:/Users/haoran/Desktop/assignment/x_v.npy",x_v)