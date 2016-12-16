import numpy

#load all intermedia result

lst=numpy.load("C:/Users/haoran/Desktop/assignment/pos_lst.npy")
lst=list(lst)
lst.insert(0,'S1\n')
lst.insert(0,'S2\n')

lst2=numpy.load("C:/Users/haoran/Desktop/assignment/wrd_lst.npy")   
lst2=list(lst2)     #wrd list

pos_fq=numpy.load("C:/Users/haoran/Desktop/assignment/pos_fq.npy")
pos_fq=list(pos_fq)								   #pos frequence

total_freq=numpy.load("C:/Users/haoran/Desktop/assignment/total_freq.npy")
total_freq=list(total_freq)     #total frequence count the word-tag relationship

two=numpy.load("C:/Users/haoran/Desktop/assignment/two.npy")
three=numpy.load("C:/Users/haoran/Desktop/assignment/three.npy")

v_w_u=numpy.load("C:/Users/haoran/Desktop/assignment/v_w_u.npy")
x_v=numpy.load("C:/Users/haoran/Desktop/assignment/x_v.npy")

#third part conditional p

#forth part viterbi

def lst_index( l,x ):
	'''
	find the index of the word in lst2, or catch up the item not in the list and
	write features for them
	'''
	try:
		return( l.index( x ) )
	except:
		#print( 'lst_index wrong' )
		#print( x )
		if( len( x ) == 13 ) : return -2  #names chengyu
		l1=['\xe7\x8e\x8b','\xe6\x9d\x8e','\xe5\xbc\xa0', '\xe5\x88\x98', '\xe9\x99\x88', '\xe6\x9d\xa8', '\xe9\xbb\x84', '\xe8\xb5\xb5', '\xe5\x91\xa8', '\xe5\x90\xb4']
		for item in l1:
			if item in x:
				return -4		
		if ( len( x ) == 10 ) :return -3  #means nr things
		l=['\xe4\xb8\x80','\xe4\xba\x8c','\xe4\xb8\x89' , '\xe5\x9b\x9b','\xe4\xba\x94' ,'\xe5\x85\xad' , '\xe4\xb8\x83' ,'\xe5\x85\xab','\xe4\xb9\x9d','\xe5\x8d\x81','\xe7\x99\xbe','\xe5\x8d\x83','\xe4\xb8\x87','\xe4\xba\xbf']
		for item in l:
			if item in x:
				return -5        #contain numbers
		return -1

def v2( tmp ):
	'''
    only used when len(tmp)==2
	'''
	l_tmp = len( tmp )
	#F[i][u][v], i means the position in tmp.  u,v are the possible pos for
	#tmp[i],tmp[i-1]
	F = [[[0 for i in range( len( lst ) )] for j in range( len( lst ) )] for k in range( l_tmp )]
	path = [[[0 for i in range( len( lst ) )] for j in range( len( lst ) )] for k in range( l_tmp )]
	
	#case i=0
	num1 = lst_index( lst2,tmp[ 0 ] )
	for k in range( 2,len( lst ) ):
		F[ 0 ][ 1 ][ k ] = 1.0 * v_w_u[0][1][k] * x_v[k][num1+5]
		#print(k)
		#print(F[0][1][k],v_given_w_u(0,1,k),lst2.index(tmp[0]),x_given_v(lst2.index(tmp[0]),k),tmp[0],lst[k])
	#case i=1
	num2 = lst_index( lst2,tmp[ 1 ] )
	for k in range( 2,len( lst ) ):
		for j in range( 2,len( lst ) ):
			F[ 1 ][ k ][ j ] = 1.0 * F[ 0 ][ 1 ][ k ] * v_w_u[ 1][k][j] * x_v[j][num2+5]
			#print(k,j)
	
	max_F = 0.0
	i = len( tmp ) - 1
	last1 = 0
	last2 = 0
	for k in range( len( lst ) ):
		for j in range( len( lst ) ):
			if F[ i ][ k ][ j ] > max_F:
				max_F = F[ i ][ k ][ j ]
				last1 = j
				last2 = k

	ans_lst = [ last2,last1 ]

	for i in range( len( tmp ) - 1,1,-1 ):
		position = path[ i ][ ans_lst[ 0 ] ][ ans_lst[ 1 ] ]
		ans_lst.insert( 0,position )

	#print(ans_lst)
	ans_lst2 = []
	for i in range( len( ans_lst ) ):
		ans_lst2.append( lst[ ans_lst[ i ] ] )

	return ans_lst2

def v( tmp ):
	'''
	orignal viterbi algorithm
	'''
	l_tmp = len( tmp )
	#F[i][u][v], i means the position in tmp.  u,v are the possible pos for
	#tmp[i],tmp[i-1]
	F = [[[0 for i in range( len( lst ) )] for j in range( len( lst ) )] for k in range( l_tmp )]
	path = [[[0 for i in range( len( lst ) )] for j in range( len( lst ) )] for k in range( l_tmp )]
	
	#case i=0
	num1 = lst_index( lst2,tmp[ 0 ] )
	for k in range( 2,len( lst ) ):
		F[ 0 ][ 1 ][ k ] = 1.0 * v_w_u[0][1][k] * x_v[k][num1+5]
		#print(k)
		#print(F[0][1][k],v_given_w_u(0,1,k),lst2.index(tmp[0]),x_given_v(lst2.index(tmp[0]),k),tmp[0],lst[k])
	#case i=1
	num2 = lst_index( lst2,tmp[ 1 ] )
	for k in range( 2,len( lst ) ):
		for j in range( 2,len( lst ) ):
			F[ 1 ][ k ][ j ] = 1.0 * F[ 0 ][ 1 ][ k ] * v_w_u[ 1][k][j] * x_v[j][num2+5]
			#print(k,j)
	#for cases >=2
	for i in range( 2,len( tmp ) ):
		num3 = lst_index( lst2,tmp[ i ] )
		for k in range( 2,len( lst ) ):
			for j in range( 2,len( lst ) ):
				m_F = 0
				position = 0
				for w in range( 2,len( lst ) ):
					a = 1.0 * F[ i - 1 ][ w ][ k ] * v_w_u[w][k][j ] * x_v[j][ num3+5]
					if(a and a > m_F ):
						m_F = a
						position = w
					#print(i,k,j,w)
				#F[i][k][j]=max([ F[i-1][w][k]*v_given_w_u(w,k,j)*x_given_v(i,j) for w in
				#range(len(lst)) ])
				F[ i ][ k ][ j ] = m_F
				path[ i ][ k ][ j ] = position
					#print(F[i][j][k])
	#this to find the largest possibility
	max_F = 0.0
	i = len( tmp ) - 1
	last1 = 0
	last2 = 0
	for k in range( len( lst ) ):
		for j in range( len( lst ) ):
			if F[ i ][ k ][ j ] > max_F:
				max_F = F[ i ][ k ][ j ]
				last1 = j
				last2 = k

	ans_lst = [ last2,last1 ]

	for i in range( len( tmp ) - 1,1,-1 ):
		position = path[ i ][ ans_lst[ 0 ] ][ ans_lst[ 1 ] ]
		ans_lst.insert( 0,position )

	#print(ans_lst)
	ans_lst2 = []
	for i in range( len( ans_lst ) ):
		ans_lst2.append( lst[ ans_lst[ i ] ] )

	return ans_lst2


def main(item):
	'''
	induce the sentences in tst.wrd to viterbi function
	'''
	#input dev_wrd file
	f_wrd=open(dr+'/'+item,'rt')
	dev_wrd=list(f_wrd)
	f_wrd.close()

	ans_lst=[]
	j=0
	try:
		while(j<=len(dev_wrd)-1):   #soft constrain
			tmp=[]
			while(dev_wrd[j] != '\n'):
				tmp.append(dev_wrd[j])
				j+=1
			j+=1
			if(len(tmp)>=3):
				ans_lst+=v(tmp)
				ans_lst+='\n'
			elif(len(tmp)==2):
				ans_lst+=v2(tmp)
				ans_lst+='\n'
			elif(len(tmp)==1):
				idx=lst_index(lst2,tmp[0])
				m=0
				idx2=0
				for i in range(len(lst)-2):
					a=total_freq[i][idx]
					if (a>m):
						m=a
						idx2=i
				ans_lst+=lst[i]
				ans_lst+='\n' 
	finally:
		f=file('C:/Users/haoran/Desktop/pos_train/'+item+'.pos','w')
		for item in ans_lst:
			f.write(item)
		f.close()

dr='C:/Users/haoran/Desktop/partition_train'
import os
os.chdir(dr)
file_list=os.listdir(os.getcwd())
for item in file_list:
	main(item)
	

