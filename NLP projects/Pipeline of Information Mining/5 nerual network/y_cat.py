#after trying, I choose to use dictionary here
dic_y_cat={}
f=open('/home/haoran/桌面/f_y','r')
cnt=0
y_list=[]
for line in f:
    lst_of_line=line.split()
    y=lst_of_line[0]
    y_list.append(y)
    x=lst_of_line[3]
    dic_y_cat.setdefault(int(x),int(y))
f.close()

y_list=y_list.sort(reverse=True)

def y_cat (wrd_idx):
    if wrd_idx in dic_y_cat.keys():
        return dic_y_cat(wrd_idx)
    else:
        return y_list[0]



