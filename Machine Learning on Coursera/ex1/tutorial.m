%octave variable

%semicolon supressing output  ;

disp(a);
disp(sprintg('2 decimal: %0.2f',a)) 

format long

A=[1 2; 3 4; 5 6]

v=[1 2 3]  %vector
v=[1; 2; 3]
v=1:0.1:2  %from 1 to 2 step 0.1
v=1:6  %1 to 6

ones(2,3)
%2 by 3 matrics
c=2*ones(2,3)
w=zeros(1,3)

w=rand(1,3)  %random 0-1
w=randn(1,3) %n-distribution

hist(w) %plot history 

eye(3)

help eye


%--------------------------%
%moving data around
sz=size(A)
size(A,1);size(A,2);
length(v) %usually vector

pwd %current path
cd 'C:\Users\ang\Desktop' %change directory
ls  %show files


load featuresX.dat
load('faturesX.dat')

who  %variables

featuresX
size(featuresX)

whos 
clear featuresX

v=priceY(1:10) %取前1到10个数

save hello.mat v  
save hello.txt v -ascii 
load helo.mat

A(3,2) %INDEX

A(2,:) %':'MEANS EVERY ELEMENT ,那一行的！
A(:,3)

A([1 3], :)

%ASSIGNMENT
A(:,2)=[10; 11; 12]

A=[A, [100; 101; 102]]  %APPEND A ROW
A(:)  %PUT ALL THE ELEMENT INTO A SIGLE VECTOR

C=[A B] %CAT THEM TOGATHER

C=[A;B] %CAT THEM UP AND DOWN


%-------------------------------%
%COMPUTATIONAL OPERATION

A=[1 2; 3 4; 5 6]

A*C 

A .*B  %EVERY ELEMENT
A .^2

V=[1; 2; 3]
1 ./V

1 ./A

log(V)
exp(V)
abs(V)

-V %-1*V

V + ones(length(v),1)  %equal to v + 1

A' %TRANSFOR

val=max(a)
[val,idx]=max(a)

a < 3
%return whether true or false

find(a<3)
%return idx

A=magic(3)  %shu du...

[r,c] =find(A>=7)
%JUST HELP FUNCTION

sum(a)
prod(a)

floor(a)
ceil(a)

rand(3)
max(rand(3),rand(3))

max(A,[],1)  %EQUAL TO max(A)
%PER COL
max(A,[],2)
%PER ROW

max(max(A))
max(A(:))

A=magic(9)

sum(A,1)
%PER COL

A .*eye(9)
sum(sum(A .*eye(9)))

flipud(eye(9))

A=magic(3)

pinv(A) *A

%---------------------------------%
%PLOTTING
plot(t,y1);
hold on;
plot(t,y2,'r');
xlable('time')
ylable('value')
legend('sin','cos')
title('my plot')
cd 'C:\'; print -dpng 'myplot.png'
close

figure(1);plot(t,y1);
figure(2);plot(t,y2);

subplot(1,2,1); 
plot(t,y1);
subplot(1,2,2);
plot(t,y2);
axis([0.5 1 -1 1])

clf;
imagesc(A)
imagesc(A),colorbar,colormap grap

a=1,b=2,c=3

%comma : multiple comand in one line


%----------------------------------%
%control 

for i=1:10,
	v(i)=2^i;
end;

indices=1:10

for i=indices,
	disp(i);
end;

i=1;
while i<=5;
	i=i+1;
end;


while true,
	if i==6,
		break;
	end;
end;

if v(1)==1,

elseif v(1)==2,

else
	
end;

%-----------------------------%
function y=squareThinsNumber(x)
	y=x^2


addpath('C:\')

function [y1,y2]=sq(x)
	y1=x^2;
	y2=x^3
%multiple return


%----------------------------------%
%vectoration


%----------------------------------%
%submit
submit()

















