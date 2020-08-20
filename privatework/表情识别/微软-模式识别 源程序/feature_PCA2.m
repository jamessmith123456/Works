%***************************************
%******������Ԫ�����������ݽ�ά�о�*******
%***************************************
%XΪԭʼ���ݾ���ÿ��������һ������������ά����һ����
function newData=feature_PCA2(X,threshold)


% ****�������ԭʼ����X��ȥƽ��ֵ���õ�meanadjusted
[n,p]=size(X);   %pΪԭʼ����ά����nΪԭʼ���������ĸ���
means=mean(X);   %ƽ��ֵ��meansΪ1*n��������
for var = 1:p
tmpmeanadjusted(:,var) = X(:,var) - means(:,var);
end
meanadjusted = tmpmeanadjusted;
% disp('   �����������:')
% disp(meanadjusted)

%*** ����������Э�������c
%For each pair of variables, x1, x2, calculate
% sum ((x1 - meanx1)'*(x2-meanx2))/(m-1)
sizex=size(meanadjusted);
meanx = mean (meanadjusted);% Get the mean of each column
for i= 1:sizex(2)
    x1=meanadjusted(:,i);
    mx1=meanx(i);
    for j=i:sizex(2)
        x2=meanadjusted(:,j);
        mx2=meanx (j);
        v=((x1 - mx1)' * (x2 - mx2))/(sizex(1) - 1);
        cv(i,j) = v;
        cv(j,i) = v;
    end
end
C=cv;
% disp('   Э�������')
% disp(C)

%***�����ģ�����Э������������ֵ������������������������
%    �õ���������Ϊvsort������ֵΪdsort��һά��������
[v,d]=eig(C);
d1=diag(d)';
[d2 index]=sort(d1); %����������
cols=size(v,2);% �����������������
vsort=[];
for i=1:cols
vsort(:,i) = v(:,index(cols-i+1) );
end  %������������Ľ�������
tmp=sort(-1*d2);
dsort=-1*tmp;%�������ֵ�Ľ�������
% disp('   ������(����):')
% disp(dsort)
% disp('   ��������������:')
% disp(vsort)

%***�����壺ѡ��������γ�����ʸ��
%��1�������ۼƷ������addcontri;
%     �����ݱ���ԭ����Ϣ��85%ԭ��ȷ����Ԫ����m
%��2��ѡ������ʸ��featuresvector
Xsum=sum(dsort);
for i=1:cols
    contri(i)=dsort(i)/Xsum;
end
m=1;
for i=1:cols
    addcontri(i)= sum(contri(1:i))^2;
    if(addcontri(i)<threshold)
        m=m+1;      
    end
end
featurevector = vsort(:,1:m);
% disp('   �ۼƷ������:')
% disp(addcontri)
% disp('   ��ά�����Ԫ������')
% disp(m)
% disp('   ����ʸ����')
% disp(featurevector)

%***���������Ƴ��µ����ݼ���
rowfeaturevector=featurevector';
rowdataadjust=meanadjusted';
finaldata= rowfeaturevector*rowdataadjust;
rowfinaldata=finaldata';
newData=rowfinaldata;
% disp('   �µ����ݼ��飺')
% disp(rowfinaldata)

% %***�����ߣ��һؾ�����originaldata
% originalmean=ones(n,1)*means;
% originaldata=(rowfeaturevector'*finaldata)'+originalmean;
% disp('   �һصľ����ݣ�')
% disp(originaldata)

    