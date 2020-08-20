%***************************************
%******基于主元分析法的数据降维研究*******
%***************************************
%X为原始数据矩阵，每个样本是一个行向量，各维都在一行上
function newData=feature_PCA2(X,threshold)


% ****步骤二：原始数据X减去平均值，得到meanadjusted
[n,p]=size(X);   %p为原始数据维数，n为原始数据样本的个数
means=mean(X);   %平均值，means为1*n的行向量
for var = 1:p
tmpmeanadjusted(:,var) = X(:,var) - means(:,var);
end
meanadjusted = tmpmeanadjusted;
% disp('   调整后的数据:')
% disp(meanadjusted)

%*** 步骤三：求协方差矩阵c
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
% disp('   协方差矩阵：')
% disp(C)

%***步骤四：计算协方差矩阵的特征值和特征向量，并按降序排序，
%    得到特征向量为vsort，特征值为dsort（一维行向量）
[v,d]=eig(C);
d1=diag(d)';
[d2 index]=sort(d1); %以升序排序
cols=size(v,2);% 特征向量矩阵的列数
vsort=[];
for i=1:cols
vsort(:,i) = v(:,index(cols-i+1) );
end  %完成特征向量的降序排列
tmp=sort(-1*d2);
dsort=-1*tmp;%完成特征值的降序排列
% disp('   特征根(降序):')
% disp(dsort)
% disp('   特征向量（降序）:')
% disp(vsort)

%***步骤五：选择组件并形成特征矢量
%（1）计算累计方差贡献率addcontri;
%     并根据保留原有信息的85%原则，确定主元个数m
%（2）选择特征矢量featuresvector
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
% disp('   累计方差贡献率:')
% disp(addcontri)
% disp('   降维后的主元个数：')
% disp(m)
% disp('   特征矢量：')
% disp(featurevector)

%***步骤六：推出新的数据集组
rowfeaturevector=featurevector';
rowdataadjust=meanadjusted';
finaldata= rowfeaturevector*rowdataadjust;
rowfinaldata=finaldata';
newData=rowfinaldata;
% disp('   新的数据集组：')
% disp(rowfinaldata)

% %***步骤七：找回旧数据originaldata
% originalmean=ones(n,1)*means;
% originaldata=(rowfeaturevector'*finaldata)'+originalmean;
% disp('   找回的旧数据：')
% disp(originaldata)

    