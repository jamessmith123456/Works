function [CI, CR, W] = getW(A)
%此函数用于分析N阶矩阵 从而计算出权重系数 并且判断原始的矩阵是否合理

[n,~]=size(A);%由于矩阵构造方法的原因，矩阵都是正方形的所以关于矩阵的大小只需要取一个参数
Asum=sum(A,1);%求每一列的和 对于矩阵A  逐列求和  每列的所有行相加得到一个值
Aprogress=A./(ones(n,1)*Asum);%计算每一列个元素在这一列占的比重 即 逐列计算 每列中的每一行 分别计算在该列的比例
W=sum(Aprogress,2)./n;%每一行元素相加取平均值，需要注意这里W是个列项量且所有值加起来等于1 逐行计算 每行的全部值加起来求平均
w=A*W;%如果A的矩阵是理想状况的话这里W=w
lam=sum(w./W)/n;%通过这一步最大lam 最大特征根
RI=[0,0,0.58,0.9,1.12,1.24,1.32,1.41,1.45]; %引入判断矩阵的平均随机一致性指标RI(rand index) 不同的算法这个指标不一样
CI=(lam-n)/(n-1); %CI是一致性指标
CR=CI/RI(n);%计算误差  RI(n)是随机一致性指标   CR是一致性比率
if CR<0.10 %如果误差小于0.1则可以接受 CR小于0.1 通过一致性检验
    disp('此矩阵的一致性可以接受!');
    disp('CR');
    disp(CR);
    disp('lam_max:');
    disp(max(lam));
    %fprintf('Cl=');disp(CI);
    %fprintf('CR=');disp(CR);
    %fprintf('W=');disp(W);
else
    disp('此矩阵的一致性不可以接受!');
end

end

