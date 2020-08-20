%利用系统自带的PCA函数进行降维的方法
%data:要降维的数据，每一行代表一个样本的数据
function dataPCA = feature_PCA1(data,featureNum)

% 
% clear;
% clc
% %% load cities data
% load cities
% whos
%   Name             Size         Bytes  Class
%   categories       9x14           252  char array
%   names          329x43         28294  char array
%   ratings        329x9          23688  double array
%% box plot for ratings data
% To get a quick impression of the ratings data, make a box plot
% figure;
% boxplot(ratings,'orientation','horizontal','labels',categories);
% grid on;
% print -djpeg 1;

% %% pre-process
% [m,n]=size(data);
% stdr = std(data);
% sr = data./repmat(stdr,m,1);
% %% use princomp 
% [coef,score,latent,t2] = princomp(sr);

[m,n]=size(data);
%标准化矩阵
temp=zscore(data);
%调用主成分分析函数
[pc,score,latent,tsquare]  = princomp(temp);

%取出前featureNum个主要成分
if featureNum>n
    featureNum=n;
end
dataPCA=score(:,1:featureNum);

%% 输出参数讲解

% coef:9*9
% 主成分系数:即原始数据线性组合生成主成分数据中每一维数据前面的系数.
% coef的每一列代表一个新生成的主成分的系数.
% 比如你想取出前三个主成分的系数,则如下可实现:pca3 = coef(:,1:3);

% score:329*9
% 字面理解:主成分得分
% 即原始数据在新生成的主成分空间里的坐标值.

% latent:9*1
% 一个列向量,由sr的协方差矩阵的特征值组成.
% 即 latent = sort(eig(cov(sr)),'descend');
% 测试如下:
% sort(eig(cov(sr)),'descend') =
%     3.4083
%     1.2140
%     1.1415
%     0.9209
%     0.7533
%     0.6306
%     0.4930
%     0.3180
%     0.1204
% latent =
%     3.4083
%     1.2140
%     1.1415
%     0.9209
%     0.7533
%     0.6306
%     0.4930
%     0.3180
%     0.1204

% t2:329*1
% 一中多元统计距离,记录的是每一个观察量到中心的距离
%% 如何提取主成分,达到降为的目的
% 通过latent,可以知道提取前几个主成分就可以了.
% figure;
% percent_explained = 100*latent/sum(latent);
% pareto(percent_explained);
% xlabel('Principal Component');
% ylabel('Variance Explained (%)');
% print -djpeg 2;

% 图中的线表示的累积变量解释程度.
% 通过看图可以看出前七个主成分可以表示出原始数据的90%.
% 所以在90%的意义下只需提取前七个主成分即可,进而达到主成分提取的目的.
%% Visualizing the Results

% 结果的可视化
% figure;
% biplot(coef(:,1:2), 'scores',score(:,1:2),... 
% 'varlabels',categories);
% axis([-.26 1 -.51 .51]);
% print -djpeg 3;

% 横坐标和纵坐标分别表示第一主成分和第二主成分
% 红色的点代表329个观察量,其坐标就是那个score
% 蓝色的向量的方向和长度表示了每个原始变量对新的主成分的贡献,其坐标就是那个coef.