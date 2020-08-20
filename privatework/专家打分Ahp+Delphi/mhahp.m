%僵尸网络的评估指标(5个一级指标，每个一级指标下分为3个二级指标)：
%1.网络的基本模式,分为3种:(1)网络分布模式 (2)网络连接模式 (3)网络控制模式
%2.网络的规模，分为3种:(1)网络数量规模 (2)网络用户规模 (3)网络金额规模
%3.网络的攻击性/窃密性，分为3种:(1)网络攻击方式多样性 (2)网络传播速度 (3)网络权限突破能力
%4.网络隐蔽性，分为3种:(1)网络反查杀功能 (2)网络伪装功能 (3)网络二次传播能力
%5.网络可控性，分为3种:(1)网络的规模可控性 (2)网络攻击的可控性 (3)网络传播的可控性


%重要性评估方式：两两因素互相比较
%1.表示同样重要
%3.表示一个比一个稍微重要
%5.表示一个比一个明显重要
%7.表示一个比一个强烈重要
%9.表示一个比一个极端重要
%(2,4,6,8)为中间值
%倒数 aij=k;aji=(1/k)
%矩阵A 5种因素自身的重要性评估 这里为了方便 假设5种因素的重要性递减
%load scoreweight.mat;
A=[1,3,5,7,9;1/3,1,3,5,7;1/5,1/3,1,3,5;1/7,1/5,1/3,1,3;1/9,1/7,1/5,1/3,1;];
disp('校园区域网络A');
disp(A);
%矩阵B 三个等待测评的网络 对于5种危害因素的得分
B1 = [1,2,7;1/2,1,2;1/7,1/2,1];
B2 = [1,3,4;1/3,1,3;1/4,1/3,1];
B3 = [1,2,5;1/2,1,2;1/5,1/2,1];
B4 = [1,2,6;1/2,1,2;1/6,1/2,1];
B5 = [1,4,9;1/4,1,4;1/9,1/4,1];
disp('校园区域网络B1');
disp(B1);
disp('校园区域网络B2');
disp(B2);
disp('校园区域网络B3');
disp(B3);
disp('校园区域网络B4');
disp(B4);
disp('校园区域网络B5');
disp(B5);

[CIA, CRA, WA] = getW(A);
[CI_B1, CR_B1, W_B1] = getW(B1);
[CI_B2, CR_B2, W_B2] = getW(B2);
[CI_B3, CR_B3, W_B3] = getW(B3);
[CI_B4, CR_B4, W_B4] = getW(B4);
[CI_B5, CR_B5, W_B5] = getW(B5);
disp('校园区域网络WA');
disp(WA);
disp('校园区域网络W_B1');
disp(W_B1);
disp('校园区域网络W_B2');
disp(W_B2);
disp('校园区域网络W_B3');
disp(W_B3);
disp('校园区域网络W_B4');
disp(W_B4);
disp('校园区域网络W_B5');
disp(W_B5);

load SecondIndex1.mat;
W_B1 = W_B1'*analysis;
load SecondIndex2.mat;
W_B2 = W_B2'*analysis;
load SecondIndex3.mat;
W_B3 = W_B3'*analysis;
load SecondIndex4.mat;
W_B4 = W_B4'*analysis;
load SecondIndex5.mat;
W_B5 = W_B5'*analysis;
disp('校园区域网络B1');
disp(W_B1);
disp('校园区域网络B2');
disp(W_B2);
disp('校园区域网络B3');
disp(W_B3);
disp('校园区域网络B4');
disp(W_B4);
disp('校园区域网络B5');
disp(W_B5);

all_result = [W_B1;W_B2;W_B3;W_B4;W_B5]*WA;  %最终得到的值 即为3个不同僵尸网络最终的得分 哪个高哪个就是危害较大的 即安全性差
disp('校园区域网络综合评判：');
disp(all_result);
[a,b] = sort(all_result);
disp('校园区域网络安全性指标评估从低到高为：')
disp(b);