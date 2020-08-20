function [a, b] = mhahphanshu(A,B1,B2,B3,B4,B5,class)
% A=[1,3,5,7,9;1/3,1,3,5,7;1/5,1/3,1,3,5;1/7,1/5,1/3,1,3;1/9,1/7,1/5,1/3,1;];
% 
% B1 = [1,2,7;1/2,1,2;1/7,1/2,1];
% B2 = [1,3,4;1/3,1,3;1/4,1/3,1];
% B3 = [1,2,5;1/2,1,2;1/5,1/2,1];
% B4 = [1,2,6;1/2,1,2;1/6,1/2,1];
% B5 = [1,4,9;1/4,1,4;1/9,1/4,1];

[CIA, CRA, WA] = getW(A);
[CI_B1, CR_B1, W_B1] = getW(B1);
[CI_B2, CR_B2, W_B2] = getW(B2);
[CI_B3, CR_B3, W_B3] = getW(B3);
[CI_B4, CR_B4, W_B4] = getW(B4);
[CI_B5, CR_B5, W_B5] = getW(B5);
if class==1
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
end

if class==2
    load SecondIndex1-2.mat;
    W_B1 = W_B1'*analysis;
    load SecondIndex2-2.mat;
    W_B2 = W_B2'*analysis;
    load SecondIndex3-2.mat;
    W_B3 = W_B3'*analysis;
    load SecondIndex4-2.mat;
    W_B4 = W_B4'*analysis;
    load SecondIndex5-2.mat;
    W_B5 = W_B5'*analysis;
end

if class==3
    load SecondIndex1-3.mat;
    W_B1 = W_B1'*analysis;
    load SecondIndex2-3.mat;
    W_B2 = W_B2'*analysis;
    load SecondIndex3-3.mat;
    W_B3 = W_B3'*analysis;
    load SecondIndex4-3.mat;
    W_B4 = W_B4'*analysis;
    load SecondIndex5-3.mat;
    W_B5 = W_B5'*analysis;
end


all_result = [W_B1;W_B2;W_B3;W_B4;W_B5]*WA;  %最终得到的值 即为3个不同僵尸网络最终的得分 哪个高哪个就是危害较大的 即安全性差
[a,b] = sort(all_result);
end

