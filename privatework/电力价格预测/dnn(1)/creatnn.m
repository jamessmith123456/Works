function [dnn,parameter] = creatnn(K)
%UNTITLED6 此处显示有关此函数的摘要
% parameter 是结构体，包括参数：
%                       learning_rate: 学习率
%                       momentum： 动量系数,一般为0.5，0.9，0.99
%                       attenuation_rate： 衰减系数
%                       delta：稳定数值
%                       step: 步长 一般为 0.001
%                       method: 方法{'SGD','mSGD','nSGD','AdaGrad','RMSProp','nRMSProp','Adam'}
L = size(K.a,2);
for i = 1:L-1
    dnn{i}.W = unifrnd(-sqrt(6/(K.a(i)+K.a(i+1))),sqrt(6/(K.a(i)+K.a(i+1))),K.a(i+1),K.a(i));
    % dnn{i}.W = normrnd(0,0.1,K.a(i+1),K.a(i));
    dnn{i}.function = K.f{i};
    dnn{i}.b = 0.01*ones(K.a(i+1),1);
end
    parameter.learning_rate = 0.01;
    parameter.momentum = 0.9;
    parameter.attenuation_rate = 0.9;
    parameter.delta = 1e-6;
    parameter.step = 0.001;
    parameter.method = "SGD";
    parameter.beta1 = 0.9;
    parameter.beta2 = 0.999;
end

