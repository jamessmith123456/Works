function [dnn,parameter] = creatnn(K)
%UNTITLED6 �˴���ʾ�йش˺�����ժҪ
% parameter �ǽṹ�壬����������
%                       learning_rate: ѧϰ��
%                       momentum�� ����ϵ��,һ��Ϊ0.5��0.9��0.99
%                       attenuation_rate�� ˥��ϵ��
%                       delta���ȶ���ֵ
%                       step: ���� һ��Ϊ 0.001
%                       method: ����{'SGD','mSGD','nSGD','AdaGrad','RMSProp','nRMSProp','Adam'}
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

