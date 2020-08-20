function [NET,minP,maxP,minT,maxT] = net_Train(P,T)
% %�������������
% net.trainFcn = 'trainscg';
% net.trainParam.lr = 0.4;
% net.trainParam.epochs = 400;
% net.trainParam.show = 10;
% net.trainParam.goal = 1e-3;
% 
% %ѵ��������
% [outPut,tr] = train(net,P,T);
% save net net
% NET=net;

%ѵ�����ݹ�һ��
[PP,minP,maxP,TT,minT,maxT]=premnmx(P,T);

%  ����һ���µ�ǰ��������   
Net=newff([minP maxP],[50,100,50,7],{'tansig','tansig','tansig','purelin'},'traingdm');

%  ��ǰ�����Ȩֵ����ֵ   
inputWeights=Net.IW{1,1};
inputbias=Net.b{1};
%  ��ǰ�����Ȩֵ����ֵ   
layerWeights=Net.LW{2,1};
layerbias=Net.b{2};

%  ����ѵ������   
Net.trainParam.show = 100;   
Net.trainParam.lr = 0.05;   
Net.trainParam.mc = 0.9;   
Net.trainParam.epochs = 4000;   
Net.trainParam.goal = 1e-1;   


% [P,minp,maxp,T,mint,maxt] = premnmx(P,T); 
% % ��׼������ 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% % ������������� 
% %====���������� 
% Para.Goal = 0.0001; 
% % ����ѵ��Ŀ����� 
% Para.Epochs = 800; 
% % ����ѵ������ 
% Para.LearnRate = 0.1; 
% % ����ѧϰ���� 
% %==== 
% Para.Show = 5; 
% % ����ѵ����ʾ��� 
% Para.InRange = repmat([-1 1],size(P,1),1); 
% % ���������������� 
% Para.Neurons = [size(P,1)*2+1 1]; 
% % �����������Ԫ���� 
% Para.TransferFcn= {'logsig' 'purelin'}; 
% % �������ֵ���� 
% Para.TrainFcn = 'trainlm'; 
% % ����ѵ��������ֵ 
% % traingd : �ݶ��½����򴫲��� 
% % traingda : ����Ӧѧϰ���ʵ��ݶ��½��� 
% % traingdm : ���������ݶ��½��� 
% % traingdx : 
% % ������������Ӧѧϰ���ʵ��ݶ��½��� 
% Para.LearnFcn = 'learngdm'; 
% % ����ѧϰ���� 
% Para.PerformFcn = 'sse'; 
% % ��������� 
% Para.InNum = size(P,1); 
% % ������ά�� 
% Para.IWNum = Para.InNum*Para.Neurons(1); 
% % ����Ȩ�ظ��� 
% Para.LWNum = prod(Para.Neurons); 
% % ��Ȩ�ظ��� 
% Para.BiasNum = sum(Para.Neurons); 
% % ƫ�ø��� 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Net = newff(Para.InRange,Para.Neurons,Para.TransferFcn,... 
% Para.TrainFcn,Para.LearnFcn,Para.PerformFcn); 
% % �������� 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Net.trainParam.show = Para.Show; 
% % ѵ����ʾ�����ֵ 
% Net.trainParam.goal = Para.Goal; 
% % ѵ��Ŀ����ֵ 
% Net.trainParam.lr = Para.LearnRate; 
% % ����ѧϰ���ʸ�ֵ 
% Net.trainParam.epochs = Para.Epochs; 
% % ѵ��������ֵ 
% Net.trainParam.lr = Para.LearnRate; 
% 
% Net.performFcn = Para.PerformFcn; 
% % ������ֵ 




%  ���� TRAINGDM �㷨ѵ�� BP ����   
[Net,tr]=train(Net,PP,TT);  



% %  ����������   
% E = T - A;  
% MSE=mse(E);  

%���������
save mat_Net Net;
NET=Net;

echo off   