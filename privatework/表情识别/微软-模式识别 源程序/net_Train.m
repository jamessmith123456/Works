function [NET,minP,maxP,minT,maxT] = net_Train(P,T)
% %设置神经网络参数
% net.trainFcn = 'trainscg';
% net.trainParam.lr = 0.4;
% net.trainParam.epochs = 400;
% net.trainParam.show = 10;
% net.trainParam.goal = 1e-3;
% 
% %训练神经网络
% [outPut,tr] = train(net,P,T);
% save net net
% NET=net;

%训练数据归一化
[PP,minP,maxP,TT,minT,maxT]=premnmx(P,T);

%  创建一个新的前向神经网络   
Net=newff([minP maxP],[50,100,50,7],{'tansig','tansig','tansig','purelin'},'traingdm');

%  当前输入层权值和阈值   
inputWeights=Net.IW{1,1};
inputbias=Net.b{1};
%  当前网络层权值和阈值   
layerWeights=Net.LW{2,1};
layerbias=Net.b{2};

%  设置训练参数   
Net.trainParam.show = 100;   
Net.trainParam.lr = 0.05;   
Net.trainParam.mc = 0.9;   
Net.trainParam.epochs = 4000;   
Net.trainParam.goal = 1e-1;   


% [P,minp,maxp,T,mint,maxt] = premnmx(P,T); 
% % 标准化数据 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% % 神经网络参数设置 
% %====可以修正处 
% Para.Goal = 0.0001; 
% % 网络训练目标误差 
% Para.Epochs = 800; 
% % 网络训练代数 
% Para.LearnRate = 0.1; 
% % 网络学习速率 
% %==== 
% Para.Show = 5; 
% % 网络训练显示间隔 
% Para.InRange = repmat([-1 1],size(P,1),1); 
% % 网络的输入变量区间 
% Para.Neurons = [size(P,1)*2+1 1]; 
% % 网络后两层神经元配置 
% Para.TransferFcn= {'logsig' 'purelin'}; 
% % 各层的阈值函数 
% Para.TrainFcn = 'trainlm'; 
% % 网络训练函数赋值 
% % traingd : 梯度下降后向传播法 
% % traingda : 自适应学习速率的梯度下降法 
% % traingdm : 带动量的梯度下降法 
% % traingdx : 
% % 带动量，自适应学习速率的梯度下降法 
% Para.LearnFcn = 'learngdm'; 
% % 网络学习函数 
% Para.PerformFcn = 'sse'; 
% % 网络的误差函数 
% Para.InNum = size(P,1); 
% % 输入量维数 
% Para.IWNum = Para.InNum*Para.Neurons(1); 
% % 输入权重个数 
% Para.LWNum = prod(Para.Neurons); 
% % 层权重个数 
% Para.BiasNum = sum(Para.Neurons); 
% % 偏置个数 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Net = newff(Para.InRange,Para.Neurons,Para.TransferFcn,... 
% Para.TrainFcn,Para.LearnFcn,Para.PerformFcn); 
% % 建立网络 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Net.trainParam.show = Para.Show; 
% % 训练显示间隔赋值 
% Net.trainParam.goal = Para.Goal; 
% % 训练目标误差赋值 
% Net.trainParam.lr = Para.LearnRate; 
% % 网络学习速率赋值 
% Net.trainParam.epochs = Para.Epochs; 
% % 训练代数赋值 
% Net.trainParam.lr = Para.LearnRate; 
% 
% Net.performFcn = Para.PerformFcn; 
% % 误差函数赋值 




%  调用 TRAINGDM 算法训练 BP 网络   
[Net,tr]=train(Net,PP,TT);  



% %  计算仿真误差   
% E = T - A;  
% MSE=mse(E);  

%保存分类器
save mat_Net Net;
NET=Net;

echo off   