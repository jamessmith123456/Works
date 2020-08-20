% 1.用matlab训练一个自己的卷积神经网络中的Alexnet模型，
% 就识别花的种类（数据集的花的种类有十几种，每种80张图片）.
% 不要求训练完的网络最后正确率很高。主要是网络要能跑就行。


% 2.建立深度神经网络独立成分初始化模型，提取神经网络多层次特征，
% 以期获得更加优秀的目标分类和识别神经网络。在训练好的Alexnet模型上，
% 加入ICA的运用（对立成分分析），训练一个新的神经网络，要能跑，同样不要求很高的正确率。网络要能跑。
% 最后要给我一个传统的Alexnet模型，和一个用了ICA的修改过的Alexnet模型。

net = alexnet;
layers = net.Layers(1:end-3);
new_layers = [layers
%     net.Layers(1:end-3)
%               myICA()
%               net.Layers(21:22)
              fullyConnectedLayer(12,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)       
              softmaxLayer('name','soft')
              classificationLayer('name','classify')
              ];
imds = imageDatastore('F:\识别花朵\Flowers', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
labelCount = countEachLabel(imds);
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized'); 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ... %雾草这里是个坑！还好有经验
    'MiniBatchSize',3, ...
    'MaxEpochs',40, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
% ops = trainingOptions('sgdm', ...
%                       'InitialLearnRate',0.001, ...
%                       'ValidationData',imageTest, ...
%                       'Plots','training-progress', ...
%                       'MiniBatchSize',5, ...
%                       'MaxEpochs',5,...
%                       'ValidationPatience',Inf,...
%                       'Verbose',false);
% 使用训练数据训练网络
gpuDevice(1)
net_train = trainNetwork(imdsTrain,new_layers,options);

% 对验证图像进行分类并计算准确度
YPred = classify(net_train,imdsTest);
YValidation = imdsTest.Labels;
% accuracy = sum(YPred == YValidation)/numel(YValidation);




% name = 'F:\识别花朵\Flowers\bluebell\image_0241.jpg';
% image_val = imread(name);
% [label,conf] = classify(net_train,image_val);
% soft = activations(net_train,image_val,'fc7');%824464 
% YPred = activations(net_train,imdsTest,'soft');
% %单独显示第一层卷积的16张图像
% act1 = activations(net_train,image_val,'conv1');%824464 
% sz = size(act1);
% act1 = reshape(act1,[227 227 1 16]);
% imshow(I)




%下面开始第二部分,把这个网络的无监督学习的部分替换成ICA（独立约束性学习）来算权重w
%然后有监督加ICA总的跑一遍，不要求正确率
%AlexNet网络很深，conv卷积层的目的是为了提取特征，不同的conv层提取的特征数量、内容都是不一样的
%要想充分利用ICA，就应该在网络的无监督部分的最底层，也就是无监督网络的最后一层，提取特征
%也就是对全连接层：fully connected layer

%下面是对所有的训练图片，都输入到上面的网络里，获取所有的训练样本的fully connected层参数
%训练样本有m个,该层权重参数有n个,则提取出来的所有的权重参数为m*n的矩阵
AllWeights = activations(net_train,imds,'fc7'); %m=728,728个训练样本  n=4096该层共4096个权重参数

%下面进行ICA独立性成分分析，其中q是需要自己设定的超参数，我暂定1000
q = 1000;
Mdl = rica(AllWeights,q,'IterationLimit',100)
newfeature = AllWeights*Mdl.TransformWeights; %得到的newfeature是m*q的矩阵,每行表示一个样本


mkdir('ICA_NewFeatures');
%在ICA_NewFeatures这个文件夹下,按照类别创建12个子文件夹,并将各个样本按照子文件夹分类
templabel = cellstr(imds.Labels);
AllLabel = cellstr(unique(imds.Labels));
mkdir('ICA_NewFeatures\');
for i=1:size(AllLabel,1)
    mkdir(['ICA_NewFeatures\',AllLabel{i}]);
end

names = cellstr(imds.Files);
for i=1:size(newfeature,1)
    temp = reshape(newfeature(i,:),1,1000,1);
    path = ['F:\识别花朵\ICA_NewFeatures\',templabel{i},'\',names{i}(end-13:end)];
    imwrite(temp,path);
end

clear;
clc;
%为经过ICA独立性分析的特征单独构建一个网络,因为此时每个样本的特征是二维的，1*1000
%且其含义本身就代表经过卷积操作得到的特征，所以没必要再添加卷积层了
%直接添加全连接层，也就是构建普通的神经网络进行分类即可
imds2 = imageDatastore('F:\识别花朵\ICA_NewFeatures', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
labelCount = countEachLabel(imds2);
[imdsTrain2,imdsTest2] = splitEachLabel(imds2,0.7,'randomized'); 
inputSize = [1 1000 1];
numClasses = 12;

layers = [
    imageInputLayer(inputSize)
    fullyConnectedLayer(500)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'ValidationData',imdsTest2, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net_train2 = trainNetwork(imdsTrain2,layers,options);
YPred = classify(net_train2,imdsTest2);
YValidation = imdsTest2.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
