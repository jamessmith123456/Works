% 就是之前传统的ALEXNET每层卷积层不都是有卷积核吗？
% 比如第一层的卷积层输入为样本，然后中间的卷积核是用约束性算法得到的，96个大小为11*11*3的卷积核。
% 第一层ICA的输出就是替换原来约束性算法得到的卷积核。
% 同理第二层ICA的结果替换第二层卷积层中的卷积核，以此类推。
% 就是说ICA是一种算法，5层ICA并不是相连的，相当于是在Alexnet中运用5次ICA。
% 最后接个三层神经网络就行。这个ICA不用写进神经网络里，单独拿出来做
net = alexnet;
layers = net.Layers(1:end-3);
%首先产生数据集
imds = imageDatastore('F:\识别花朵\Flowers', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
labelCount = countEachLabel(imds);
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
AllData = [];
for i=1:size(imds.Files,1)
    filename = imds.Files{i};
    AllData(i,:,:,:)=imread(filename);
end
TrainData = [];
for i=1:size(imdsTrain.Files,1)
    filename = imdsTrain.Files{i};
    TrainData(i,:,:,:)=imread(filename);
end
TestData = [];
for i=1:size(imdsTest.Files,1)
    filename = imdsTest.Files{i};
    TestData(i,:,:,:)=imread(filename);
end
save myshiyan.mat AllData TrainData TestData;


%咱们针对上面的数据，把它化成2维矩阵
[m,mm,mmm,mmmm] = size(AllData);
AllData2 = zeros(m,mm*mmm*mmmm);
for i=1:m
    temp = AllData(i,:,:,:);
    temp2 = reshape(temp,1,mm*mmm*mmmm);
    AllData2(i,:) = temp2;
end
    
q = 1000;
Mdl = rica(AllData2(:,1:4000),q,'IterationLimit',100)
newfeature = AllData2(:,1:4000)*Mdl.TransformWeights; %得到的newfeature是m*q的矩阵,每行表示一个样本

index = randperm(m);
index2 = index(1:96);
weighttemp = newfeature(index2,1:363);
weight = reshape(weighttemp,11,11,3,96);

tmp_net = net.saveobj;
tmp_net.Layers(2,1).Weights = single(weight); %替换第一层conv1
net = net.loadobj(tmp_net);

gpuDevice(1)
pool1 = activations(net,imdsTrain,'pool1');

q=1200;
Mdl = rica(pool1(:,1:4000),q,'IterationLimit',100)
newfeature = pool1(:,1:4000)*Mdl.TransformWeights;
index = randperm(728); %728是TrainData的样本数
index2 = index(1:256);
weighttemp = newfeature(index2,1:1200);
weight = reshape(weighttemp,5,5,48,256); 
tmp_net = net.saveobj;
tmp_net.Layers(6,1).Weights = single(weight); %替换第二层conv2
net = net.loadobj(tmp_net);
gpuDevice(1)
pool2 = activations(net,imdsTrain,'pool2');


q=2304;
Mdl = rica(pool2(:,1:4000),q,'IterationLimit',100)
newfeature = pool2(:,1:4000)*Mdl.TransformWeights;
index = randperm(728); %728是TrainData的样本数
index2 = index(1:384);
weighttemp = newfeature(index2,1:2304);
weight = reshape(weighttemp,3,3,256,384);
tmp_net = net.saveobj;
tmp_net.Layers(10,1).Weights = single(weight); %替换第三层conv3
net = net.loadobj(tmp_net);
gpuDevice(1)
relu3 = activations(net,imdsTrain,'relu3');


q=1728;
Mdl = rica(relu3(:,1:4000),q,'IterationLimit',100)
newfeature = relu3(:,1:4000)*Mdl.TransformWeights;
index = randperm(728); %728是TrainData的样本数
index2 = index(1:384);
weighttemp = newfeature(index2,1:1728);
weight = reshape(weighttemp,3,3,192,384);
tmp_net = net.saveobj;
tmp_net.Layers(12,1).Weights = single(weight); %替换第四层conv4
net = net.loadobj(tmp_net);
gpuDevice(1)
relu4 = activations(net,imdsTrain,'relu4');


q=1728;
Mdl = rica(relu4(:,1:4000),q,'IterationLimit',100)
newfeature = relu4(:,1:4000)*Mdl.TransformWeights;
index = randperm(728); %728是TrainData的样本数
index2 = index(1:256);
weighttemp = newfeature(index2,1:1728);
weight = reshape(weighttemp,3,3,192,256);
tmp_net = net.saveobj;
tmp_net.Layers(14,1).Weights = single(weight); %替换第五层conv5
net = net.loadobj(tmp_net);
gpuDevice(1)
relu4 = activations(net,imdsTrain,'relu4');


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ... %雾草这里是个坑！还好有经验
    'MiniBatchSize',5, ...
    'MaxEpochs',60, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
layers = net.Layers(1:end-3);
new_layers = [layers
              fullyConnectedLayer(12,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
              softmaxLayer('name','soft')
              classificationLayer('name','classify')
              ];
gpuDevice(1)
net2 = trainNetwork(imdsTrain,new_layers,options);

% 对验证图像进行分类并计算准确度
YPred = classify(net2,imdsTest);
YValidation = imdsTest.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
