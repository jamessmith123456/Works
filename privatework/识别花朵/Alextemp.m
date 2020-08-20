% 1.��matlabѵ��һ���Լ��ľ���������е�Alexnetģ�ͣ�
% ��ʶ�𻨵����ࣨ���ݼ��Ļ���������ʮ���֣�ÿ��80��ͼƬ��.
% ��Ҫ��ѵ��������������ȷ�ʺܸߡ���Ҫ������Ҫ���ܾ��С�


% 2.�����������������ɷֳ�ʼ��ģ�ͣ���ȡ���������������
% ���ڻ�ø��������Ŀ������ʶ�������硣��ѵ���õ�Alexnetģ���ϣ�
% ����ICA�����ã������ɷַ�������ѵ��һ���µ������磬Ҫ���ܣ�ͬ����Ҫ��ܸߵ���ȷ�ʡ�����Ҫ���ܡ�
% ���Ҫ����һ����ͳ��Alexnetģ�ͣ���һ������ICA���޸Ĺ���Alexnetģ�͡�

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
imds = imageDatastore('F:\ʶ�𻨶�\Flowers', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
labelCount = countEachLabel(imds);
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized'); 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ... %��������Ǹ��ӣ������о���
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
% ʹ��ѵ������ѵ������
gpuDevice(1)
net_train = trainNetwork(imdsTrain,new_layers,options);

% ����֤ͼ����з��ಢ����׼ȷ��
YPred = classify(net_train,imdsTest);
YValidation = imdsTest.Labels;
% accuracy = sum(YPred == YValidation)/numel(YValidation);




% name = 'F:\ʶ�𻨶�\Flowers\bluebell\image_0241.jpg';
% image_val = imread(name);
% [label,conf] = classify(net_train,image_val);
% soft = activations(net_train,image_val,'fc7');%824464 
% YPred = activations(net_train,imdsTest,'soft');
% %������ʾ��һ������16��ͼ��
% act1 = activations(net_train,image_val,'conv1');%824464 
% sz = size(act1);
% act1 = reshape(act1,[227 227 1 16]);
% imshow(I)




%���濪ʼ�ڶ�����,�����������޼ලѧϰ�Ĳ����滻��ICA������Լ����ѧϰ������Ȩ��w
%Ȼ���мල��ICA�ܵ���һ�飬��Ҫ����ȷ��
%AlexNet������conv������Ŀ����Ϊ����ȡ��������ͬ��conv����ȡ���������������ݶ��ǲ�һ����
%Ҫ��������ICA����Ӧ����������޼ල���ֵ���ײ㣬Ҳ�����޼ල��������һ�㣬��ȡ����
%Ҳ���Ƕ�ȫ���Ӳ㣺fully connected layer

%�����Ƕ����е�ѵ��ͼƬ�������뵽������������ȡ���е�ѵ��������fully connected�����
%ѵ��������m��,�ò�Ȩ�ز�����n��,����ȡ���������е�Ȩ�ز���Ϊm*n�ľ���
AllWeights = activations(net_train,imds,'fc7'); %m=728,728��ѵ������  n=4096�ò㹲4096��Ȩ�ز���

%�������ICA�����Գɷַ���������q����Ҫ�Լ��趨�ĳ����������ݶ�1000
q = 1000;
Mdl = rica(AllWeights,q,'IterationLimit',100)
newfeature = AllWeights*Mdl.TransformWeights; %�õ���newfeature��m*q�ľ���,ÿ�б�ʾһ������


mkdir('ICA_NewFeatures');
%��ICA_NewFeatures����ļ�����,������𴴽�12�����ļ���,�������������������ļ��з���
templabel = cellstr(imds.Labels);
AllLabel = cellstr(unique(imds.Labels));
mkdir('ICA_NewFeatures\');
for i=1:size(AllLabel,1)
    mkdir(['ICA_NewFeatures\',AllLabel{i}]);
end

names = cellstr(imds.Files);
for i=1:size(newfeature,1)
    temp = reshape(newfeature(i,:),1,1000,1);
    path = ['F:\ʶ�𻨶�\ICA_NewFeatures\',templabel{i},'\',names{i}(end-13:end)];
    imwrite(temp,path);
end

clear;
clc;
%Ϊ����ICA�����Է�����������������һ������,��Ϊ��ʱÿ�������������Ƕ�ά�ģ�1*1000
%���京�屾��ʹ�������������õ�������������û��Ҫ����Ӿ������
%ֱ�����ȫ���Ӳ㣬Ҳ���ǹ�����ͨ����������з��༴��
imds2 = imageDatastore('F:\ʶ�𻨶�\ICA_NewFeatures', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
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
