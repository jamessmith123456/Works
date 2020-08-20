% ����֮ǰ��ͳ��ALEXNETÿ�����㲻�����о������
% �����һ��ľ��������Ϊ������Ȼ���м�ľ��������Լ�����㷨�õ��ģ�96����СΪ11*11*3�ľ���ˡ�
% ��һ��ICA����������滻ԭ��Լ�����㷨�õ��ľ���ˡ�
% ͬ��ڶ���ICA�Ľ���滻�ڶ��������еľ���ˣ��Դ����ơ�
% ����˵ICA��һ���㷨��5��ICA�����������ģ��൱������Alexnet������5��ICA��
% ���Ӹ�������������С����ICA����д��������������ó�����
net = alexnet;
layers = net.Layers(1:end-3);
%���Ȳ������ݼ�
imds = imageDatastore('F:\ʶ�𻨶�\Flowers', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
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


%���������������ݣ���������2ά����
[m,mm,mmm,mmmm] = size(AllData);
AllData2 = zeros(m,mm*mmm*mmmm);
for i=1:m
    temp = AllData(i,:,:,:);
    temp2 = reshape(temp,1,mm*mmm*mmmm);
    AllData2(i,:) = temp2;
end
    
q = 1000;
Mdl = rica(AllData2(:,1:4000),q,'IterationLimit',100)
newfeature = AllData2(:,1:4000)*Mdl.TransformWeights; %�õ���newfeature��m*q�ľ���,ÿ�б�ʾһ������

index = randperm(m);
index2 = index(1:96);
weighttemp = newfeature(index2,1:363);
weight = reshape(weighttemp,11,11,3,96);

tmp_net = net.saveobj;
tmp_net.Layers(2,1).Weights = single(weight); %�滻��һ��conv1
net = net.loadobj(tmp_net);

gpuDevice(1)
pool1 = activations(net,imdsTrain,'pool1');

q=1200;
Mdl = rica(pool1(:,1:4000),q,'IterationLimit',100)
newfeature = pool1(:,1:4000)*Mdl.TransformWeights;
index = randperm(728); %728��TrainData��������
index2 = index(1:256);
weighttemp = newfeature(index2,1:1200);
weight = reshape(weighttemp,5,5,48,256); 
tmp_net = net.saveobj;
tmp_net.Layers(6,1).Weights = single(weight); %�滻�ڶ���conv2
net = net.loadobj(tmp_net);
gpuDevice(1)
pool2 = activations(net,imdsTrain,'pool2');


q=2304;
Mdl = rica(pool2(:,1:4000),q,'IterationLimit',100)
newfeature = pool2(:,1:4000)*Mdl.TransformWeights;
index = randperm(728); %728��TrainData��������
index2 = index(1:384);
weighttemp = newfeature(index2,1:2304);
weight = reshape(weighttemp,3,3,256,384);
tmp_net = net.saveobj;
tmp_net.Layers(10,1).Weights = single(weight); %�滻������conv3
net = net.loadobj(tmp_net);
gpuDevice(1)
relu3 = activations(net,imdsTrain,'relu3');


q=1728;
Mdl = rica(relu3(:,1:4000),q,'IterationLimit',100)
newfeature = relu3(:,1:4000)*Mdl.TransformWeights;
index = randperm(728); %728��TrainData��������
index2 = index(1:384);
weighttemp = newfeature(index2,1:1728);
weight = reshape(weighttemp,3,3,192,384);
tmp_net = net.saveobj;
tmp_net.Layers(12,1).Weights = single(weight); %�滻���Ĳ�conv4
net = net.loadobj(tmp_net);
gpuDevice(1)
relu4 = activations(net,imdsTrain,'relu4');


q=1728;
Mdl = rica(relu4(:,1:4000),q,'IterationLimit',100)
newfeature = relu4(:,1:4000)*Mdl.TransformWeights;
index = randperm(728); %728��TrainData��������
index2 = index(1:256);
weighttemp = newfeature(index2,1:1728);
weight = reshape(weighttemp,3,3,192,256);
tmp_net = net.saveobj;
tmp_net.Layers(14,1).Weights = single(weight); %�滻�����conv5
net = net.loadobj(tmp_net);
gpuDevice(1)
relu4 = activations(net,imdsTrain,'relu4');


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ... %��������Ǹ��ӣ������о���
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

% ����֤ͼ����з��ಢ����׼ȷ��
YPred = classify(net2,imdsTest);
YValidation = imdsTest.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
