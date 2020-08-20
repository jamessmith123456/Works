% 类比Alexnet的前五层,不需要用网络实现
% 最后的输出接上两个全连接层,一个output层,不要求最后的识别准确率
% Alexnet有5层,每一层都有权重W,原本的权重是采用稀疏约束得到的,现在就要用5层ICA来实现原来网络的前五层卷积层
% ICA与ICA之间的处理是难点(也就是中间层的实现是难点吧)

net = alexnet;
layers = net.Layers(1:end-3)

imds = imageDatastore('F:\识别花朵\Flowers', 'IncludeSubfolders', true, 'labelsource', 'foldernames');
labelCount = countEachLabel(imds);
% 训练集测试集按照7-3划分
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');


%数据输入层 227*227*3
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

% 读取的有点慢，存储下来，供后面实验用
save myshiyan.mat AllData TrainData TestData;

%下面进行ICA独立性成分分析，其中q是需要自己设定的超参数，我暂定1000
q = 1000;
Mdl = rica(AllWeights,q,'IterationLimit',100)
newfeature = AllWeights*Mdl.TransformWeights; %得到的newfeature是m*q的矩阵,每行表示一个样本


%  1   'data'    Image Input                   227x227x3 images with 'zerocenter' normalization
%  2   'conv1'   Convolution                   96 11x11x3 convolutions with stride [4  4] and padding [0  0  0  0]
%  3   'relu1'   ReLU                          ReLU
%  4   'norm1'   Cross Channel Normalization   cross channel normalization with 5 channels per element
%  5   'pool1'   Max Pooling                   3x3 max pooling with stride [2  2] and padding [0  0  0  0]
%  6   'conv2'   Convolution                   256 5x5x48 convolutions with stride [1  1] and padding [2  2  2  2]
%  7   'relu2'   ReLU                          ReLU
%  8   'norm2'   Cross Channel Normalization   cross channel normalization with 5 channels per element
%  9   'pool2'   Max Pooling                   3x3 max pooling with stride [2  2] and padding [0  0  0  0]
% 10   'conv3'   Convolution                   384 3x3x256 convolutions with stride [1  1] and padding [1  1  1  1]
% 11   'relu3'   ReLU                          ReLU
% 12   'conv4'   Convolution                   384 3x3x192 convolutions with stride [1  1] and padding [1  1  1  1]
% 13   'relu4'   ReLU                          ReLU
% 14   'conv5'   Convolution                   256 3x3x192 convolutions with stride [1  1] and padding [1  1  1  1]
% 15   'relu5'   ReLU                          ReLU
% 16   'pool5'   Max Pooling                   3x3 max pooling with stride [2  2] and padding [0  0  0  0]
% 17   'fc6'     Fully Connected               4096 fully connected layer
% 18   'relu6'   ReLU                          ReLU
% 19   'drop6'   Dropout                       50% dropout
% 20   'fc7'     Fully Connected               4096 fully connected layer
% 21   'relu7'   ReLU                          ReLU
% 22   'drop7'   Dropout                       50% dropout