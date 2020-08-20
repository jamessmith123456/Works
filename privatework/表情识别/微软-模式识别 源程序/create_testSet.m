%创建测试数据集

function testSet=create_testSet(G,srcDir,imageName)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%从Temp目录下读入分割好的图片
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%读入眼睛
str=strcat(srcDir,'eye');
str=strcat(str,imageName);
Ieye=imread(str);
Ieye=imresize(Ieye,[10 30]);
%读入鼻子
str=strcat(srcDir,'nose');
str=strcat(str,imageName);
Inose=imread(str);
Inose=imresize(Inose,[16 24]);
%读入嘴巴
str=strcat(srcDir,'mouth');
str=strcat(str,imageName);
Imouth=imread(str);
Imouth=imresize(Imouth,[12 18]);

disp(str);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%对每个ROI分别进行Gabor特征提取
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%没有进行PCA之前的每个ROI的Gabor特征集
% eyeGabor=zeros(1,30*10*40/64);
% noseGabor=zeros(1,24*16*40/64);
% mouthGabor=zeros(1,18*12*40/64);

%眼睛特征数据集
eyeGabor(1,:)=feature_Extract(Ieye,G,30,10);
%鼻子特征数据集
noseGabor(1,:)=feature_Extract(Inose,G,24,16);
%嘴巴特征数据集
mouthGabor(1,:)=feature_Extract(Imouth,G,18,12);
% save mat_eyeGabor eyeGabor;
% save mat_noseGabor noseGabor;
% save mat_mouthGabor mouthGabor;

% %构造原始数据矩阵，第1行是该测试样本，其他为训练集中样本
% load('mat_eyeData.mat');
% load('mat_noseData.mat');
% load('mat_mouthData.mat');
% 
% for i=2:213
%     eyeGabor(i,:)=eyeData(i,:);
%     noseGabor(i,:)=noseData(i,:);
%     mouthGabor(i,:)=mouthData(i,:);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%对每个ROI分别进行PCA降维并保存
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%构造训练集时，有两种降维方式，考虑到第二种费时，这里只采用了第一种
% %眼睛ROI
% disp('正在进行眼睛ROI的降维......');
% eyePCAtest = feature_PCA1(eyeGabor,212);
% % eyePCA = feature_PCA1(eyeData,212);
% %鼻子ROI
% disp('正在进行鼻子ROI的降维......');
% nosePCAtest = feature_PCA1(noseGabor,212);
% %嘴巴ROI
% disp('正在进行嘴巴ROI的降维......');
% mouthPCAtest = feature_PCA1(mouthGabor,212);

% testPCA=feature_PCA1([eyeGabor mouthGabor],212);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%创建训练数据集
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROItest=[eyePCAtest nosePCAtest mouthPCAtest];%三个矩阵组合到一起，每行代表一个样本数据
% testSet=ROItest';%只有一个测试样本，第一列代表此样本
% testSet=testPCA';
testSet=[eyeGabor mouthGabor]';%忽略了鼻子的数据
%保存训练数据集
save mat_testSet testSet;
disp('<提示> 训练数据集保存完毕');
