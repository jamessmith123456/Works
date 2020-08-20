%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%创建Gabor训练数据集
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [trainSet,P,T]=create_trainSet(G,samples_Data)
%没有进行PCA之前的每个ROI的Gabor特征集
% eyeData=zeros(213,30*10*40/64);%做了三次强制特征去除
% noseData=zeros(213,24*16*40/64);
% mouthData=zeros(213,18*12*40/64);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%对每个ROI分别进行Gabor特征提取
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('正在对训练集的ROI进行Gabor特征提取\n\n');
for i=1:213
    %眼睛特征数据集
    eyeData(i,:)=feature_Extract(samples_Data{i,9},G,30,10);
    %鼻子特征数据集
    noseData(i,:)=feature_Extract(samples_Data{i,10},G,24,16);
    %嘴巴特征数据集
    mouthData(i,:)=feature_Extract(samples_Data{i,11},G,18,12);
end
%保存原始特征集
% save mat_eyeData eyeData
% save mat_noseData noseData
% save mat_mouthData mouthData

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%对每个ROI分别进行PCA降维并保存
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('<开始> 对原始Gabor特征集进行降维\n\n');
% fprintf('降维方式有下面两种：\n');
% fprintf('1 Matlab工具箱自带PCA降维函数进行降维，速度较快(几分钟之内)\n');
% fprintf('2 基于累计方差贡献率的主元分析降维，速度比较漫长(一个小时)\n');
% askStr=input('请选择数据降维方式(输入降维方式序号) :','s');
% if askStr=='1'
%     fprintf('正在使用方式 <1> 进行降维......\n\n');
% %     %眼睛ROI
% %     eyePCA = feature_PCA1(eyeData,212);
% %     save mat_eyePCA eyePCA
% %     %鼻子ROI
% %     nosePCA = feature_PCA1(noseData,212);
% %     save mat_nosePCA nosePCA
% %     %嘴巴ROI
% %     mouthPCA = feature_PCA1(mouthData,212);
% %     save mat_mouthPCA mouthPCA
%     trainPCA=feature_PCA1([eyeData mouthData],212);
% elseif askStr=='2'
%     fprintf('正在使用方式 <2> 进行降维......\n\n');
%     %眼睛ROI
%     eyePCA = feature_PCA2(eyeData,0.85);
%     save mat_eyePCA eyePCA
%     %鼻子ROI
%     nosePCA = feature_PCA2(noseData,0.85);
%     save mat_nosePCA nosePCA
%     %嘴巴ROI
%     mouthPCA = feature_PCA2(mouthData,0.85);
%     save mat_mouthPCA mouthPCA
%     fprintf('PCA降维之后的Gabor特征集保存完毕\n\n');
% else
%     msgbox('没有正确选择降维方式，请重新运行主程序！');
%     return;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%创建训练数据集
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROIset=[eyePCA nosePCA mouthPCA];
% trainSet=cell(213,7);
% for i=1:213
%     trainSet{i,1}=samples_Data{i,1};%样本序号
%     trainSet{i,2}=samples_Data{i,2};%人名
%     trainSet{i,3}=samples_Data{i,3};%人名数字表示
%     trainSet{i,4}=samples_Data{i,4};%表情名
%     trainSet{i,5}=samples_Data{i,5};%表情数字表示
%     trainSet{i,6}=ROIset(i,:);%PCA降维之后的数据集
% 
% end
% save mat_trainSet trainSet;%保存训练数据集

%创建神经网络输入的训练样例数据集
trainSet=[eyeData mouthData];
A=eye(7);
for i=1:213
%     P(:,i)=trainPCA(i,:)';%cell类型转为矩阵类型后，转置别忘记！输入向量是列向量
    P(:,i)=trainSet(i,:)';%cell类型转为矩阵类型后，转置别忘记！输入向量是列向量
%     T(:,i)=samples_Data{i,5};%标签数据
      T(:,i)=A(:,samples_Data{i,5});
end


save mat_trainSetP P;
save mat_trainSetT T;
fprintf('<保存> 训练集已保存完毕\n\n');

