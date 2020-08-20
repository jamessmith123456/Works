%�����������ݼ�

function testSet=create_testSet(G,srcDir,imageName)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��TempĿ¼�¶���ָ�õ�ͼƬ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�����۾�
str=strcat(srcDir,'eye');
str=strcat(str,imageName);
Ieye=imread(str);
Ieye=imresize(Ieye,[10 30]);
%�������
str=strcat(srcDir,'nose');
str=strcat(str,imageName);
Inose=imread(str);
Inose=imresize(Inose,[16 24]);
%�������
str=strcat(srcDir,'mouth');
str=strcat(str,imageName);
Imouth=imread(str);
Imouth=imresize(Imouth,[12 18]);

disp(str);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��ÿ��ROI�ֱ����Gabor������ȡ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%û�н���PCA֮ǰ��ÿ��ROI��Gabor������
% eyeGabor=zeros(1,30*10*40/64);
% noseGabor=zeros(1,24*16*40/64);
% mouthGabor=zeros(1,18*12*40/64);

%�۾��������ݼ�
eyeGabor(1,:)=feature_Extract(Ieye,G,30,10);
%�����������ݼ�
noseGabor(1,:)=feature_Extract(Inose,G,24,16);
%����������ݼ�
mouthGabor(1,:)=feature_Extract(Imouth,G,18,12);
% save mat_eyeGabor eyeGabor;
% save mat_noseGabor noseGabor;
% save mat_mouthGabor mouthGabor;

% %����ԭʼ���ݾ��󣬵�1���Ǹò�������������Ϊѵ����������
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
%��ÿ��ROI�ֱ����PCA��ά������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����ѵ����ʱ�������ֽ�ά��ʽ�����ǵ��ڶ��ַ�ʱ������ֻ�����˵�һ��
% %�۾�ROI
% disp('���ڽ����۾�ROI�Ľ�ά......');
% eyePCAtest = feature_PCA1(eyeGabor,212);
% % eyePCA = feature_PCA1(eyeData,212);
% %����ROI
% disp('���ڽ��б���ROI�Ľ�ά......');
% nosePCAtest = feature_PCA1(noseGabor,212);
% %���ROI
% disp('���ڽ������ROI�Ľ�ά......');
% mouthPCAtest = feature_PCA1(mouthGabor,212);

% testPCA=feature_PCA1([eyeGabor mouthGabor],212);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����ѵ�����ݼ�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROItest=[eyePCAtest nosePCAtest mouthPCAtest];%����������ϵ�һ��ÿ�д���һ����������
% testSet=ROItest';%ֻ��һ��������������һ�д��������
% testSet=testPCA';
testSet=[eyeGabor mouthGabor]';%�����˱��ӵ�����
%����ѵ�����ݼ�
save mat_testSet testSet;
disp('<��ʾ> ѵ�����ݼ��������');
