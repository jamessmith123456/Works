%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����Gaborѵ�����ݼ�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [trainSet,P,T]=create_trainSet(G,samples_Data)
%û�н���PCA֮ǰ��ÿ��ROI��Gabor������
% eyeData=zeros(213,30*10*40/64);%��������ǿ������ȥ��
% noseData=zeros(213,24*16*40/64);
% mouthData=zeros(213,18*12*40/64);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��ÿ��ROI�ֱ����Gabor������ȡ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('���ڶ�ѵ������ROI����Gabor������ȡ\n\n');
for i=1:213
    %�۾��������ݼ�
    eyeData(i,:)=feature_Extract(samples_Data{i,9},G,30,10);
    %�����������ݼ�
    noseData(i,:)=feature_Extract(samples_Data{i,10},G,24,16);
    %����������ݼ�
    mouthData(i,:)=feature_Extract(samples_Data{i,11},G,18,12);
end
%����ԭʼ������
% save mat_eyeData eyeData
% save mat_noseData noseData
% save mat_mouthData mouthData

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��ÿ��ROI�ֱ����PCA��ά������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('<��ʼ> ��ԭʼGabor���������н�ά\n\n');
% fprintf('��ά��ʽ���������֣�\n');
% fprintf('1 Matlab�������Դ�PCA��ά�������н�ά���ٶȽϿ�(������֮��)\n');
% fprintf('2 �����ۼƷ�����ʵ���Ԫ������ά���ٶȱȽ�����(һ��Сʱ)\n');
% askStr=input('��ѡ�����ݽ�ά��ʽ(���뽵ά��ʽ���) :','s');
% if askStr=='1'
%     fprintf('����ʹ�÷�ʽ <1> ���н�ά......\n\n');
% %     %�۾�ROI
% %     eyePCA = feature_PCA1(eyeData,212);
% %     save mat_eyePCA eyePCA
% %     %����ROI
% %     nosePCA = feature_PCA1(noseData,212);
% %     save mat_nosePCA nosePCA
% %     %���ROI
% %     mouthPCA = feature_PCA1(mouthData,212);
% %     save mat_mouthPCA mouthPCA
%     trainPCA=feature_PCA1([eyeData mouthData],212);
% elseif askStr=='2'
%     fprintf('����ʹ�÷�ʽ <2> ���н�ά......\n\n');
%     %�۾�ROI
%     eyePCA = feature_PCA2(eyeData,0.85);
%     save mat_eyePCA eyePCA
%     %����ROI
%     nosePCA = feature_PCA2(noseData,0.85);
%     save mat_nosePCA nosePCA
%     %���ROI
%     mouthPCA = feature_PCA2(mouthData,0.85);
%     save mat_mouthPCA mouthPCA
%     fprintf('PCA��ά֮���Gabor�������������\n\n');
% else
%     msgbox('û����ȷѡ��ά��ʽ������������������');
%     return;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����ѵ�����ݼ�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROIset=[eyePCA nosePCA mouthPCA];
% trainSet=cell(213,7);
% for i=1:213
%     trainSet{i,1}=samples_Data{i,1};%�������
%     trainSet{i,2}=samples_Data{i,2};%����
%     trainSet{i,3}=samples_Data{i,3};%�������ֱ�ʾ
%     trainSet{i,4}=samples_Data{i,4};%������
%     trainSet{i,5}=samples_Data{i,5};%�������ֱ�ʾ
%     trainSet{i,6}=ROIset(i,:);%PCA��ά֮������ݼ�
% 
% end
% save mat_trainSet trainSet;%����ѵ�����ݼ�

%���������������ѵ���������ݼ�
trainSet=[eyeData mouthData];
A=eye(7);
for i=1:213
%     P(:,i)=trainPCA(i,:)';%cell����תΪ�������ͺ�ת�ñ����ǣ�����������������
    P(:,i)=trainSet(i,:)';%cell����תΪ�������ͺ�ת�ñ����ǣ�����������������
%     T(:,i)=samples_Data{i,5};%��ǩ����
      T(:,i)=A(:,samples_Data{i,5});
end


save mat_trainSetP P;
save mat_trainSetT T;
fprintf('<����> ѵ�����ѱ������\n\n');

