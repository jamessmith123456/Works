%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�����������ݺ���
%ǰ������:JAFFE_ROI�ļ���������ROIȫ���ļ���������Ϣ�����Ѿ�����
%����˵��
%roiDi        : ROI��ŵ�Ŀ¼
%faceDir      : ����ͼƬ��ŵ�Ŀ¼
%samples_Info : ������Ϣ����������ԭʼ�ľ���
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function samples_Data=create_samplesData(roiDir,faceDir,samples_Info)
samples_Data=cell(213,10);
for i=1:213
    %ͼƬ�������
    samples_Data{i,1}=samples_Info(i,1);
    
    %����
    switch samples_Info(i,2)
        case 1
            samples_Data{i,2}='KA';
        case 2
            samples_Data{i,2}='KL';
        case 3
            samples_Data{i,2}='KM';
        case 4
            samples_Data{i,2}='KR';
        case 5
            samples_Data{i,2}='MK';
        case 6
            samples_Data{i,2}='NA';
        case 7
            samples_Data{i,2}='NM';
        case 8
            samples_Data{i,2}='TM';
        case 9
            samples_Data{i,2}='UY';
        case 10
            samples_Data{i,2}='YM';
    end
    
    %���������ֱ�ʾ
    samples_Data{i,3}=samples_Info(i,2);
    
    %��������
    switch samples_Info(i,3)
        case 1
            samples_Data{i,4}='AN';
        case 2
            samples_Data{i,4}='DI';
        case 3
            samples_Data{i,4}='FE';
        case 4
            samples_Data{i,4}='HA';
        case 5
            samples_Data{i,4}='NE';
        case 6
            samples_Data{i,4}='SA';
        case 7
            samples_Data{i,4}='SU';
    end
    %��������ֱ�ʾ
    samples_Data{i,5}=samples_Info(i,3);
    
    %���˸ñ���ͼƬ���
    samples_Data{i,6}=samples_Info(i,4);
    
    %��ͼƬ���������������
    samples_Data{i,7}=samples_Info(i,5);
    
    %ͼƬ����
    imageName=samples_Data{i,2};
    imageName=strcat(imageName,'.');
    imageName=strcat(imageName,samples_Data{i,4});
    imageName=strcat(imageName,num2str(samples_Info(i,4)));
    imageName=strcat(imageName,'.');
    imageName=strcat(imageName,num2str(samples_Info(i,5)));
    imageName=strcat(imageName,'.tiff');
    samples_Data{i,8}=imageName;
    
    %�۾�����
    str=strcat(roiDir,'eye');
    str=strcat(str,imageName);
    I=imread(str);%��Ŀ¼�¶�ȡROIͼƬ
    B=imresize(I,[10 30]);
    samples_Data{i,9}=B;
    
    %��������
    str=strcat(roiDir,'nose');
    str=strcat(str,imageName);
    I=imread(str);%��Ŀ¼�¶�ȡROIͼƬ
    B=imresize(I,[16 24]);
    samples_Data{i,10}=B;
    
    %�������
    str=strcat(roiDir,'mouth');
    str=strcat(str,imageName);
    I=imread(str);%��Ŀ¼�¶�ȡROIͼƬ
    B=imresize(I,[12 18]);
    samples_Data{i,11}=B;
    
    %����ȫ������
    str=strcat(faceDir,imageName);
    I=imread(str);%��Ŀ¼�¶�ȡROIͼƬ
    B=imresize(I,[50 40]);
    samples_Data{i,12}=B;
end

%�����������ݾ���
save mat_SamplesData samples_Data