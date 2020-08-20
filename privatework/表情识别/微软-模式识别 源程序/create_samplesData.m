%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%创建样本数据函数
%前提条件:JAFFE_ROI文件夹下已有ROI全部文件；样本信息矩阵已经存在
%参数说明
%roiDi        : ROI存放的目录
%faceDir      : 脸部图片存放的目录
%samples_Info : 样本信息矩阵，这是最原始的矩阵
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function samples_Data=create_samplesData(roiDir,faceDir,samples_Info)
samples_Data=cell(213,10);
for i=1:213
    %图片样本序号
    samples_Data{i,1}=samples_Info(i,1);
    
    %人名
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
    
    %人名的数字表示
    samples_Data{i,3}=samples_Info(i,2);
    
    %表情名字
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
    %表情的数字表示
    samples_Data{i,5}=samples_Info(i,3);
    
    %该人该表情图片序号
    samples_Data{i,6}=samples_Info(i,4);
    
    %该图片在所有样本中序号
    samples_Data{i,7}=samples_Info(i,5);
    
    %图片名称
    imageName=samples_Data{i,2};
    imageName=strcat(imageName,'.');
    imageName=strcat(imageName,samples_Data{i,4});
    imageName=strcat(imageName,num2str(samples_Info(i,4)));
    imageName=strcat(imageName,'.');
    imageName=strcat(imageName,num2str(samples_Info(i,5)));
    imageName=strcat(imageName,'.tiff');
    samples_Data{i,8}=imageName;
    
    %眼睛数据
    str=strcat(roiDir,'eye');
    str=strcat(str,imageName);
    I=imread(str);%从目录下读取ROI图片
    B=imresize(I,[10 30]);
    samples_Data{i,9}=B;
    
    %鼻子数据
    str=strcat(roiDir,'nose');
    str=strcat(str,imageName);
    I=imread(str);%从目录下读取ROI图片
    B=imresize(I,[16 24]);
    samples_Data{i,10}=B;
    
    %嘴巴数据
    str=strcat(roiDir,'mouth');
    str=strcat(str,imageName);
    I=imread(str);%从目录下读取ROI图片
    B=imresize(I,[12 18]);
    samples_Data{i,11}=B;
    
    %脸的全部数据
    str=strcat(faceDir,imageName);
    I=imread(str);%从目录下读取ROI图片
    B=imresize(I,[50 40]);
    samples_Data{i,12}=B;
end

%保存样本数据矩阵
save mat_SamplesData samples_Data