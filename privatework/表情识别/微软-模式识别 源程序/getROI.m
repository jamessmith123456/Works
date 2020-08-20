%���Ժ���
%����˵����
%style='name' ,��ʾ�������������ͼƬ�����֣������ڶ���û�ã�
%style='mat'  ,��ʾ�������������ͼƬ�����֣������ڶ�����ͼƬ��Ӧ�ľ���
%style='no'   ,��ʾ�������������ͼƬ��JAFFE�����ţ������ڶ���û��
%srcDir : ͼƬԴĿ¼
%saveDir: ROIͼƬ����Ŀ¼
%samples_Info: ������Ϣ����
%imMat  : ͼƬ���ݾ���
%image  : ͼƬ���ƻ�����Ų���
function isComplete=getROI(style,srcDir,saveDir,samples_Info,imMat,image)
imageStr='';%����ROIͼƬ��ͼƬ����
I=[];%��Ŷ���ͼ��ľ���
%����ͼ��
switch style
    case 'name'
        close all;%ֱ������ͼƬ���ƣ�˵���������������ԣ���Ҫ�ر���ǰ���д��ڣ���������˾�������˵�����Ǳ�ĺ����ڵ��ã����ܹر���ǰ���ڡ�
        str=strcat(srcDir,image);
        I=imread(str);
        imageStr=image;
        %�������
        str='����ͼƬ���֣� <';
        str=strcat(str,image);
        str=strcat(str,'> ');
        disp(str);
    case 'mat'
        I=imMat;
        imageStr=image;
        %�������
        str='����ͼƬ���֣� <';
        str=strcat(str,image);
        str=strcat(str,'> ');
        disp(str);
    case 'no'
        %ȡ���˵�����
        switch samples_Info(image,2)
            case 1
                Name='KA';
            case 2
                Name='KL';
            case 3
                Name='KM';
            case 4
                Name='KR';
            case 5
                Name='MK';
            case 6
                Name='NA';
            case 7
                Name='NM';
            case 8
                Name='TM';
            case 9
                Name='UY';
            case 10
                Name='YM';
        end
        %ȡ�ñ�������
        switch samples_Info(image,3)
            case 1
                Label='AN';
            case 2
                Label='DI';
            case 3
                Label='FE';
            case 4
                Label='HA';
            case 5
                Label='NE';
            case 6
                Label='SA';
            case 7
                Label='SU';
        end
        %ȡ��ͼƬ�����ַ���
        imageName=Name;
        imageName=strcat(imageName,'.');
        imageName=strcat(imageName,Label);
        imageName=strcat(imageName,num2str(samples_Info(image,4)));
        imageName=strcat(imageName,'.');
        imageName=strcat(imageName,num2str(samples_Info(image,5)));
        imageName=strcat(imageName,'.tiff');
        imageStr=imageName;
        %����ͼƬ
        str=strcat(srcDir,imageName);
        I=imread(str);
        %�������
        str='����ͼƬ���֣� <';
        str=strcat(str,imageName);
        str=strcat(str,'> ');
        disp(str);
end        
figure,imshow(I),title('ԭͼ��');
hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����ȡROI�������������ȡ�㣬�Ҽ����ȡ���һ���㲢����ѭ����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��ȡ3��ROI
eyeXY=[];noseXY=[];mouthXY=[];%�洢3��ROI������������õ��Ľ�����
ROIno=0;%ROI���
while (1)
    ROIno=ROIno+1;
    if ROIno>3%3��ROIѡȡ��ϣ����˳�
        break;
    end
    xy=[];%�����ݴ����������ľ���
    n=0;
    but=1;
    %ѯ���ַ���
    switch ROIno
        case 1
            ROIname='�۾�';
        case 2
            ROIname='����';
        case 3
            ROIname='���';
    end
    str='Ҫ���� <';    str=strcat(str,ROIname);    str=strcat(str,'> �����ȡ��? (Y/N) :');
    askStr=input(str,'s');
    if askStr=='Y'||askStr=='y'
        while but==1
            n=n+1;
            [xi,yi,but]=ginput(1);
            plot(xi,yi,'ro');
            xy(:,n)=[xi;yi];
        end
        %����ֵ���������ԵĴ洢����
        switch ROIno
            case 1
                eyeXY=xy;
            case 2
                noseXY=xy;
            case 3
                mouthXY=xy;
        end
    elseif askStr=='N'
        msgbox('3��Ŀ������û��ѡȡ��ϣ�');
        isComplete=0;
        return;
    end
end
disp('3��Ŀ�������ȡ��ϣ�');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��ȡ3��ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��ȡ�۾�ROI����////////////////////////////////////////////////////////////
[eyeM,eyeN]=size(eyeXY);
if eyeN<=1  %����ѡȡ2���㣬���������Σ����ܶ�λ����
    msgbox('����������λ�㣬�۾������޷���λ��');
    isComplete=0;
    return;
end
eyeROI=[];%�洢�۾�ROI����ľ���
eyeYup=-65536;eyeYdown=65536;eyeXleft=-65536;eyeXright=65536;%�������յ�ROI����ı߽�ֵ
eyeROImean=mean(eyeXY,2);
eyeXY(1,:)=eyeXY(1,:)-eyeROImean(1,1);
eyeXY(2,:)=eyeXY(2,:)-eyeROImean(2,1);
[eyeM,eyeN]=size(eyeXY);
for j=1:eyeN
    %ѡȡ��߽�
    if eyeXY(1,j)<0
        if eyeXY(1,j)>eyeXleft
            eyeXleft=eyeXY(1,j);
        end
    end
    %ѡȡ�ұ߽�
    if eyeXY(1,j)>0
        if eyeXY(1,j)<eyeXright
            eyeXright=eyeXY(1,j);
        end
    end
    %ѡȡ�ϱ߽�
    if eyeXY(2,j)<0
        if eyeXY(2,j)>eyeYup
            eyeYup=eyeXY(2,j);
        end
    end
    %ѡȡ�±߽�
    if eyeXY(2,j)>0
        if eyeXY(2,j)<eyeYdown
            eyeYdown=eyeXY(2,j);
        end
    end
end
eyeXleft=round(eyeXleft+eyeROImean(1,1));
eyeXright=round(eyeXright+eyeROImean(1,1));
eyeYup=round(eyeYup+eyeROImean(2,1));
eyeYdown=round(eyeYdown+eyeROImean(2,1));
eyeROI=imcrop(I,[eyeXleft,eyeYup,eyeXright-eyeXleft,eyeYdown-eyeYup]);%���в��洢�۾�ROI����
hold on;rectangle('Position',[eyeXleft,eyeYup,eyeXright-eyeXleft,eyeYdown-eyeYup],'Curvature',[0,0],'LineWidth',2,'LineStyle',':','EdgeColor','g');%�����ο���ʾ�ü�����

%��ȡ����ROI����////////////////////////////////////////////////////////////
[noseM,noseN]=size(noseXY);
if noseN<=1  %����ѡȡ2���㣬���������Σ����ܶ�λ����
    msgbox('����������λ�㣬���������޷���λ��');
    isComplete=0;
    return;
end
noseROI=[];%�洢����ROI����ľ���
noseYup=-65536;noseYdown=65536;noseXleft=-65536;noseXright=65536;%�������յ�ROI����ı߽�ֵ
noseROImean=mean(noseXY,2);
noseXY(1,:)=noseXY(1,:)-noseROImean(1,1);
noseXY(2,:)=noseXY(2,:)-noseROImean(2,1);
[noseM,noseN]=size(noseXY);
for j=1:noseN
    %ѡȡ��߽�
    if noseXY(1,j)<0
        if noseXY(1,j)>noseXleft
            noseXleft=noseXY(1,j);
        end
    end
    %ѡȡ�ұ߽�
    if noseXY(1,j)>0
        if noseXY(1,j)<noseXright
            noseXright=noseXY(1,j);
        end
    end
    %ѡȡ�ϱ߽�
    if noseXY(2,j)<0
        if noseXY(2,j)>noseYup
            noseYup=noseXY(2,j);
        end
    end
    %ѡȡ�±߽�
    if noseXY(2,j)>0
        if noseXY(2,j)<noseYdown
            noseYdown=noseXY(2,j);
        end
    end
end
noseXleft=round(noseXleft+noseROImean(1,1));
noseXright=round(noseXright+noseROImean(1,1));
noseYup=round(noseYup+noseROImean(2,1));
noseYdown=round(noseYdown+noseROImean(2,1));
noseROI=imcrop(I,[noseXleft,noseYup,noseXright-noseXleft,noseYdown-noseYup]);%���в��洢����ROI����
hold on;rectangle('Position',[noseXleft,noseYup,noseXright-noseXleft,noseYdown-noseYup],'Curvature',[0,0],'LineWidth',2,'LineStyle',':','EdgeColor','g');%�����ο���ʾ�ü�����

%��ȡ���ROI����////////////////////////////////////////////////////////////
[mouthM,mouthN]=size(mouthXY);
if mouthN<=1  %����ѡȡ�����㣬���������Σ����ܶ�λ����
    msgbox('����������λ�㣬��������޷���λ��');
    isComplete=0;
    return;
end
mouthROI=[];%�洢���ROI����ľ���
mouthYup=-65536;mouthYdown=65536;mouthXleft=-65536;mouthXright=65536;%�������յ�ROI����ı߽�ֵ
mouthROImean=mean(mouthXY,2);
mouthXY(1,:)=mouthXY(1,:)-mouthROImean(1,1);
mouthXY(2,:)=mouthXY(2,:)-mouthROImean(2,1);
[mouthM,mouthN]=size(mouthXY);
for j=1:mouthN
    %ѡȡ��߽�
    if mouthXY(1,j)<0
        if mouthXY(1,j)>mouthXleft
            mouthXleft=mouthXY(1,j);
        end
    end
    %ѡȡ�ұ߽�
    if mouthXY(1,j)>0
        if mouthXY(1,j)<mouthXright
            mouthXright=mouthXY(1,j);
        end
    end
    %ѡȡ�ϱ߽�
    if mouthXY(2,j)<0
        if mouthXY(2,j)>mouthYup
            mouthYup=mouthXY(2,j);
        end
    end
    %ѡȡ�±߽�
    if mouthXY(2,j)>0
        if mouthXY(2,j)<mouthYdown
            mouthYdown=mouthXY(2,j);
        end
    end
end
mouthXleft=round(mouthXleft+mouthROImean(1,1));
mouthXright=round(mouthXright+mouthROImean(1,1));
mouthYup=round(mouthYup+mouthROImean(2,1));
mouthYdown=round(mouthYdown+mouthROImean(2,1));
mouthROI=imcrop(I,[mouthXleft,mouthYup,mouthXright-mouthXleft,mouthYdown-mouthYup]);%���в��洢���ROI����
hold on;rectangle('Position',[mouthXleft,mouthYup,mouthXright-mouthXleft,mouthYdown-mouthYup],'Curvature',[0,0],'LineWidth',2,'LineStyle',':','EdgeColor','g');%�����ο���ʾ�ü�����

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����3��ROI����ߴ磬ʹ�ù�һ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����ROI����ߴ綨��
eyeWidth=100;
eyeHeight=30;
noseWidth=45;
noseHeight=30;
mouthWidth=50;
mouthHeight=30;
%����
eyeROIre=imresize(eyeROI,[eyeHeight eyeWidth]);
figure,subplot(331),imshow(eyeROIre),title('�۾���һ����');
noseROIre=imresize(noseROI,[noseHeight noseWidth]);
subplot(332),imshow(noseROIre),title('���ӹ�һ����');
mouthROIre=imresize(mouthROI,[mouthHeight mouthWidth]);
subplot(333),imshow(mouthROIre),title('��͹�һ����');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�ж�ͼƬ����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off;
if isgray(I)==1  %�Ҷ�ͼ��
    disp('ԴͼƬ���ͣ� <�Ҷ�ͼ��> ');
    eyeROIgray=eyeROIre;
    noseROIgray=noseROIre;
    mouthROIgray=mouthROIre;
elseif isrgb(I)==1  %���ͼ��
    disp('ԴͼƬ���ͣ� <���ͼ��> ');
    %���ղ���
    L=getROI_lightCom(I);
    %ת��Ϊ�Ҷ�ͼ
    eyeROIgray=rgb2gray(eyeROIre);
    noseROIgray=rgb2gray(noseROIre);
    mouthROIgray=rgb2gray(mouthROIre);
else
    msgbox('��ͼƬ�Ȳ��ǻҶ�ͼ��Ҳ�������ͼ�������˳���');
    isComplete=0;
    return;
end
%��ʾ3��ROI�Ҷ�ͼ
subplot(334),imshow(eyeROIgray),title('�۾��Ҷ�ͼ��');
subplot(335),imshow(noseROIgray),title('���ӻҶ�ͼ��');
subplot(336),imshow(mouthROIgray),title('��ͻҶ�ͼ��');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ֱ��ͼ���⻯3��ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�۾�
eyeHist=imadjust(eyeROIgray);%������ǿ
eyeHisteq=histeq(eyeHist);%ֱ��ͼ���⻯
subplot(337),imshow(eyeHisteq),title('�۾����⻯��');
%����
noseHist=imadjust(noseROIgray);%������ǿ
noseHisteq=histeq(noseHist);%ֱ��ͼ���⻯
subplot(338),imshow(noseHisteq),title('���Ӿ��⻯��');
%���
mouthHist=imadjust(mouthROIgray);%������ǿ
mouthHisteq=histeq(mouthHist);%ֱ��ͼ���⻯
subplot(339),imshow(mouthHisteq),title('��;��⻯��');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����3��ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�۾�
saveLoc=strcat(saveDir,'eye');
saveLoc=strcat(saveLoc,imageStr);
imwrite(eyeHisteq,saveLoc);
%����
saveLoc=strcat(saveDir,'nose');
saveLoc=strcat(saveLoc,imageStr);
imwrite(noseHisteq,saveLoc);
%���
saveLoc=strcat(saveDir,'mouth');
saveLoc=strcat(saveLoc,imageStr);
imwrite(mouthHisteq,saveLoc);

%��ʾ��ʾ��Ϣ
str='ROIͼƬ�ѱ��浽ָ��Ŀ¼ <';
str=strcat(str,saveDir);
str=strcat(str,'> ��');
disp(str);

isComplete=1;






