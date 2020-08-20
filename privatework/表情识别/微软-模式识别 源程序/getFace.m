%Ԥ�����ĵ�
%hf=figure('NumberTitle','off','name','Horizontal Pjoection of A Gray Image');
%function PreImage(dir,dir_cut,image,levelAvg,delta1,delta2)
%����˵����
%inputStyle:ͼƬ�������뷽ʽ��1��ֱ������ͼƬ���ƣ�2������ͼƬ��������Ϣ�����е����ֵ��3���������ͼƬ��������ʱ���ñ�����              
%srcDir:ԴͼƬĿ¼                                     %dirCut:�ü���ͼƬ�洢Ŀ¼            
%samples_Info:������Ϣ����                                  %levelAvg����ֵ΢��ֵ
%delta1Y��ԭͼ���۾�ͶӰ������ֱ�������ֵ            
%delta2Y����ת��ͼ���۾�ͶӰ������ֱ�������ֵ            %delta2X����ת��ͼ���۾�ͶӰ����ˮƽ�������ֵ        
%delta3X,delta3Y��ǿ�Ƶ����ü��������Ͻǵ�����          
%imageNo��ͼƬ��������Ϣ��������ţ�1-213

function result=getFace(style,srcDir,dirCut,samples_Info,levelAvg,delta1Y,delta2Y,delta2X,delta3Y,delta3X,image)
close all;%�رյ�ǰ���д���
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����ͼƬ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%���һ��������ͼƬ����
switch style
    case 'name'
        imageStr=image;
        
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
        imageStr=Name;
        imageStr=strcat(imageStr,'.');
        imageStr=strcat(imageStr,Label);
        imageStr=strcat(imageStr,num2str(samples_Info(image,4)));
        imageStr=strcat(imageStr,'.');
        imageStr=strcat(imageStr,num2str(samples_Info(image,5)));
        imageStr=strcat(imageStr,'.tiff');
        disp(imageStr);%��ӡ�����ַ���
end

%control:����������ͣ�value:���ֶ��������͵ĵ���ֵ
%control=1:����levelAvg
%control=2:����delta1Y
%control=3:����delta2Y
%control=4:����delta2X
%control=5:����delta3Y
%control=6:����delta3X
%control=7:����ͼƬ��ת�Ƕȣ���Ϊǿ�Ƶ���

% function getFace(style,srcDir,dirCut,levelAvg,control,value,image)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %�޸ĵ�������
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% switch control
%     case 1
%         levelAvg=value;
%     case 2
%         delta1Ye=value;
%     case 3
%         delta2Y=value;
%     case 4
%         delta2X=value;        
%     case 5
%         delta3Y=value;        
%     case 6
%         delta3X=value;        
%     case 7
%         angle=value;   
% end
% imageStr=image;%���ͼƬ���Ƶ��ַ���

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����ͼƬ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imageLoc=strcat(srcDir,imageStr);
I=imread(imageLoc);
figure,subplot(241),imshow(I),title('ԭͼ��');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ֱ��ͼ���⻯
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J=imadjust(I);
J=histeq(I);
% J=adapthisteq(I);
subplot(242),imshow(J),title('���⻯֮��');

%//////////////////////////////////////////////////////////////////////////
%���ۼ��ͶӰ�������
%//////////////////////////////////////////////////////////////////////////
leftUp=110+delta1Y;
leftDown=150+delta1Y;
leftLeft=80;
leftRight=125;
rightUp=110+delta1Y;
rightDown=150+delta1Y;
rightLeft=131;
rightRight=176;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��ֵ��ͼ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
level=graythresh(J);
bw=im2bw(J,level/levelAvg);%������ֵ��ֵ��
bw=~bw;
subplot(243),imshow(bw),title('��ֵ����');
hold on;plot(leftLeft,leftUp,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����
hold on;plot(leftLeft,leftDown,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����
hold on;plot(rightRight,rightUp,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����
hold on;plot(rightRight,rightDown,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ˮƽ����ֵ��������ͶӰ���Դ�����λ�۾�λ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
leftX=0;%��������λ��X����
leftY=0;%��������λ��Y����
rightX=0;%��������λ��X����
rightY=0;%��������λ��Y����
%�������ۼ�⺯��
[leftX,leftY,rightX,rightY]=getFace_eyeDetect(bw,leftUp,leftDown,leftLeft,leftRight,rightUp,rightDown,rightLeft,rightRight);%���ۼ��//////////////////////
%��ʾ������ҵ����۾�λ��
subplot(244),imshow(J),title('�۾���λͼ��');
hold on;plot(leftX,leftY,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����
hold on;plot(rightX,rightY,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��תͼ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if control==7
%     angle=value;
% else
%     angle=atan((rightY-leftY)/(rightX-leftX));
% end
angle=atan((rightY-leftY)/(rightX-leftX));
Imrotate=imrotate(J,0.7*angle*180/pi);%����ͼƬ��ת����
subplot(245),imshow(Imrotate),title('��ת���ͼ��');

%//////////////////////////////////////////////////////////////////////////
%���ۼ��ͶӰ�������
%//////////////////////////////////////////////////////////////////////////
leftUp=120+delta2Y;
leftDown=160+delta2Y;
leftLeft=80+delta2X;
leftRight=135;
rightUp=120+delta2Y;
rightDown=160+delta2Y;
rightLeft=141;
rightRight=176+delta2X;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��ֵ����ת���ͼ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
level=graythresh(Imrotate);
bw=im2bw(Imrotate,level/levelAvg);%������ֵ��ֵ��
bw=~bw;
subplot(246),imshow(bw),title('��ֵ����ת��ͼ��');
hold on;plot(leftLeft,leftUp,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����
hold on;plot(leftLeft,leftDown,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����
hold on;plot(rightRight,rightUp,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����
hold on;plot(rightRight,rightDown,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ˮƽ����ֵ��������ͶӰ���Դ�����λ�۾�λ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
leftX=0;%��������λ��X����
leftY=0;%��������λ��Y����
rightX=0;%��������λ��X����
rightY=0;%��������λ��Y����
%�������ۼ�⺯��
[leftX,leftY,rightX,rightY]=getFace_eyeDetect(bw,leftUp,leftDown,leftLeft,leftRight,rightUp,rightDown,rightLeft,rightRight);%���ۼ��//////////////////////
%��ʾ������ҵ����۾�λ��
subplot(247),imshow(Imrotate),title('�۾���λͼ��');
hold on;plot(leftX,leftY,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����
hold on;plot(rightX,rightY,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%�����

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�ü�Ŀ������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cutWidth=120;%�ü�������
cutHeight=150;%�ü�����߶�
LeftUpX=round(delta3X+(leftX+rightX)/2-cutWidth/2);%�ü��������Ͻǵ�X���꣬ȡ����
LeftUpY=round(delta3Y+(leftY+rightY)/2-cutHeight/4);%�ü��������Ͻǵ�Y���꣬ȡ����
hold on;rectangle('Position',[LeftUpX,LeftUpY,cutWidth,cutHeight],'Curvature',[0,0],'LineWidth',2,'LineStyle',':','EdgeColor','g');%�����ο���ʾ�ü�����
Imcut=imcrop(Imrotate,[LeftUpX,LeftUpY,cutWidth,cutHeight]);%���òü�����
subplot(248),imshow(Imcut),title('�ü�����ͼ��');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�����������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_cutLoc=strcat(dirCut,imageStr);
imwrite(Imcut,image_cutLoc);%дͼƬ����

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����getROI����������ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%getROI('mat',dirCut,dirROI,Imcut,imageStr);








