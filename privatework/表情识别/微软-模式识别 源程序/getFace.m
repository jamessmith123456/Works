%预处理文档
%hf=figure('NumberTitle','off','name','Horizontal Pjoection of A Gray Image');
%function PreImage(dir,dir_cut,image,levelAvg,delta1,delta2)
%参数说明：
%inputStyle:图片名称输入方式，1：直接输入图片名称；2：输入图片在样本信息矩阵中的序号值；3：输入测试图片，即测试时调用本函数              
%srcDir:源图片目录                                     %dirCut:裁剪后图片存储目录            
%samples_Info:样本信息矩阵                                  %levelAvg：阈值微调值
%delta1Y：原图像眼睛投影区域竖直方向调节值            
%delta2Y：旋转后图像眼睛投影区域竖直方向调节值            %delta2X：旋转后图像眼睛投影区域水平方向调节值        
%delta3X,delta3Y：强制调整裁剪区域左上角点坐标          
%imageNo：图片在样本信息矩阵中序号，1-213

function result=getFace(style,srcDir,dirCut,samples_Info,levelAvg,delta1Y,delta2Y,delta2X,delta3Y,delta3X,image)
close all;%关闭当前所有窗口
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%读入图片
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%最后一个参数是图片名称
switch style
    case 'name'
        imageStr=image;
        
    case 'no'
        %取得人的名字
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
        %取得表情名字
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
        %取得图片名字字符串
        imageStr=Name;
        imageStr=strcat(imageStr,'.');
        imageStr=strcat(imageStr,Label);
        imageStr=strcat(imageStr,num2str(samples_Info(image,4)));
        imageStr=strcat(imageStr,'.');
        imageStr=strcat(imageStr,num2str(samples_Info(image,5)));
        imageStr=strcat(imageStr,'.tiff');
        disp(imageStr);%打印名字字符串
end

%control:代表调节类型；value:该手动调节类型的调节值
%control=1:调节levelAvg
%control=2:调节delta1Y
%control=3:调节delta2Y
%control=4:调节delta2X
%control=5:调节delta3Y
%control=6:调节delta3X
%control=7:调节图片旋转角度，此为强制调节

% function getFace(style,srcDir,dirCut,levelAvg,control,value,image)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %修改调节类型
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
% imageStr=image;%存放图片名称的字符串

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%读入图片
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imageLoc=strcat(srcDir,imageStr);
I=imread(imageLoc);
figure,subplot(241),imshow(I),title('原图：');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%直方图均衡化
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J=imadjust(I);
J=histeq(I);
% J=adapthisteq(I);
subplot(242),imshow(J),title('均衡化之后：');

%//////////////////////////////////////////////////////////////////////////
%人眼检测投影区域变量
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
%二值化图像
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
level=graythresh(J);
bw=im2bw(J,level/levelAvg);%根据阈值二值化
bw=~bw;
subplot(243),imshow(bw),title('二值化后：');
hold on;plot(leftLeft,leftUp,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记
hold on;plot(leftLeft,leftDown,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记
hold on;plot(rightRight,rightUp,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记
hold on;plot(rightRight,rightDown,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%水平和数值两个方向投影，以此来定位眼睛位置
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
leftX=0;%左眼中心位置X坐标
leftY=0;%左眼中心位置Y坐标
rightX=0;%右眼中心位置X坐标
rightY=0;%右眼中心位置Y坐标
%调用人眼检测函数
[leftX,leftY,rightX,rightY]=getFace_eyeDetect(bw,leftUp,leftDown,leftLeft,leftRight,rightUp,rightDown,rightLeft,rightRight);%人眼检测//////////////////////
%显示并标记找到的眼睛位置
subplot(244),imshow(J),title('眼睛定位图：');
hold on;plot(leftX,leftY,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记
hold on;plot(rightX,rightY,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%旋转图像
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if control==7
%     angle=value;
% else
%     angle=atan((rightY-leftY)/(rightX-leftX));
% end
angle=atan((rightY-leftY)/(rightX-leftX));
Imrotate=imrotate(J,0.7*angle*180/pi);%调用图片旋转函数
subplot(245),imshow(Imrotate),title('旋转后的图像：');

%//////////////////////////////////////////////////////////////////////////
%人眼检测投影区域变量
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
%二值化旋转后的图像
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
level=graythresh(Imrotate);
bw=im2bw(Imrotate,level/levelAvg);%根据阈值二值化
bw=~bw;
subplot(246),imshow(bw),title('二值化旋转后图像：');
hold on;plot(leftLeft,leftUp,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记
hold on;plot(leftLeft,leftDown,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记
hold on;plot(rightRight,rightUp,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记
hold on;plot(rightRight,rightDown,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%水平和数值两个方向投影，以此来定位眼睛位置
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
leftX=0;%左眼中心位置X坐标
leftY=0;%左眼中心位置Y坐标
rightX=0;%右眼中心位置X坐标
rightY=0;%右眼中心位置Y坐标
%调用人眼检测函数
[leftX,leftY,rightX,rightY]=getFace_eyeDetect(bw,leftUp,leftDown,leftLeft,leftRight,rightUp,rightDown,rightLeft,rightRight);%人眼检测//////////////////////
%显示并标记找到的眼睛位置
subplot(247),imshow(Imrotate),title('眼睛定位图：');
hold on;plot(leftX,leftY,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记
hold on;plot(rightX,rightY,'md','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor',[.49 1 .63], 'MarkerSize',4);%做标记

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%裁剪目标区域
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cutWidth=120;%裁剪区域宽度
cutHeight=150;%裁剪区域高度
LeftUpX=round(delta3X+(leftX+rightX)/2-cutWidth/2);%裁剪区域左上角点X坐标，取整后
LeftUpY=round(delta3Y+(leftY+rightY)/2-cutHeight/4);%裁剪区域左上角点Y坐标，取整后
hold on;rectangle('Position',[LeftUpX,LeftUpY,cutWidth,cutHeight],'Curvature',[0,0],'LineWidth',2,'LineStyle',':','EdgeColor','g');%画矩形框，显示裁剪区域
Imcut=imcrop(Imrotate,[LeftUpX,LeftUpY,cutWidth,cutHeight]);%调用裁剪函数
subplot(248),imshow(Imcut),title('裁剪区域图：');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%保存剪切区域
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_cutLoc=strcat(dirCut,imageStr);
imwrite(Imcut,image_cutLoc);%写图片函数

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%调用getROI函数，剪切ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%getROI('mat',dirCut,dirROI,Imcut,imageStr);








