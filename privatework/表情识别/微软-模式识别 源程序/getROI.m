%测试函数
%参数说明：
%style='name' ,表示最后参数输入的是图片的名字，倒数第二个没用；
%style='mat'  ,表示最后参数输入的是图片的名字，倒数第二个是图片对应的矩阵；
%style='no'   ,表示最后参数输入的是图片在JAFFE里的序号，倒数第二个没用
%srcDir : 图片源目录
%saveDir: ROI图片保存目录
%samples_Info: 样本信息矩阵
%imMat  : 图片数据矩阵
%image  : 图片名称或者序号参数
function isComplete=getROI(style,srcDir,saveDir,samples_Info,imMat,image)
imageStr='';%保存ROI图片的图片名称
I=[];%存放读入图像的矩阵
%读入图像
switch style
    case 'name'
        close all;%直接输入图片名称，说明可能是真正测试，需要关闭以前所有窗口，如果输入了矩阵，至少说明了是别的函数在调用，不能关闭以前窗口。
        str=strcat(srcDir,image);
        I=imread(str);
        imageStr=image;
        %输出名字
        str='输入图片名字： <';
        str=strcat(str,image);
        str=strcat(str,'> ');
        disp(str);
    case 'mat'
        I=imMat;
        imageStr=image;
        %输出名字
        str='输入图片名字： <';
        str=strcat(str,image);
        str=strcat(str,'> ');
        disp(str);
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
        imageName=Name;
        imageName=strcat(imageName,'.');
        imageName=strcat(imageName,Label);
        imageName=strcat(imageName,num2str(samples_Info(image,4)));
        imageName=strcat(imageName,'.');
        imageName=strcat(imageName,num2str(samples_Info(image,5)));
        imageName=strcat(imageName,'.tiff');
        imageStr=imageName;
        %读入图片
        str=strcat(srcDir,imageName);
        I=imread(str);
        %输出名字
        str='输入图片名字： <';
        str=strcat(str,imageName);
        str=strcat(str,'> ');
        disp(str);
end        
figure,imshow(I),title('原图：');
hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%鼠标点取ROI区域。鼠标左键点击取点，右键点击取最后一个点并结束循环。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%点取3个ROI
eyeXY=[];noseXY=[];mouthXY=[];%存储3个ROI区域的鼠标点击获得的四角坐标
ROIno=0;%ROI序号
while (1)
    ROIno=ROIno+1;
    if ROIno>3%3个ROI选取完毕，则退出
        break;
    end
    xy=[];%声明暂存鼠标点击坐标的矩阵
    n=0;
    but=1;
    %询问字符串
    switch ROIno
        case 1
            ROIname='眼睛';
        case 2
            ROIname='鼻子';
        case 3
            ROIname='嘴巴';
    end
    str='要进行 <';    str=strcat(str,ROIname);    str=strcat(str,'> 区域点取吗? (Y/N) :');
    askStr=input(str,'s');
    if askStr=='Y'||askStr=='y'
        while but==1
            n=n+1;
            [xi,yi,but]=ginput(1);
            plot(xi,yi,'ro');
            xy(:,n)=[xi;yi];
        end
        %坐标值拷贝给各自的存储数组
        switch ROIno
            case 1
                eyeXY=xy;
            case 2
                noseXY=xy;
            case 3
                mouthXY=xy;
        end
    elseif askStr=='N'
        msgbox('3个目标区域还没有选取完毕！');
        isComplete=0;
        return;
    end
end
disp('3个目标区域点取完毕！');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%挖取3个ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%挖取眼睛ROI区域////////////////////////////////////////////////////////////
[eyeM,eyeN]=size(eyeXY);
if eyeN<=1  %至少选取2个点，构成三角形，才能定位矩形
    msgbox('少于两个定位点，眼睛区域无法定位！');
    isComplete=0;
    return;
end
eyeROI=[];%存储眼睛ROI区域的矩阵
eyeYup=-65536;eyeYdown=65536;eyeXleft=-65536;eyeXright=65536;%定义最终的ROI区域的边界值
eyeROImean=mean(eyeXY,2);
eyeXY(1,:)=eyeXY(1,:)-eyeROImean(1,1);
eyeXY(2,:)=eyeXY(2,:)-eyeROImean(2,1);
[eyeM,eyeN]=size(eyeXY);
for j=1:eyeN
    %选取左边界
    if eyeXY(1,j)<0
        if eyeXY(1,j)>eyeXleft
            eyeXleft=eyeXY(1,j);
        end
    end
    %选取右边界
    if eyeXY(1,j)>0
        if eyeXY(1,j)<eyeXright
            eyeXright=eyeXY(1,j);
        end
    end
    %选取上边界
    if eyeXY(2,j)<0
        if eyeXY(2,j)>eyeYup
            eyeYup=eyeXY(2,j);
        end
    end
    %选取下边界
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
eyeROI=imcrop(I,[eyeXleft,eyeYup,eyeXright-eyeXleft,eyeYdown-eyeYup]);%剪切并存储眼睛ROI区域
hold on;rectangle('Position',[eyeXleft,eyeYup,eyeXright-eyeXleft,eyeYdown-eyeYup],'Curvature',[0,0],'LineWidth',2,'LineStyle',':','EdgeColor','g');%画矩形框，显示裁剪区域

%挖取鼻子ROI区域////////////////////////////////////////////////////////////
[noseM,noseN]=size(noseXY);
if noseN<=1  %至少选取2个点，构成三角形，才能定位矩形
    msgbox('少于两个定位点，鼻子区域无法定位！');
    isComplete=0;
    return;
end
noseROI=[];%存储鼻子ROI区域的矩阵
noseYup=-65536;noseYdown=65536;noseXleft=-65536;noseXright=65536;%定义最终的ROI区域的边界值
noseROImean=mean(noseXY,2);
noseXY(1,:)=noseXY(1,:)-noseROImean(1,1);
noseXY(2,:)=noseXY(2,:)-noseROImean(2,1);
[noseM,noseN]=size(noseXY);
for j=1:noseN
    %选取左边界
    if noseXY(1,j)<0
        if noseXY(1,j)>noseXleft
            noseXleft=noseXY(1,j);
        end
    end
    %选取右边界
    if noseXY(1,j)>0
        if noseXY(1,j)<noseXright
            noseXright=noseXY(1,j);
        end
    end
    %选取上边界
    if noseXY(2,j)<0
        if noseXY(2,j)>noseYup
            noseYup=noseXY(2,j);
        end
    end
    %选取下边界
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
noseROI=imcrop(I,[noseXleft,noseYup,noseXright-noseXleft,noseYdown-noseYup]);%剪切并存储鼻子ROI区域
hold on;rectangle('Position',[noseXleft,noseYup,noseXright-noseXleft,noseYdown-noseYup],'Curvature',[0,0],'LineWidth',2,'LineStyle',':','EdgeColor','g');%画矩形框，显示裁剪区域

%挖取嘴巴ROI区域////////////////////////////////////////////////////////////
[mouthM,mouthN]=size(mouthXY);
if mouthN<=1  %至少选取两个点，构成三角形，才能定位矩形
    msgbox('少于两个定位点，嘴巴区域无法定位！');
    isComplete=0;
    return;
end
mouthROI=[];%存储嘴巴ROI区域的矩阵
mouthYup=-65536;mouthYdown=65536;mouthXleft=-65536;mouthXright=65536;%定义最终的ROI区域的边界值
mouthROImean=mean(mouthXY,2);
mouthXY(1,:)=mouthXY(1,:)-mouthROImean(1,1);
mouthXY(2,:)=mouthXY(2,:)-mouthROImean(2,1);
[mouthM,mouthN]=size(mouthXY);
for j=1:mouthN
    %选取左边界
    if mouthXY(1,j)<0
        if mouthXY(1,j)>mouthXleft
            mouthXleft=mouthXY(1,j);
        end
    end
    %选取右边界
    if mouthXY(1,j)>0
        if mouthXY(1,j)<mouthXright
            mouthXright=mouthXY(1,j);
        end
    end
    %选取上边界
    if mouthXY(2,j)<0
        if mouthXY(2,j)>mouthYup
            mouthYup=mouthXY(2,j);
        end
    end
    %选取下边界
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
mouthROI=imcrop(I,[mouthXleft,mouthYup,mouthXright-mouthXleft,mouthYdown-mouthYup]);%剪切并存储嘴巴ROI区域
hold on;rectangle('Position',[mouthXleft,mouthYup,mouthXright-mouthXleft,mouthYdown-mouthYup],'Curvature',[0,0],'LineWidth',2,'LineStyle',':','EdgeColor','g');%画矩形框，显示裁剪区域

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%调整3个ROI区域尺寸，使得归一化
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%三个ROI区域尺寸定义
eyeWidth=100;
eyeHeight=30;
noseWidth=45;
noseHeight=30;
mouthWidth=50;
mouthHeight=30;
%调整
eyeROIre=imresize(eyeROI,[eyeHeight eyeWidth]);
figure,subplot(331),imshow(eyeROIre),title('眼睛归一化：');
noseROIre=imresize(noseROI,[noseHeight noseWidth]);
subplot(332),imshow(noseROIre),title('鼻子归一化：');
mouthROIre=imresize(mouthROI,[mouthHeight mouthWidth]);
subplot(333),imshow(mouthROIre),title('嘴巴归一化：');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%判断图片类型
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off;
if isgray(I)==1  %灰度图像
    disp('源图片类型： <灰度图像> ');
    eyeROIgray=eyeROIre;
    noseROIgray=noseROIre;
    mouthROIgray=mouthROIre;
elseif isrgb(I)==1  %真彩图像
    disp('源图片类型： <真彩图像> ');
    %光照补偿
    L=getROI_lightCom(I);
    %转化为灰度图
    eyeROIgray=rgb2gray(eyeROIre);
    noseROIgray=rgb2gray(noseROIre);
    mouthROIgray=rgb2gray(mouthROIre);
else
    msgbox('该图片既不是灰度图，也不是真彩图，程序退出！');
    isComplete=0;
    return;
end
%显示3个ROI灰度图
subplot(334),imshow(eyeROIgray),title('眼睛灰度图：');
subplot(335),imshow(noseROIgray),title('鼻子灰度图：');
subplot(336),imshow(mouthROIgray),title('嘴巴灰度图：');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%直方图均衡化3个ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%眼睛
eyeHist=imadjust(eyeROIgray);%像素增强
eyeHisteq=histeq(eyeHist);%直方图均衡化
subplot(337),imshow(eyeHisteq),title('眼睛均衡化：');
%鼻子
noseHist=imadjust(noseROIgray);%像素增强
noseHisteq=histeq(noseHist);%直方图均衡化
subplot(338),imshow(noseHisteq),title('鼻子均衡化：');
%嘴巴
mouthHist=imadjust(mouthROIgray);%像素增强
mouthHisteq=histeq(mouthHist);%直方图均衡化
subplot(339),imshow(mouthHisteq),title('嘴巴均衡化：');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%保存3个ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%眼睛
saveLoc=strcat(saveDir,'eye');
saveLoc=strcat(saveLoc,imageStr);
imwrite(eyeHisteq,saveLoc);
%鼻子
saveLoc=strcat(saveDir,'nose');
saveLoc=strcat(saveLoc,imageStr);
imwrite(noseHisteq,saveLoc);
%嘴巴
saveLoc=strcat(saveDir,'mouth');
saveLoc=strcat(saveLoc,imageStr);
imwrite(mouthHisteq,saveLoc);

%显示提示信息
str='ROI图片已保存到指定目录 <';
str=strcat(str,saveDir);
str=strcat(str,'> 下');
disp(str);

isComplete=1;






