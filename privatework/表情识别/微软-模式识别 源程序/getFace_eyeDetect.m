%眼睛定位函数
%参数：一个图像矩阵，必须是灰度化的图像矩阵
%返回值：左右眼的坐标
function [leftXloc,leftYloc,rightXloc,rightYloc]=getFace_eyeDetect(imageMat,leftUp,leftDown,leftLeft,leftRight,rightUp,rightDown,rightLeft,rightRight)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%定位人眼
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

leftXloc=0;%左眼中心位置X坐标
leftYloc=0;%左眼中心位置Y坐标
rightXloc=0;%右眼中心位置X坐标
rightYloc=0;%右眼中心位置Y坐标

%定位左眼瞳孔位置
m=leftDown-leftUp;
n=leftRight-leftLeft;
[m,n]= size(imageMat);
Hproj=zeros(m,1);%水平
Vproj=zeros(1,n);%数值
MaxHproj=0;
MaxVproj=0;
for h=leftUp:leftDown
    Hproj(h) = sum(imageMat(h,leftLeft:leftRight)); 
    if Hproj(h)>MaxHproj
        MaxHproj=Hproj(h);
        leftYloc=h;
    end
end;
%subplot(245),plot(Hproj),title('左眼二值水平投影：');
for v=leftLeft:leftRight
    Vproj(v) = sum(imageMat(leftUp:leftDown,v));  
    if Vproj(v)>MaxVproj
        MaxVproj=Vproj(v);
        leftXloc=v;
    end
end;
%subplot(245),plot(Vproj),title('左眼二值竖直投影：');
%显示左眼坐标
% str='左眼中心坐标：(';
% str=strcat(str,num2str(leftXloc));
% str=strcat(str,',');
% str=strcat(str,num2str(leftYloc));
% str=strcat(str,')');
% disp(str);

%定位右眼瞳孔位置
m=rightDown-rightUp;
n=rightRight-rightLeft;
[m,n]= size(imageMat);
Hproj=zeros(m,1);%水平
Vproj=zeros(1,n);%数值
MaxHproj=0;
MaxVproj=0;
for h=rightUp:rightDown
    Hproj(h) = sum(imageMat(h,rightLeft:rightRight)); 
    if Hproj(h)>MaxHproj
        MaxHproj=Hproj(h);
        rightYloc=h;
    end
end;
%subplot(246),plot(Hproj),title('右眼二值水平投影：');
for v=rightLeft:rightRight
    Vproj(v) = sum(imageMat(rightUp:rightDown,v));  
    if Vproj(v)>MaxVproj
        MaxVproj=Vproj(v);
        rightXloc=v;
    end
end;
%subplot(247),plot(Vproj),title('右眼二值竖直投影：');
%显示右眼坐标
% str='右眼中心坐标：(';
% str=strcat(str,num2str(rightXloc));
% str=strcat(str,',');
% str=strcat(str,num2str(rightYloc));
% str=strcat(str,')');
% disp(str);