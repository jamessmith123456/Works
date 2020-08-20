%�۾���λ����
%������һ��ͼ����󣬱����ǻҶȻ���ͼ�����
%����ֵ�������۵�����
function [leftXloc,leftYloc,rightXloc,rightYloc]=getFace_eyeDetect(imageMat,leftUp,leftDown,leftLeft,leftRight,rightUp,rightDown,rightLeft,rightRight)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��λ����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

leftXloc=0;%��������λ��X����
leftYloc=0;%��������λ��Y����
rightXloc=0;%��������λ��X����
rightYloc=0;%��������λ��Y����

%��λ����ͫ��λ��
m=leftDown-leftUp;
n=leftRight-leftLeft;
[m,n]= size(imageMat);
Hproj=zeros(m,1);%ˮƽ
Vproj=zeros(1,n);%��ֵ
MaxHproj=0;
MaxVproj=0;
for h=leftUp:leftDown
    Hproj(h) = sum(imageMat(h,leftLeft:leftRight)); 
    if Hproj(h)>MaxHproj
        MaxHproj=Hproj(h);
        leftYloc=h;
    end
end;
%subplot(245),plot(Hproj),title('���۶�ֵˮƽͶӰ��');
for v=leftLeft:leftRight
    Vproj(v) = sum(imageMat(leftUp:leftDown,v));  
    if Vproj(v)>MaxVproj
        MaxVproj=Vproj(v);
        leftXloc=v;
    end
end;
%subplot(245),plot(Vproj),title('���۶�ֵ��ֱͶӰ��');
%��ʾ��������
% str='�����������꣺(';
% str=strcat(str,num2str(leftXloc));
% str=strcat(str,',');
% str=strcat(str,num2str(leftYloc));
% str=strcat(str,')');
% disp(str);

%��λ����ͫ��λ��
m=rightDown-rightUp;
n=rightRight-rightLeft;
[m,n]= size(imageMat);
Hproj=zeros(m,1);%ˮƽ
Vproj=zeros(1,n);%��ֵ
MaxHproj=0;
MaxVproj=0;
for h=rightUp:rightDown
    Hproj(h) = sum(imageMat(h,rightLeft:rightRight)); 
    if Hproj(h)>MaxHproj
        MaxHproj=Hproj(h);
        rightYloc=h;
    end
end;
%subplot(246),plot(Hproj),title('���۶�ֵˮƽͶӰ��');
for v=rightLeft:rightRight
    Vproj(v) = sum(imageMat(rightUp:rightDown,v));  
    if Vproj(v)>MaxVproj
        MaxVproj=Vproj(v);
        rightXloc=v;
    end
end;
%subplot(247),plot(Vproj),title('���۶�ֵ��ֱͶӰ��');
%��ʾ��������
% str='�����������꣺(';
% str=strcat(str,num2str(rightXloc));
% str=strcat(str,',');
% str=strcat(str,num2str(rightYloc));
% str=strcat(str,')');
% disp(str);