function im=getROI_lightCom(I)

[m0,n0,l]=size(I);
%figure(1),imshow(I)
thresholdco=0.05;   %����ϵ��
thresholdnum=100;   %���ظ������ٽ糣��
histogram=zeros(1,256);  %�Ҷȼ� ����
if m0*n0*thresholdco<thresholdnum
    disp('����ͼ��̫С���뻻һ�ţ�');
    return
end
gray=0;
index0=0;
for i=1:m0        %scan image
     for j=1:n0
          gray=round(I(i,j,1)*.299+I(i,j,2)*.587+I(i,j,3)*.114);
          index0=gray+1;
          histogram(1,index0)= histogram(1,index0)+1;
     end
 end
 calnum=0;
 total=m0*n0;
 num=0;
 %next�������ϵ��thresholdco���ٽ�Ҷȼ�
 index1=0;
 for i=1:256
     if calnum/total<thresholdco
         index1=256-i+1;
         calnum=calnum+histogram(1,index1);
         num=i;
     else
         break;
     end
 end
 averagegray=0;
 calnum=0;
 k=256-num+1;
 % ������������������ܵĻҶ�ֵ
 for i=256:-1:k
     averagegray=averagegray+histogram(1,i)*i;%�����ܻҶ�ֵ
     calnum=calnum+histogram(1,i);
 end
 averagegray=averagegray/calnum;%ÿ������ƽ���Ҷ�ֵ
 co=255.0/averagegray;
 %���й��߲���
  for i=1:m0
      for j=1:n0
          I(i,j,1)=I(i,j,1)*co;
          if I(i,j,1)>255
              I(i,j,1)=255;
          end
          I(i,j,2)=I(i,j,2)*co;
          if I(i,j,2)>255
              I(i,j,2)=255;
          end
          I(i,j,3)=I(i,j,3)*co;
          if I(i,j,3)>255
              I(i,j,3)=255;
          end
      end
  end
 im=zeros(m0,n0,l);
im=I; %
