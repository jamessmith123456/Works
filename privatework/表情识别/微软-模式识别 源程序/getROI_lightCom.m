function im=getROI_lightCom(I)

[m0,n0,l]=size(I);
%figure(1),imshow(I)
thresholdco=0.05;   %比例系数
thresholdnum=100;   %像素个数的临界常数
histogram=zeros(1,256);  %灰度级 数组
if m0*n0*thresholdco<thresholdnum
    disp('输入图像太小，请换一张！');
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
 %next获得满足系数thresholdco的临界灰度级
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
 % 获得满足条件的像素总的灰度值
 for i=256:-1:k
     averagegray=averagegray+histogram(1,i)*i;%像素总灰度值
     calnum=calnum+histogram(1,i);
 end
 averagegray=averagegray/calnum;%每个像素平均灰度值
 co=255.0/averagegray;
 %进行光线补偿
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
