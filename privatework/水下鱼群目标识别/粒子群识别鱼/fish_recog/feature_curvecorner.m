function [cont_K]=feature_curvecorner(varargin)
%function [cout,marked_img, align, cstd, weightcenterx weightcentery]=feature_corner(varargin)


%   CORNER Find corners in intensity image. 
%   
%       CORNER works by the following step:
%       1.	Apply the Canny edge detector to the gray level image and obtain a
%       binary edge-map.
%       2.	Extract the edge contours from the edge-map, fill the gaps in the
%       contours.
%       3.	Compute curvature at a low scale for each contour to retain all
%       true corners.
%       4.	All of the curvature local maxima are considered as corner
%       candidates, then rounded corners and false corners due to boundary
%       noise and details were eliminated.
%       5.  End points of line mode curve were added as corner, if they are not
%       close to the above detected corners.
%
%       Syntax :    
%       [cout,marked_img]=corner(I,C,T_angle,sig,H,L,Endpiont,Gap_size)
%
%       Input :
%       I -  the input image, it could be gray, color or binary image. If I is
%           empty([]), input image can be get from a open file dialog box.
%       C -  denotes the minimum ratio of major axis to minor axis of an ellipse, 
%           whose vertex could be detected as a corner by proposed detector.  
%           The default value is 1.5.
%       T_angle -  denotes the maximum obtuse angle that a corner can have when 
%           it is detected as a true corner, default value is 162.
%       Sig -  denotes the standard deviation of the Gaussian filter when
%           computeing curvature. The default sig is 3.
%       H,L -  high and low threshold of Canny edge detector. The default value
%           is 0.35 and 0.
%       Endpoint -  a flag to control whether add the end points of a curve
%           as corner, 1 means Yes and 0 means No. The default value is 1.
%       Gap_size -  a paremeter use to fill the gaps in the contours, the gap
%           not more than gap_size were filled in this stage. The default 
%           Gap_size is 1 pixels.
%
%       Output :
%       cout -  a position pair list of detected corners in the input image.
%       marked_image -  image with detected corner marked.
%
%       Examples
%       -------
%       I = imread('alumgrns.tif');
%       cout = corner(I,[],[],[],0.2);
%
%       [cout, marked_image] = corner;
%
%       cout = corner([],1.6,155);
%
%
%   Composed by He Xiaochen 
%   HKU EEE Dept. ITSR, Apr. 2005
%
%   Algorithm is derived from :
%       X.C. He and N.H.C. Yung, Curvature Scale Space Corner Detector with  
%       Adaptive Threshold and Dynamic Region of Support, Proceedings of the
%       17th International Conference on Pattern Recognition, 2:791-794, August 2004.
%   Improved algorithm is included in :
%   	X.C. He and N.H.C. Yung, �Corner detector based on global and local curvature properties�, 
%       Optical Engineering, 47(5), pp: 057008, 2008.
%   

[I,C,T_angle,sig,H,L,Endpoint,Gap_size] = parse_inputs(varargin{:});

if size(I,3)==3
    I=rgb2gray(I); % Transform RGB image to a Gray one. 
end

%tic
BW=edge(I,'canny',[L,H]);  % Detect edges
%time_for_detecting_edge=toc

%tic
[curve,curve_start,curve_end,curve_mode,curve_num]=extract_curve(BW,Gap_size);  % Extract curves
%time_for_extracting_curve=toc

%tic
[cont_K] =get_corner(curve,curve_start,curve_end,curve_mode,curve_num,BW,sig,Endpoint,C,T_angle); % Detect corners
%[cout cstd kout_ori xout yout] =get_corner(curve,curve_start,curve_end,curve_mode,curve_num,BW,sig,Endpoint,C,T_angle); % Detect corners
%time_for_detecting_corner=toc

% kout = zeros(length(kout_ori),1);
% kout(2:end) = abs(kout_ori(2:end)-kout_ori(1:end-1));
% 
% weightcenterx = sum(kout.*yout)/sum(kout);
% weightcentery = sum(kout.*xout)/sum(kout);

% img=I;
% for i=1:size(cout,1)
%     img=mark(img,cout(i,1),cout(i,2),5);
% end
% marked_img=img;
% figure(2)
% imshow(marked_img);
% title('Detected corners')
%hold off;
%saveas(gcf, ['2_' num2str(ceil(now*10000000))], 'png');
% hold on;
% for i=1:size(cout,1)
%     text(cout(i,1),cout(i,2),num2str(i));
% end
% hold off;

%imwrite(marked_img,'corner.jpg');

% centr = regionprops(I,'centroid');
% centx = centr(255).Centroid(1);
% cout_netindex = cout(:,3) > 0;
% sum_cout = sum(cout(cout_netindex,2) .* cout(cout_netindex,4) / sum(cout(cout_netindex,4)));
% if centx > sum_cout
%     align = 1; %tail left
% else
%     align = 2; %tail right
% end



end


function [curve,curve_start,curve_end,curve_mode,cur_num]=extract_curve(BW,Gap_size)

%   Function to extract curves from binary edge map, if the endpoint of a
%   contour is nearly connected to another endpoint, fill the gap and continue
%   the extraction. The default gap size is 1 pixles.

curve = [];
curve_start = [];
curve_end = [];
curve_mode = [];
cur_num = [];

[L,W]=size(BW);
BW1=zeros(L+2*Gap_size,W+2*Gap_size);
BW_edge=zeros(L,W);
BW1(Gap_size+1:Gap_size+L,Gap_size+1:Gap_size+W)=BW;

cur_num=0;
[r,c]=find(BW1(:,:)==1);
warning off;
centr = regionprops(BW1,'centroid');
centx = int32(centr.Centroid(1));
warning on;
cenindex = find(c == centx);
if length(cenindex) < 1
    cenindex = find(r == centx);
end
r = r(cenindex);
c = c(cenindex);

while size(r,1)>0
    point=[r(1),c(1)];
    cur=point;
    BW1(point(1),point(2))=0;
    [I,J]=find(BW1(point(1)-Gap_size:point(1)+Gap_size,point(2)-Gap_size:point(2)+Gap_size)==1);
    while size(I,1)>0
        dist=(I-Gap_size-1).^2+(J-Gap_size-1).^2;
        [min_dist,index]=min(dist);
        point=point+[I(index),J(index)]-Gap_size-1;
        cur=[cur;point];
        BW1(point(1),point(2))=0;
        [I,J]=find(BW1(point(1)-Gap_size:point(1)+Gap_size,point(2)-Gap_size:point(2)+Gap_size)==1);
    end
    
    % Extract edge towards another direction
    point=[r(1),c(1)];
    cur2=point;
    BW1(point(1),point(2))=0;
    [I,J]=find(BW1(point(1)-Gap_size:point(1)+Gap_size,point(2)-Gap_size:point(2)+Gap_size)==1);
    while size(I,1)>0
        dist=(I-Gap_size-1).^2+(J-Gap_size-1).^2;
        [min_dist,index]=min(dist);
        point=point+[I(index),J(index)]-Gap_size-1;
        cur2=[point;cur2];
        BW1(point(1),point(2))=0;
        [I,J]=find(BW1(point(1)-Gap_size:point(1)+Gap_size,point(2)-Gap_size:point(2)+Gap_size)==1);
    end
    
    cur = [cur;cur2];
        
    if size(cur,1)>(size(BW,1)+size(BW,2))/25
        cur_num=cur_num+1;
        curve{cur_num}=cur-Gap_size;
    end
    [r,c]=find(BW1==1);
    
end

for i=1:cur_num
    cross_sum = 0;
    curve_length = size(curve{i}(:,:),1);
    for j = 1:(curve_length-1)
        x2 = curve{i}(j+1,1); x1 = curve{i}(j,1);
        y2 = curve{i}(j+1,2); y1 = curve{i}(j,2);
        vector_A = [(x2-x1),0,(y2-y1)];
        vector_B = [(x2 - centr.Centroid(2)), 0, (y2 - centr.Centroid(1))];
        cross_point = cross(vector_A,vector_B);
        cross_sum = cross_sum + sign(cross_point(2));
    end
    
    if cross_sum < 0
        curve{i}(:,:) = flipud(curve{i}(:,:));
    end
    
    
    curve_start(i,:)=curve{i}(1,:);
    curve_end(i,:)=curve{i}(size(curve{i},1),:);
    if (curve_start(i,1)-curve_end(i,1))^2+...
        (curve_start(i,2)-curve_end(i,2))^2<=32
        curve_mode(i,:)='loop';
    else
        curve_mode(i,:)='line';
    end
    
    BW_edge(curve{i}(:,1)+(curve{i}(:,2)-1)*L)=1;
end
% figure(4)
% imshow(~BW_edge)
% title('Edge map')
%saveas(gcf, ['4_' num2str(ceil(now*10000000))], 'png');
%imwrite(~BW_edge,'edge.jpg');

end


function [cont_K]=get_corner(curve,curve_start,curve_end,curve_mode,curve_num,BW,sig,Endpoint,C,T_angle)

corner_num=0;
cout=[];
cont_K = [];

if isempty(curve) || isempty(curve_num)
    return;
end

GaussianDieOff = .0001; 
pw = 1:30; 
ssq = sig*sig;
width = max(find(exp(-(pw.*pw)/(2*ssq))>GaussianDieOff));
if isempty(width)
    width = 1;  
end
t = (-width:width);
gau = exp(-(t.*t)/(2*ssq))/(2*pi*ssq); 
gau=gau/sum(gau);

maxcurve_index = 1;
maxcurve_num = length(curve{1});

for i = 2:curve_num
    if length(curve{i}) > maxcurve_num
        maxcurve_num = length(curve{i});
        maxcurve_index = i;
    end
end

for i=maxcurve_index;
    x=curve{i}(:,1);
    y=curve{i}(:,2);
    W=width;
    L=size(x,1);
    if L>W
        
        % Calculate curvature
        if curve_mode(i,:)=='loop'
            x1=[x(L-W+1:L);x;x(1:W)];
            y1=[y(L-W+1:L);y;y(1:W)];
        else
            x1=[ones(W,1)*2*x(1)-x(W+1:-1:2);x;ones(W,1)*2*x(L)-x(L-1:-1:L-W)];
            y1=[ones(W,1)*2*y(1)-y(W+1:-1:2);y;ones(W,1)*2*y(L)-y(L-1:-1:L-W)];
        end
       
        xx=conv(x1,gau);
        xx=xx(W+1:L+3*W);
        yy=conv(y1,gau);
        yy=yy(W+1:L+3*W);

        Xu=[xx(2)-xx(1) ; (xx(3:L+2*W)-xx(1:L+2*W-2))/2 ; xx(L+2*W)-xx(L+2*W-1)];
        Yu=[yy(2)-yy(1) ; (yy(3:L+2*W)-yy(1:L+2*W-2))/2 ; yy(L+2*W)-yy(L+2*W-1)];
        Xuu=[Xu(2)-Xu(1) ; (Xu(3:L+2*W)-Xu(1:L+2*W-2))/2 ; Xu(L+2*W)-Xu(L+2*W-1)];
        Yuu=[Yu(2)-Yu(1) ; (Yu(3:L+2*W)-Yu(1:L+2*W-2))/2 ; Yu(L+2*W)-Yu(L+2*W-1)];

        K2 = (Xu.*Yuu-Xuu.*Yu)./((Xu.*Xu+Yu.*Yu).^1.5);
        K2=K2(W+1:L+W);
        
        pos_index = find(K2>=1);
        neg_index = find(K2<1);
        K2(pos_index) = log(K2(pos_index));
        K2(neg_index) = -log(2-K2(neg_index));
        
        K3=K2-min(K2);
        K4=max(K2)-K2;
        Kdelta = (max(K2)-min(K2))/10;
               
        % Find curvature local maxima as corner candidates
        extremum=[];
        N=size(K2,1);
        n=0;
        Search=1;
        
        for j=1:N-1
            if (K2(j+1)-K2(j))*Search>0
                n=n+1;
                extremum(n)=j;  % In extremum, odd points is minima and even points is maxima
                Search=-Search;
            end
        end
        if mod(size(extremum,2),2)==0
            n=n+1;
            extremum(n)=N;
        end
        
        n=size(extremum,2);
        flag=ones(size(extremum));
  
        % Compare with adaptive local threshold to remove round corners
        for j=3:2:n-1
           
            [drop,index1]=min(K4(extremum(j):-1:extremum(j-1)));
            [drop,index2]=min(K4(extremum(j):extremum(j+1)));
            ROS=max(K4(extremum(j)-index1+1),K4(extremum(j)+index2-1));
            K_thre(j)=C*Kdelta+mean(ROS);
            if K4(extremum(j))<K_thre(j)
                flag(j)=0;
            end
        end
        

        for j=2:2:n
            [drop,index1]=min(K3(extremum(j):-1:extremum(j-1)));
            [drop,index2]=min(K3(extremum(j):extremum(j+1)));
            ROS=max(K3(extremum(j)-index1+1),K3(extremum(j)+index2-1));
            K_thre(j)=C*Kdelta+mean(ROS);
            if K3(extremum(j))<K_thre(j)
                flag(j)=0;
            end
        end
        
%         extremum_all = extremum(flag==1);
%         dista = zeros(length(extremum_all),1);
%         for ii = 2:length(extremum_all)-1
%             dista(ii) = abs(K3(extremum_all(ii-1)) - K3(extremum_all(ii))) + abs(K3(extremum_all(ii+1)) - K3(extremum_all(ii)));
%         end
%         [maxv, maxi] = max(dista);
%         distb = std(K3(extremum_all(max(1,maxi-1)) : extremum_all(maxi+1)));
%         cstd = distb/std(K3);

        extremum_neg=extremum(1:2:n-1);
        flag1=flag(1:2:n-1);
        extremum_neg=extremum_neg(flag1==1);

        extremum=extremum(2:2:n);
        flag2=flag(2:2:n);
        extremum=extremum(flag2==1);

        extremum=extremum(extremum>2 & extremum<=L-2);
        extremum_neg=extremum_neg(extremum_neg>2 & extremum_neg<=L-2);

        cont_K = zeros(L,4);
        cont_K(:,1)=xx(W+1:L+W);
        cont_K(:,2)=yy(W+1:L+W);        
        cont_K(:,3) = K3;
        cont_K(extremum,4) = 1;
        cont_K(extremum_neg,4) = -1;
        
        %add start & end point.
        extrem_all = find(cont_K(:,4)~=0);
        cont_K(1, 4) = -cont_K(extrem_all(1), 4);
        cont_K(end, 4) = -cont_K(extrem_all(end), 4);

%         extremum = find(cont_K(:,4)==1);
%         extremum_neg = find(cont_K(:,4)==-1);
%         
%         n=size(extremum,1);
%         m=size(extremum_neg,1);
%         
%         figure(3);
%         subplot(2,1,1);
%         plot(K3);
%         hold on;
%         for j=1:n     
%             plot(extremum(j), K3(extremum(j)), 'r*');
%         end
%         for j=1:m     
%             plot(extremum_neg(j), K3(extremum_neg(j)), 'r+');
%         end
%         plot(1:L, 0, 'g');
%         hold off;
%         
%         subplot(2,1,2);
%         plot(yy(W+1:L+W), -xx(W+1:L+W));
%         hold on;
%         text(curve_start(i,2),-curve_start(i,1), 'begin');
%         text(curve_end(i,2), -curve_end(i,1), 'end');        
% 
%         for j=1:n     
%             plot(yy(W+extremum(j)), -xx(W+extremum(j)), 'r*');
%         end
%         for j=1:m     
%             plot(yy(W+extremum_neg(j)), -xx(W+extremum_neg(j)), 'r+');
%         end
%         hold off;
        
%        saveas(gcf, ['8_' num2str(ceil(now*10000000))], 'png');

        
    end
end


end

function ang=curve_tangent(cur,center)

for i=1:2
    if i==1
        curve=cur(center:-1:1,:);
    else
        curve=cur(center:size(cur,1),:);
    end
    L=size(curve,1);
    
    if L>3
        if sum(curve(1,:)~=curve(L,:))~=0
            M=ceil(L/2);
            x1=curve(1,1);
            y1=curve(1,2);
            x2=curve(M,1);
            y2=curve(M,2);
            x3=curve(L,1);
            y3=curve(L,2);
        else
            M1=ceil(L/3);
            M2=ceil(2*L/3);
            x1=curve(1,1);
            y1=curve(1,2);
            x2=curve(M1,1);
            y2=curve(M1,2);
            x3=curve(M2,1);
            y3=curve(M2,2);
        end
        
        if abs((x1-x2)*(y1-y3)-(x1-x3)*(y1-y2))<1e-8  % straight line
            tangent_direction=angle(complex(curve(L,1)-curve(1,1),curve(L,2)-curve(1,2)));
        else
            % Fit a circle
            % http://mysite.verizon.net/res148h4j/zenosamples/zs_circle3pts.html
            x0 = 1/2*(-y1*x2^2+y3*x2^2-y3*y1^2-y3*x1^2-y2*y3^2+x3^2*y1+y2*y1^2-y2*x3^2-y2^2*y1+y2*x1^2+y3^2*y1+y2^2*y3)/(-y1*x2+y1*x3+y3*x2+x1*y2-x1*y3-x3*y2);
            y0 = -1/2*(x1^2*x2-x1^2*x3+y1^2*x2-y1^2*x3+x1*x3^2-x1*x2^2-x3^2*x2-y3^2*x2+x3*y2^2+x1*y3^2-x1*y2^2+x3*x2^2)/(-y1*x2+y1*x3+y3*x2+x1*y2-x1*y3-x3*y2);
            % R = (x0-x1)^2+(y0-y1)^2;

            radius_direction=angle(complex(x0-x1,y0-y1));
            adjacent_direction=angle(complex(x2-x1,y2-y1));
            tangent_direction=sign(sin(adjacent_direction-radius_direction))*pi/2+radius_direction;
        end
    
    else % very short line
        tangent_direction=angle(complex(curve(L,1)-curve(1,1),curve(L,2)-curve(1,2)));
    end
    direction(i)=tangent_direction*180/pi;
end
ang=abs(direction(1)-direction(2));

end

function img1=mark(img,x,y,w)

[M,N,C]=size(img);
img1=img;

if isa(img,'logical')
    img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:)=...
        (img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:)<1);
    img1(x-floor(w/2)+1:x+floor(w/2)-1,y-floor(w/2)+1:y+floor(w/2)-1,:)=...
        img(x-floor(w/2)+1:x+floor(w/2)-1,y-floor(w/2)+1:y+floor(w/2)-1,:);
else
    img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:)=...
        (img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:)<128)*255;
    img1(x-floor(w/2)+1:x+floor(w/2)-1,y-floor(w/2)+1:y+floor(w/2)-1,:)=...
        img(x-floor(w/2)+1:x+floor(w/2)-1,y-floor(w/2)+1:y+floor(w/2)-1,:);
end

end


function [I,C,T_angle,sig,H,L,Endpoint,Gap_size] = parse_inputs(varargin)

error(nargchk(0,8,nargin));

%Para=[1.5,162,3,0.35,0,1,1]; %Default experience value;
Para=[0.4,169,3,0.35,0,0,2]; %Default experience value;

if nargin>=2
    I=varargin{1};
    for i=2:nargin
        if size(varargin{i},1)>0
            Para(i-1)=varargin{i};
        end
    end
end

if nargin==1
    I=varargin{1};
end
    
if nargin==0 | size(I,1)==0
    [fname,dire]=uigetfile('*.bmp;*.jpg;*.gif','Open the image to be detected');
    I=imread([dire,fname]);
end

C=Para(1);
T_angle=Para(2);
sig=Para(3);
H=Para(4);
L=Para(5);
Endpoint=Para(6);
Gap_size=Para(7);

end